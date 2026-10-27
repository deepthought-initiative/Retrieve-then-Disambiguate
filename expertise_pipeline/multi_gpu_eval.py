"""Runs evaluate_labels() across multiple GPU worker PROCESSES from a live Python session (e.g. a
notebook) -- same one-process-per-GPU pattern as run_eval_batch.py's SLURM path (CUDA contexts don't
share across processes, so each worker loads its own chars2vec/SPECTER2/NAND model copies), but
callable directly instead of through argparse/$SLURM_ARRAY_TASK_ID, and supporting MULTIPLE worker
processes per physical GPU (repeat a device string in `devices`), for GPUs with enough spare memory
to hold 2+ model copies at once.

Every read-heavy, worker-independent piece of setup -- Stage 1 retrieval, the TF-IDF vectorizer/
matrix, paper metadata, the ORCID ground-truth lookup -- is loaded/computed exactly ONCE by the
caller (typically already sitting in the calling notebook's own variables from its own loading
cells) and handed to workers as plain in-memory arguments, never re-read from disk per worker.
Measured cost of the naive "every worker loads its own copy" approach: ~30GB RES and 400s+ PER
WORKER just to load and reorder the TF-IDF matrix alone, for work that only ever needs to happen
once regardless of worker count. Only the neural models (chars2vec/SPECTER2/NAND) are genuinely
per-worker, since CUDA contexts don't share across processes. See run_multi_gpu_eval's and
_run_worker's docstrings for the full design.

GPU memory is released by PROCESS EXIT, not by anything explicit in this module: each worker's `with
ProcessPoolExecutor(...)` block finishes, the worker function returns, and the worker process itself
terminates -- at that point its CUDA context (and everything in it: model weights, activations,
cached allocator memory) is gone. run_multi_gpu_eval() joins every worker before returning, so by the
time it hands control back, every GPU this run touched is back to whatever it looked like before the
call -- safe to immediately start a different experiment (e.g. a different top_n_papers) afterward
without manual del/empty_cache.
"""
import multiprocessing
import os
import queue
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

# Set BEFORE any TensorFlow-importing module loads (nand_embeddings -> chars2vec pulls it in) --
# each worker process is a fresh interpreter under spawn, so TF re-does its noisy C++ startup
# logging (cuDNN/cuFFT/cuBLAS "already registered" lines) once per worker. Multiplied by several
# concurrent workers, this -- plus per-worker tqdm bars (silenced below in _run_worker) -- is enough
# raw stdout volume to trip Jupyter's "IOStream.flush timed out" warning. '3' = errors only.
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

import pandas as pd
from tqdm.auto import tqdm as _tqdm_cls

from . import nand_embeddings, nand_model
from . import evaluate as evaluate_module
from .batch import chunk_list
from .evaluate import evaluate_labels


class _DisabledTqdm(_tqdm_cls):
    """Same tqdm class, forced to disable=True -- keeps the full API evaluate.py actually uses
    (iteration, .set_postfix() on the returned instance, the classmethod .write()) intact, unlike
    swapping in a bare lambda (which would break both of those). See _run_worker's docstring for why
    this gets patched into evaluate.py's module namespace for the duration of a worker process."""
    def __init__(self, *args, **kwargs):
        kwargs['disable'] = True
        super().__init__(*args, **kwargs)


def _run_worker(
    device, worker_labels, worker_candidates_cache, index_papers, bibcode_to_row, orcid_lookup,
    query_papers, top_n_papers, dbscan_min_eps, dbscan_max_eps, dbscan_max_expected_block_size,
    treat_noise_as_singletons, cpu_workers_per_gpu, n_similarity_threads,
    paper_embedding_cache, author_embedding_cache, result_queue, worker_id,
):
    """Runs in its own process, pinned to `device`. If `device` is shared with another worker (e.g.
    ['cuda:0', 'cuda:0', 'cuda:1', 'cuda:1']), each still loads a FULL SEPARATE copy of SPECTER2 +
    NAND into that same GPU's memory -- check `!nvidia-smi` for free memory before running more than
    one worker per physical GPU; there's no guard here against a worker OOMing.

    Deliberately does NOT run Stage 1 retrieval itself, does NOT load the TF-IDF vectorizer/matrix,
    and does NOT re-read the paper-metadata/author-records files from disk at all -- everything that
    would otherwise be redundant per-worker I/O (`index_papers`, `bibcode_to_row`, `orcid_lookup`,
    and this worker's slice of `worker_candidates_cache`) is loaded/computed ONCE by the caller, in
    the notebook's own process, and passed straight in here as plain in-memory objects. This matters
    a lot in practice, not just in theory: an earlier version where every worker independently loaded
    the TF-IDF matrix alone measured ~30GB RES and 400s+ PER WORKER, for work that only ever needs to
    happen once regardless of how many workers exist.

    The one thing that genuinely CANNOT be shared this way is the neural models (chars2vec/SPECTER2/
    NAND) below -- each worker is a separate OS process with its own CUDA context, and a loaded
    PyTorch model doesn't transfer across a process boundary via normal object passing. That's not
    an oversight; it's the actual reason this uses multiprocessing at all (real parallelism past the
    GIL) -- the unavoidable tradeoff is every worker loads its own model copy, which is comparatively
    cheap (seconds) next to the matrix problem above.

    Prints with `flush=True` at every setup step -- without these, model loading (the one genuinely
    per-worker-necessary cost) is silent from the calling notebook's point of view, indistinguishable
    from a hang.

    cpu_workers_per_gpu / n_similarity_threads are typically set to (or near) the FULL core count,
    not divided by the number of GPU workers -- see run_multi_gpu_eval's docstring for why
    oversubscribing here is intentional, not an oversight. (n_similarity_threads mostly goes unused
    now that retrieval is precomputed -- kept for the rare candidates_cache miss, and for API
    symmetry with evaluate_labels' other callers.)
    """
    tag = f'[worker {worker_id} pid={os.getpid()} device={device}]'
    t0 = time.time()
    print(f'{tag} starting -- loading models onto {device}...', flush=True)

    # evaluate_labels() renders its own tqdm bars ("Pre-pass...", "Evaluating query papers...")
    # unconditionally -- fine for a single-process notebook cell, but with several of these worker
    # processes all re-rendering progress bars into the SAME inherited stdout concurrently, the
    # sheer update volume is enough to trip Jupyter's "IOStream.flush timed out" warning. This
    # worker's own tag-prefixed prints above/below already tell the notebook what's happening, so
    # evaluate_labels' internal bars are pure noise here -- patched to _DisabledTqdm (still a real
    # tqdm instance, just silent) rather than removed outright, since evaluate.py also relies on
    # tqdm.write() (a classmethod) and progress.set_postfix() (an instance method).
    evaluate_module.tqdm = _DisabledTqdm

    chars2vec_model = nand_embeddings.load_chars2vec_model()
    specter_tokenizer, specter_model = nand_embeddings.load_specter2_model(device=device)
    nand_encoder = nand_model.load_nand_model(device=device)
    print(f'{tag} models loaded ({time.time() - t0:.1f}s) -- starting evaluate_labels on '
          f'{len(worker_labels)} label records...', flush=True)

    # Own small CPU pool for Stage 2's per-name-block clustering, same pattern as
    # run_eval_batch.py's per-GPU worker -- spawn context required, this process already has CUDA
    # initialized (models loaded above) by the time this pool would otherwise default to fork.
    # similarity_executor only matters on a candidates_cache miss now (see docstring) -- kept anyway
    # since it costs nothing idle and evaluate_labels accepts it unconditionally.
    cpu_ctx = multiprocessing.get_context('spawn')
    with ProcessPoolExecutor(max_workers=cpu_workers_per_gpu, mp_context=cpu_ctx) as cpu_executor, \
         ThreadPoolExecutor(max_workers=n_similarity_threads) as similarity_executor:
        rank_results, _disambiguation_results = evaluate_labels(
            worker_labels, query_papers, None, None, index_papers, bibcode_to_row,
            None, chars2vec_model, specter_tokenizer, specter_model, nand_encoder,
            top_n_papers=top_n_papers, device=device, executor=cpu_executor, orcid_lookup=orcid_lookup,
            similarity_executor=similarity_executor, n_similarity_chunks=n_similarity_threads,
            candidates_cache=worker_candidates_cache,
            dbscan_min_eps=dbscan_min_eps, dbscan_max_eps=dbscan_max_eps,
            dbscan_max_expected_block_size=dbscan_max_expected_block_size,
            treat_noise_as_singletons=treat_noise_as_singletons,
            paper_embedding_cache=paper_embedding_cache,
            author_embedding_cache=author_embedding_cache,
            progress_every=10,
            progress_label=tag,
        )

    print(f'{tag} done ({time.time() - t0:.1f}s total) -- {len(rank_results)} rank rows', flush=True)
    result_queue.put((worker_id, rank_results))


def run_multi_gpu_eval(
    labels, query_papers, index_papers, bibcode_to_row, candidates_cache, orcid_lookup, devices,
    top_n_papers, dbscan_min_eps, dbscan_max_eps, dbscan_max_expected_block_size,
    treat_noise_as_singletons=True, cpu_workers_per_gpu=4, n_similarity_threads=None,
    paper_embedding_cache=None, author_embedding_cache=None,
):
    """Splits `labels` by BIBCODE (keeping each paper's labels together, via batch.chunk_list --
    evaluate_labels() runs Stage 1/2/3 once per bibcode, so splitting at the label level instead
    could put one paper's labels on two different workers, silently doubling that paper's work)
    across len(devices) worker processes -- one process per entry in `devices`. Repeat a device
    string to run multiple workers on the same physical GPU, e.g.
    ['cuda:0', 'cuda:0', 'cuda:1', 'cuda:1'] for 2 workers each on 2 GPUs (4 papers being actively
    processed at once, one per worker, across the fleet).

    index_papers / bibcode_to_row / orcid_lookup: all loaded/built exactly ONCE by the caller (e.g.
    this notebook's own `stage1.load_tfidf_index(...)` and `stage2.build_orcid_lookup(...)` cells,
    already run at the top) and passed straight through here as plain in-memory objects -- no worker
    re-reads any of these from disk. Same reasoning as candidates_cache below: cheap once, wasteful
    to redo identically in every worker process.

    candidates_cache: REQUIRED, a {bibcode: Stage 1 result} dict already covering every bibcode in
    `labels`, computed ONCE by the caller (e.g. one threaded pass over query_papers using the
    vectorizer/matrix loaded in the calling process) at this exact top_n_papers value -- workers
    never run Stage 1 retrieval themselves and never load the TF-IDF vectorizer/matrix at all (see
    _run_worker's docstring for why: doing that per worker measured ~30GB RES and 400s+ PER WORKER
    just to load and reorder the matrix, for work that only ever needs to happen once regardless of
    how many workers exist). Sliced here per worker's bibcodes before being sent to that process.

    cpu_workers_per_gpu / n_similarity_threads: sized PER WORKER, not divided across len(devices) by
    this function -- pass the caller's judgment call directly. Giving every worker access to (near)
    the FULL core count, rather than cpu_count/len(devices) each, is deliberate: the 4 workers are
    rarely all in their CPU-bound phase (Stage 2 clustering, mainly, now that retrieval is
    precomputed) at the same moment -- at any given instant some are mid-GPU-embedding instead -- so
    a strict split leaves cores idle in a GPU-busy worker while a CPU-busy worker is starved. The
    risk is the opposite case, all 4 workers hitting their CPU-bound phase simultaneously and
    oversubscribing the machine -- the OS scheduler time-shares this reasonably gracefully (idle
    process-pool workers cost little beyond memory), but if you see thrashing (system load far above
    core count, without a matching wall-clock speedup), dial these back down.

    Returns one combined rank_results DataFrame (concatenated across all workers). Does NOT return
    disambiguation_results -- discarded per-worker (see evaluate_labels' own docstring on why it's
    unused for this project right now); recombine that too if disambiguation reporting ever comes
    back into scope.
    """
    n_similarity_threads = n_similarity_threads or cpu_workers_per_gpu
    bibcode_order = list(dict.fromkeys(label['bibcode'] for label in labels))
    labels_by_bibcode = {}
    for label in labels:
        labels_by_bibcode.setdefault(label['bibcode'], []).append(label)

    bibcode_chunks = chunk_list(bibcode_order, len(devices))
    label_chunks = [
        [label for bibcode in chunk for label in labels_by_bibcode[bibcode]]
        for chunk in bibcode_chunks
    ]
    # Each worker only gets ITS slice of candidates_cache, not the whole thing -- keeps the
    # pickled/IPC-transferred payload per worker small (a handful of small DataFrames, not the
    # entire experiment's worth), and makes a missing bibcode in the cache loud (KeyError inside the
    # worker, caught by the dead-worker check below) rather than silently falling back to a
    # never-loaded vectorizer/matrix (which would crash anyway, just less clearly).
    candidates_cache_chunks = [
        {bibcode: candidates_cache[bibcode] for bibcode in chunk} for chunk in bibcode_chunks
    ]

    # spawn (not fork) is required for CUDA -- see run_eval_batch.py's identical reasoning.
    ctx = multiprocessing.get_context('spawn')
    result_queue = ctx.Queue()
    workers = {}
    for worker_id, (device, worker_labels, worker_candidates_cache) in enumerate(
        zip(devices, label_chunks, candidates_cache_chunks)
    ):
        if not worker_labels:
            continue
        worker = ctx.Process(
            target=_run_worker,
            args=(
                device, worker_labels, worker_candidates_cache, index_papers, bibcode_to_row,
                orcid_lookup, query_papers, top_n_papers, dbscan_min_eps, dbscan_max_eps,
                dbscan_max_expected_block_size, treat_noise_as_singletons, cpu_workers_per_gpu,
                n_similarity_threads, paper_embedding_cache, author_embedding_cache,
                result_queue, worker_id,
            ),
        )
        worker.start()
        workers[worker_id] = worker

    # Polling result_queue.get() with a timeout, not a plain blocking get() -- a worker that dies
    # BEFORE reaching result_queue.put() (uncaught exception during model load, CUDA init failure on
    # a bad device string, OOM-killed by the OS, etc.) would otherwise leave this loop waiting
    # forever for a message that will never arrive, hanging the whole experiment silently -- and the
    # caller's try/except around run_top_n_experiment() never gets a chance to catch it and move on
    # to the next value, since nothing ever raises. Checking is_alive() on every still-pending worker
    # every 30s catches that: a dead worker with no result is a real failure, raised here explicitly.
    rank_results_parts = []
    pending = set(workers)
    while pending:
        try:
            worker_id, rank_results = result_queue.get(timeout=30)
        except queue.Empty:
            dead = [wid for wid in pending if not workers[wid].is_alive()]
            if dead:
                for wid in dead:
                    workers[wid].join()  # reap the zombie, exitcode becomes available
                raise RuntimeError(
                    f'Worker(s) {dead} (devices {[devices[wid] for wid in dead]}, '
                    f'exit codes {[workers[wid].exitcode for wid in dead]}) died without producing '
                    f'a result -- check stderr above for the actual traceback from that process.'
                )
            continue
        rank_results_parts.append(rank_results)
        pending.discard(worker_id)
        print(f'  worker {worker_id} ({devices[worker_id]}): {len(rank_results)} rank rows')

    for worker in workers.values():
        worker.join()

    return pd.concat(rank_results_parts, ignore_index=True)
