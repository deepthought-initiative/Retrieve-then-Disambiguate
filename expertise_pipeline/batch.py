"""Splitting a label list two levels deep: across SLURM array tasks, then across GPUs within one
task. Kept separate from run_eval_batch.py so the chunking math itself is testable without SLURM,
GPUs, or the neural models.
"""


def chunk_list(items, num_chunks):
    """Splits items into num_chunks contiguous, nearly-equal chunks (sizes differ by at most 1).
    Every item appears in exactly one chunk, order preserved."""
    n = len(items)
    base, remainder = divmod(n, num_chunks)
    chunks = []
    start = 0
    for i in range(num_chunks):
        size = base + (1 if i < remainder else 0)
        chunks.append(items[start:start + size])
        start += size
    return chunks


def get_task_chunk(labels, num_array_tasks, array_task_id, gpus_per_task, gpu_id):
    """The slice of `labels` one specific GPU worker (within one specific array task) should
    process -- split across array tasks first, then that task's share split across its GPUs.

    Splits by BIBCODE, not by raw label, and keeps every label for a given bibcode together in the
    same chunk -- evaluate_labels() now runs Stage 1/2/3 once per unique bibcode in whatever label
    list it's given (see evaluate.py), reusing that one result for every label sharing it. Splitting
    at the label level instead would let one paper's labels land in different chunks/workers, so its
    pipeline (retrieval, clustering, ranking) would get computed redundantly, once per worker it
    showed up in -- silently giving back part of that fix's savings at the batch level.
    """
    bibcode_order = list(dict.fromkeys(label['bibcode'] for label in labels))
    labels_by_bibcode = {}
    for label in labels:
        labels_by_bibcode.setdefault(label['bibcode'], []).append(label)

    task_bibcodes = chunk_list(bibcode_order, num_array_tasks)[array_task_id]
    gpu_bibcodes = chunk_list(task_bibcodes, gpus_per_task)[gpu_id]

    return [label for bibcode in gpu_bibcodes for label in labels_by_bibcode[bibcode]]
