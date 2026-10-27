"""chars2vec (author name) and SPECTER2 (title + abstract) embeddings for the NAND model.

The NAND model (nand_model.py) expects each mention as a single 818-dim vector:
chars2vec(author_name) [50-dim] concatenated with SPECTER2(title + abstract) [768-dim].
"""
import time
import hashlib
import json
import pickle
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

CHARS2VEC_MODEL_NAME = 'eng_50'  # 50-dim, matches the NAND model's 50 + 768 = 818 input
SPECTER2_BASE_MODEL = 'allenai/specter2_base'
SPECTER2_ADAPTER = 'allenai/specter2'
SPECTER2_MAX_LENGTH = 512


def _embedding_cache_fingerprint(keys):
    digest = hashlib.sha256()
    for key in sorted(keys):
        digest.update(str(key).encode('utf-8'))
        digest.update(b'\0')
    return digest.hexdigest()


def save_embedding_cache(cache, cache_dir):
    """Save a string-keyed embedding cache as one dense array plus key metadata."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    keys = sorted(cache)
    if not keys:
        raise ValueError('Cannot save an empty embedding cache')
    first = np.asarray(cache[keys[0]])
    embeddings_path = cache_dir / 'embeddings.npy'
    embeddings = np.lib.format.open_memmap(
        embeddings_path, mode='w+', dtype=np.float32, shape=(len(keys), first.shape[0])
    )
    chunk_size = 8192
    for start in range(0, len(keys), chunk_size):
        chunk_keys = keys[start:start + chunk_size]
        embeddings[start:start + len(chunk_keys)] = np.stack([cache[k] for k in chunk_keys])
    embeddings.flush()
    with open(cache_dir / 'keys.pkl', 'wb') as handle:
        pickle.dump(keys, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(cache_dir / 'metadata.json', 'w') as handle:
        json.dump({'count': len(keys), 'dim': int(first.shape[0]),
                   'fingerprint': _embedding_cache_fingerprint(keys)}, handle)
    return cache_dir


def load_embedding_cache(cache_dir, expected_keys=None):
    """Load a saved embedding cache and reject it if its key set is stale or incomplete."""
    cache_dir = Path(cache_dir)
    with open(cache_dir / 'metadata.json') as handle:
        metadata = json.load(handle)
    with open(cache_dir / 'keys.pkl', 'rb') as handle:
        keys = pickle.load(handle)
    if expected_keys is not None:
        expected_keys = set(expected_keys)
        if metadata['fingerprint'] != _embedding_cache_fingerprint(expected_keys):
            raise ValueError('Saved embedding cache does not match the requested key set')
    embeddings = np.load(cache_dir / 'embeddings.npy', mmap_mode='r')
    if embeddings.shape != (metadata['count'], metadata['dim']):
        raise ValueError('Saved embedding cache has inconsistent metadata')
    return {key: embeddings[i] for i, key in enumerate(keys)}


def load_chars2vec_model():
    import chars2vec  # deferred: heavy/optional dep, only needed once you actually embed names
    return chars2vec.load_model(CHARS2VEC_MODEL_NAME)


def embed_author_names(chars2vec_model, author_names):
    """author_names: list of str, e.g. 'Smith, John A.' -> (n, 50) array."""
    return chars2vec_model.vectorize_words(list(author_names))


def load_specter2_model(device='cpu'):
    from adapters import AutoAdapterModel  # deferred: heavy/optional dep
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(SPECTER2_BASE_MODEL)
    model = AutoAdapterModel.from_pretrained(SPECTER2_BASE_MODEL)
    model.load_adapter(SPECTER2_ADAPTER, source='hf', load_as='specter2', set_active=True)
    model.to(device)
    model.eval()
    return tokenizer, model


def embed_papers(tokenizer, model, titles, abstracts, device='cpu', batch_size=32):
    """titles, abstracts: list of str, same length -> (n, 768) array.

    SPECTER2's convention is title + [SEP] + abstract as the input text, embedding taken from the
    [CLS] token of the last hidden state.
    """
    def _text(s):
        return s if isinstance(s, str) else ''

    texts = [f'{_text(title)}{tokenizer.sep_token}{_text(abstract)}' for title, abstract in zip(titles, abstracts)]

    embeddings = []
    batch_starts = range(0, len(texts), batch_size)
    with torch.no_grad():
        for start in tqdm(batch_starts, desc='SPECTER2 embedding', total=len(batch_starts)):
            batch = texts[start:start + batch_size]
            inputs = tokenizer(
                batch, padding=True, truncation=True, max_length=SPECTER2_MAX_LENGTH, return_tensors='pt'
            ).to(device)
            output = model(**inputs)
            cls_embeddings = output.last_hidden_state[:, 0, :]
            embeddings.append(cls_embeddings.cpu().numpy())

    return np.concatenate(embeddings, axis=0)


def precompute_paper_embeddings(bibcodes, papers_df, specter_tokenizer, specter_model, device='cpu', batch_size=32):
    """Embeds every unique bibcode in `bibcodes` through SPECTER2 exactly once, in as few large
    batches as possible, and returns {bibcode: embedding}.

    Meant to be called once across a whole batch of queries (e.g. evaluate.evaluate_labels) rather
    than once per query -- candidate pools drawn from the same corpus overlap heavily across
    queries, so embedding on demand per query re-embeds the same paper many times, and does it in
    many small batches instead of fewer large ones.
    """
    t0 = time.time()
    unique_bibcodes = sorted(set(bibcodes))
    unique_papers = papers_df[papers_df['bibcode'].isin(unique_bibcodes)].drop_duplicates(subset=['bibcode'])
    embeddings = embed_papers(
        specter_tokenizer, specter_model, unique_papers['title'], unique_papers['abstract'],
        device=device, batch_size=batch_size,
    )
    print(f'precompute_paper_embeddings: {time.time() - t0:.1f}s -> {len(unique_papers)} unique papers')
    return dict(zip(unique_papers['bibcode'], embeddings))


def precompute_paper_embeddings_multi_gpu(
    bibcodes, papers_df, existing_tokenizer, existing_model,
    devices=('cuda:0', 'cuda:1'), batch_size=128,
):
    """Embed a paper union across multiple GPUs in parallel.

    ``existing_tokenizer``/``existing_model`` are used for the first device, avoiding a duplicate
    model load on the notebook's current GPU. Additional devices load their own SPECTER2 copy. The
    returned dict has the same format as ``precompute_paper_embeddings`` and is suitable for passing
    as the shared read-only paper cache to the multi-GPU evaluator.
    """
    t0 = time.time()
    unique_bibcodes = sorted(set(bibcodes))
    unique_papers = papers_df[papers_df['bibcode'].isin(unique_bibcodes)].drop_duplicates(subset=['bibcode'])
    if unique_papers.empty:
        return {}

    devices = tuple(devices)
    shards = np.array_split(unique_papers, min(len(devices), len(unique_papers)))

    def _embed_shard(shard_i):
        shard = shards[shard_i]
        device = devices[shard_i]
        if shard_i == 0:
            tokenizer, model = existing_tokenizer, existing_model
        else:
            tokenizer, model = load_specter2_model(device=device)
        embeddings = embed_papers(
            tokenizer, model, shard['title'], shard['abstract'], device=device, batch_size=batch_size,
        )
        return shard['bibcode'].tolist(), embeddings

    with ThreadPoolExecutor(max_workers=len(shards)) as pool:
        results = list(pool.map(_embed_shard, range(len(shards))))

    cache = {}
    for bibcodes_shard, embeddings in results:
        cache.update(zip(bibcodes_shard, embeddings))
    print(f'precompute_paper_embeddings_multi_gpu: {time.time() - t0:.1f}s -> '
          f'{len(cache)} unique papers across {len(shards)} GPUs')
    return cache


def precompute_author_name_embeddings(author_names, chars2vec_model):
    """Embeds every unique author name string through chars2vec exactly once, returns {name: embedding}."""
    t0 = time.time()
    unique_names = sorted({name for name in author_names if isinstance(name, str)})
    if not unique_names:
        return {}
    embeddings = embed_author_names(chars2vec_model, unique_names)
    print(f'precompute_author_name_embeddings: {time.time() - t0:.1f}s -> {len(unique_names)} unique names')
    return dict(zip(unique_names, embeddings))
