"""Stage 1: find papers similar to a query paper using TF-IDF cosine similarity.

The TF-IDF vectorizer and matrix are built once (see build_tfidf_index.ipynb) and saved to disk.
find_similar_papers() only vectorizes the single query paper and compares it against the saved
matrix -- it never re-vectorizes the corpus.
"""
import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity

VECTORIZER_FILE = 'tfidf_vectorizer.joblib'
MATRIX_FILE = 'tfidf_matrix.npz'
PAPERS_FILE = 'tfidf_papers.parquet'


def load_tfidf_index(index_dir):
    """Loads the vectorizer, matrix, and paper metadata saved by build_tfidf_index.ipynb.

    Sorts papers/matrix by year (ascending) once, here, rather than once per query -- this is what
    lets find_similar_papers() find its "year < query_year" cutoff with a binary search and a
    contiguous slice, instead of a boolean-mask gather over the whole matrix on every single call
    (which, at a multi-million-row corpus, was taking minutes per query on its own -- the actual
    cosine similarity computation was never the bottleneck).

    matrix row i corresponds to papers.iloc[i]. papers has columns [bibcode, year], sorted by year.

    Also returns bibcode_to_row, a bibcode -> matrix row index lookup, built once here rather than
    per query -- Stage 3 needs it to pull specific candidate papers' TF-IDF rows out of the matrix.
    """
    vectorizer = joblib.load(f'{index_dir}/{VECTORIZER_FILE}')
    matrix = sparse.load_npz(f'{index_dir}/{MATRIX_FILE}')
    papers = pd.read_parquet(f'{index_dir}/{PAPERS_FILE}')

    sort_order = np.argsort(papers['year'].to_numpy(), kind='stable')
    papers = papers.iloc[sort_order].reset_index(drop=True)
    matrix = matrix[sort_order]

    bibcode_to_row = pd.Series(range(len(papers)), index=papers['bibcode'])
    return vectorizer, matrix, papers, bibcode_to_row


def load_paper_metadata(index_dir):
    """Lighter alternative to load_tfidf_index() for callers that already have Stage 1 retrieval
    results precomputed elsewhere (e.g. a shared candidates_cache built once centrally) and only need
    paper metadata (title/abstract/year/author/aff) + the bibcode -> row lookup to build mentions
    from a given set of candidate bibcodes -- never touches the vectorizer or TF-IDF matrix files at
    all. Those two are the expensive part (a large sparse matrix, plus a slow single-threaded
    row-reorder to sort by year) -- fine to pay once in one process, very much not fine to pay
    redundantly in every one of several worker processes that will never actually run retrieval
    themselves (measured: ~30GB RES and 400s+ per worker just to load+reorder it, before any actual
    work even starts -- see multi_gpu_eval.py's worker, which uses this instead).

    Same sort-by-year + bibcode_to_row construction as load_tfidf_index(), so bibcode_to_row values
    from the two loaders are consistent with each other (both derive from the same source file and
    sort order) even though this one has no matrix to keep index-aligned with it.
    """
    papers = pd.read_parquet(f'{index_dir}/{PAPERS_FILE}')
    sort_order = np.argsort(papers['year'].to_numpy(), kind='stable')
    papers = papers.iloc[sort_order].reset_index(drop=True)
    bibcode_to_row = pd.Series(range(len(papers)), index=papers['bibcode'])
    return papers, bibcode_to_row


def _cosine_similarity_chunk(query_vector, matrix_chunk):
    """Module-level (not a closure) so it's a stable target for ThreadPoolExecutor.submit."""
    return cosine_similarity(query_vector, matrix_chunk)[0]


def find_similar_papers(
    query_title, query_abstract, query_year, vectorizer, matrix, papers, top_n=50,
    similarity_executor=None, n_similarity_chunks=8,
):
    """Returns the top_n most similar papers to the query, as a DataFrame with
    columns [bibcode, similarity_score], sorted most similar first.

    Only considers papers published strictly before query_year, so a query paper can never retrieve
    itself or anything published after it -- required to avoid temporal leakage when evaluating
    against ground-truth authorship/citation labels.

    Requires `papers`/`matrix` sorted by year ascending (load_tfidf_index() guarantees this) -- the
    "year < query_year" cutoff is found via binary search and taken as a contiguous prefix slice,
    which is far cheaper than re-scanning a boolean mask over the whole corpus per query.

    similarity_executor: an optional concurrent.futures.ThreadPoolExecutor (threads, not processes
    -- scipy's sparse dot product releases the GIL during the actual computation, so threads give
    real parallelism here with none of ProcessPoolExecutor's cost of pickling multi-million-row
    matrix chunks to worker processes on every single query). Each candidate paper's similarity
    score is independent of every other's, so the eligible matrix is split into n_similarity_chunks
    row-chunks and computed concurrently instead of as one large sequential call. Create the
    executor ONCE and reuse it across queries. If None (default), computed sequentially in one call.
    """
    years = papers['year'].to_numpy()
    cutoff = np.searchsorted(years, query_year, side='left')  # first index with year >= query_year

    if cutoff == 0:
        # e.g. a query paper older than everything in the index (the TF-IDF corpus only covers the
        # last 15 years) -- cosine_similarity errors on a 0-row matrix instead of just returning
        # nothing, so short-circuit here rather than letting that surface as a confusing crash.
        return pd.DataFrame(columns=['bibcode', 'similarity_score'])

    eligible_matrix = matrix[:cutoff]
    eligible_bibcodes = papers['bibcode'].to_numpy()[:cutoff]

    query_text = f'{query_title} {query_abstract}'
    query_vector = vectorizer.transform([query_text])

    if similarity_executor is None or cutoff < n_similarity_chunks:
        similarity_scores = cosine_similarity(query_vector, eligible_matrix)[0]
    else:
        chunk_bounds = np.linspace(0, cutoff, n_similarity_chunks + 1, dtype=int)
        futures = [
            similarity_executor.submit(
                _cosine_similarity_chunk, query_vector, eligible_matrix[chunk_bounds[i]:chunk_bounds[i + 1]]
            )
            for i in range(n_similarity_chunks)
        ]
        similarity_scores = np.concatenate([future.result() for future in futures])

    # Selecting the top k with argpartition avoids sorting every eligible paper. This matters for
    # a large corpus because each query otherwise performs a full O(n log n) sort even though the
    # caller only needs the first top_n rows.
    n_keep = min(top_n, len(similarity_scores))
    if n_keep == 0:
        return pd.DataFrame(columns=['bibcode', 'similarity_score'])
    top_positions = np.argpartition(-similarity_scores, n_keep - 1)[:n_keep]
    top_positions = top_positions[np.argsort(-similarity_scores[top_positions], kind='stable')]
    return pd.DataFrame({
        'bibcode': eligible_bibcodes[top_positions],
        'similarity_score': similarity_scores[top_positions],
    }).reset_index(drop=True)
