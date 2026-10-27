"""Runs Stage 1 -> Stage 2 -> Stage 3 for one query paper."""
from . import stage1_retrieval as stage1
from . import stage2_disambiguation as stage2
from . import stage3_ranking as stage3


def find_experts(
    query_title, query_abstract, query_year,
    vectorizer, matrix, papers, bibcode_to_row, author_records_df,
    chars2vec_model, specter_tokenizer, specter_model, nand_encoder,
    top_n_papers=50, dbscan_min_samples=5, dbscan_year_eps=3,
    dbscan_min_eps=stage2.MIN_EPS, dbscan_max_eps=stage2.MAX_EPS,
    dbscan_max_expected_block_size=stage2.MAX_EXPECTED_BLOCK_SIZE, device='cpu', executor=None,
    paper_embedding_cache=None, author_embedding_cache=None,
    similarity_executor=None, n_similarity_chunks=8, candidates=None, orcid_lookup=None,
    treat_noise_as_singletons=False,
):
    """query_year is required: candidate papers are restricted to year < query_year to avoid
    temporal leakage. `papers` is the Stage 1 index's paper table (bibcode, title, abstract, year,
    author, aff) -- Stage 2 explodes its own author/aff lists to build the candidate pool's
    mentions. author_records_df (author_records.ipynb's `author_records`) is used only afterward, as
    a ground-truth ORCID lookup for evaluation -- it never filters which mentions Stage 2 sees.

    treat_noise_as_singletons: if True, every noise mention (cluster_id == -1) is given its own
    singleton cluster (stage2.singleton_noise()) before Stage 3, so it still gets ranked as a
    one-paper "identity" instead of being silently excluded. Use this for evaluations that only
    care about RANKING quality -- it makes 'mentions'/'ranking' below internally inconsistent with
    disambiguation_metrics() (which relies on cluster_id == -1 to mean "not confidently resolved"),
    so leave this False whenever disambiguation quality is also being reported from the same result.

    orcid_lookup: an optional pre-built stage2.build_orcid_lookup(author_records_df) result. If
    given, used directly instead of rebuilding it from author_records_df on every call -- that
    rebuild scales with the size of the ENTIRE author_records_df (potentially the whole ORCID-
    matched corpus), not with this query's tiny candidate pool, and dominates runtime at scale.
    Build it once and reuse it across queries.

    similarity_executor: a ThreadPoolExecutor for Stage 1's cosine similarity computation -- distinct
    from `executor` (a ProcessPoolExecutor for Stage 2's clustering); different stages, different
    tools, don't mix them up.

    candidates: an optional pre-computed Stage 1 result (as returned by stage1.find_similar_papers)
    for this exact query. If given, Stage 1 is skipped entirely instead of recomputing it -- e.g.
    reuse a `candidates_by_bibcode` cache built once for a batch of queries, rather than paying for
    the same similarity computation again here.

    Returns a dict with keys:
      'candidates' -- similar papers found (bibcode, similarity_score)
      'mentions'   -- every candidate-pool author mention, with resolved cluster_id (-1 = noise)
      'ranking'    -- one row per disambiguated identity, ranked by expertise_score
    """
    if candidates is None:
        candidates = stage1.find_similar_papers(
            query_title, query_abstract, query_year, vectorizer, matrix, papers, top_n=top_n_papers,
            similarity_executor=similarity_executor, n_similarity_chunks=n_similarity_chunks,
        )

    mentions = stage2.get_candidate_mentions(
        candidate_bibcodes=candidates['bibcode'],
        papers_df=papers,
        author_records_df=author_records_df,
        bibcode_to_row=bibcode_to_row,
        orcid_lookup=orcid_lookup,
    )
    if mentions.empty:
        # No candidate papers (e.g. a query paper older than everything in the index) or no author
        # mentions on the ones found -- nothing to embed/cluster/rank, so stop here rather than
        # feeding empty batches into chars2vec/SPECTER2, which don't handle that gracefully.
        empty_mentions = mentions.assign(cluster_id=[])
        empty_ranking = stage3.rank_experts(
            query_title, query_abstract, stage3.build_expertise_documents(empty_mentions)
        )
        return {'candidates': candidates, 'mentions': empty_mentions, 'ranking': empty_ranking}

    disambiguated = stage2.disambiguate_mentions(
        mentions, chars2vec_model, specter_tokenizer, specter_model, nand_encoder,
        min_samples=dbscan_min_samples, year_eps=dbscan_year_eps,
        min_eps=dbscan_min_eps, max_eps=dbscan_max_eps,
        max_expected_block_size=dbscan_max_expected_block_size, device=device, executor=executor,
        paper_embedding_cache=paper_embedding_cache, author_embedding_cache=author_embedding_cache,
    )
    if treat_noise_as_singletons:
        disambiguated = stage2.singleton_noise(disambiguated)

    expertise_documents = stage3.build_expertise_documents(disambiguated)
    ranking = stage3.rank_experts(query_title, query_abstract, expertise_documents)

    return {'candidates': candidates, 'mentions': disambiguated, 'ranking': ranking}
