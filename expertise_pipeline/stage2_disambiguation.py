"""Stage 2: neural author disambiguation.

Each author mention on a Stage 1 candidate-pool paper is embedded as chars2vec(author_name) [50-dim]
concatenated with SPECTER2(title + abstract) [768-dim], passed through the NAND model (nand_model.py)
to get a 256-dim mention embedding, then clustered with dbscanc.py's constrained DBSCAN. Each
resulting cluster is a disambiguated identity; cluster_id == -1 means the mention didn't confidently
resolve into any cluster (noise).

Mentions are only ever compared within the same name block (surname + given-name initial) -- a
'J. Smith' mention is never compared against an 'R. Robertson' mention, since blocking already knows
they're different people. This isn't just an optimization: dbscanc.py's own name_compatible() never
checks surname equality at all (only given-name-initial and middle-initial), so without blocking
it's relying entirely on embedding distance to keep different surnames apart.

Unlike the earlier ORCID-filter version of this stage, this operates on every mention in the
candidate pool (not just ones with an already-confirmed ORCID) -- ORCID is only used later, as
ground truth for evaluation, not as an input to clustering.
"""
import re

import numpy as np
import pandas as pd

from . import nand_embeddings, nand_model
from .dbscanc import dbscanc

MIN_EPS = 0.020
MAX_EPS = 0.6
MAX_EXPECTED_BLOCK_SIZE = 95632  # calibrated against the training data's block-size distribution


def block_key(author_name):
    """given-name initial + surname, e.g. 'Smith, John A.' -> 'j.smith'.

    ADS names are 'Surname, Given ...' -- the surname is split on the comma, NOT on whitespace.
    Splitting on whitespace instead (an earlier bug here) took only the first surname TOKEN, which
    silently truncated every multi-word surname to its first word: 'de los Reyes', 'De Lotto', 'de
    Luca', and 'De Laurentis' all collapsed to the same 'l.de' block, merging unrelated people who
    only share a common surname prefix into one comparison pool.
    """
    if not isinstance(author_name, str):
        return '.'
    family_part, _, given_part = author_name.lower().partition(',')
    if not family_part.strip():
        return '.'
    family = re.sub(r'\W+', '', family_part)
    given_tokens = given_part.split()
    given_initial = re.sub(r'\W+', '', given_tokens[0])[:1] if given_tokens else ''
    return f'{given_initial}.{family}'


def adaptive_eps(block_size, min_eps=MIN_EPS, max_eps=MAX_EPS, max_expected_block_size=MAX_EXPECTED_BLOCK_SIZE):
    """Larger blocks (more same-initial mentions to tell apart) get a tighter radius; small blocks
    get a looser one, since there's little room for a false merge anyway. Log-scaled between
    max_eps (block_size == 1) and min_eps (block_size >= max_expected_block_size).

    max_expected_block_size sets where the curve saturates -- it needs to be in the same ballpark as
    the actual block sizes in whatever corpus this runs on, or the curve barely tightens across the
    range of block sizes that corpus actually has. The module default (95632) was calibrated against
    a different dataset's block-size distribution; on a corpus with, say, most blocks under a few
    hundred mentions, that default keeps eps close to max_eps (loose) almost everywhere, since a
    block of size 100-300 is still tiny relative to 95632 on a log scale -- exactly the kind of
    mismatch that let real same-block collisions get merged instead of split. Tune this per corpus
    (see evaluate.tune_adaptive_eps) rather than trusting the default to transfer.
    """
    scale = (np.log1p(block_size) - np.log1p(1)) / (np.log1p(max_expected_block_size) - np.log1p(1))
    eps = max_eps - scale * (max_eps - min_eps)
    return np.clip(eps, min_eps, max_eps)


def _cluster_one_block(block, block_embeddings, years, names, eps, min_samples, year_eps):
    """Module-level (not a closure) so it's picklable for ProcessPoolExecutor -- a nested function
    can't be sent to a worker process."""
    _, local_labels = dbscanc(block_embeddings, years, names, eps=eps, min_samples=min_samples, year_eps=year_eps)
    return block, local_labels


def build_orcid_lookup(author_records_df):
    """Pre-builds a (bibcode, author_position) -> orcid_id lookup, once, for reuse across many
    get_candidate_mentions() calls. author_records_df is often the entire ORCID-matched corpus, not
    scoped to any one query -- passing it straight into a per-query merge() rebuilds that join's
    hash index from scratch on every single call, which dominates runtime at a large corpus size
    (5s+ for a ~15M-row author_records_df in local testing) regardless of how few candidates that
    particular query has. Build this once (e.g. right after loading author_records_df) and pass it
    to get_candidate_mentions() via orcid_lookup= instead of author_records_df=.

    Deduplicated on (bibcode, author_position) first (keeping the first match) -- a Series with a
    duplicate MultiIndex raises on .reindex() rather than just picking one, and author_records.ipynb
    is already supposed to guarantee this is unique, but a confusing crash from a silently-violated
    assumption is worse than a cheap drop_duplicates here.
    """
    deduped = author_records_df.drop_duplicates(subset=['bibcode', 'author_position'], keep='first')
    return pd.Series(
        deduped['orcid_id'].to_numpy(),
        index=pd.MultiIndex.from_arrays([deduped['bibcode'], deduped['author_position']]),
    )


def get_candidate_mentions(candidate_bibcodes, papers_df, author_records_df=None, bibcode_to_row=None, orcid_lookup=None):
    """Explodes every candidate paper's OWN author/affiliation list into one row per mention -- the
    full author list from the retrieved papers themselves, not filtered down to only ORCID-matched
    authors. ORCID identity (where known) is attached afterward purely as ground truth for
    evaluation, via the optional author_records_df/orcid_lookup -- never as a filter on which
    mentions exist. This is what lets disambiguation/ranking be measured across the whole retrieved
    candidate pool, not just the subset that happened to already have a confirmed ORCID match.

    papers_df needs 'author'/'aff' list columns (build_tfidf_index.ipynb saves these). Author
    positions are assigned by list order (cumcount per bibcode) -- the same convention
    author_records.ipynb used when computing its own author_position from the same underlying ADS
    author list, which is what makes the ground-truth join below line up correctly.

    bibcode_to_row: optional, from stage1_retrieval.load_tfidf_index() -- when given, candidate rows
    are selected via a direct position lookup instead of `papers_df['bibcode'].isin(...)`, which
    scans every row of the FULL candidate corpus (millions of rows) just to find the ~hundreds/
    thousands of candidate bibcodes. Falls back to the scan if not provided (correct either way,
    just slower on a large corpus).

    orcid_lookup: optional, from build_orcid_lookup(author_records_df) -- when given, used directly
    instead of merging against author_records_df fresh. If not given but author_records_df is,
    it's built here as a fallback (correct either way, just slower -- see build_orcid_lookup).
    """
    candidate_bibcodes = list(set(candidate_bibcodes))
    columns = ['bibcode', 'title', 'abstract', 'year', 'author', 'aff']
    if bibcode_to_row is not None:
        rows = bibcode_to_row.reindex(candidate_bibcodes).dropna().astype(int)
        candidates = papers_df.iloc[rows.to_numpy()][columns].copy()
    else:
        candidates = papers_df[papers_df['bibcode'].isin(set(candidate_bibcodes))][columns].copy()

    mentions = candidates.explode(['author', 'aff'])
    # author_position from cumcount BEFORE dropping anything -- must reflect each mention's original
    # position in the paper's author list (matching author_records.ipynb's convention) for the
    # ground-truth join below to line up, even for a paper where a later slot gets dropped next.
    mentions['author_position'] = mentions.groupby('bibcode').cumcount()
    mentions = mentions.rename(columns={'author': 'author_name', 'aff': 'affiliation'})

    # Real ADS author lists can have null/blank slots (a missing name in that position) -- not a
    # real mention, so drop it rather than let a non-string author_name reach chars2vec/dbscanc
    # downstream (both assume a real string; a None here surfaced as a bare "KeyError: None" deep
    # in the embedding cache instead of a clear error at the source). No print here -- this runs
    # once per query, including inside evaluate_labels' per-label loop, where it would spam stdout.
    is_valid_name = mentions['author_name'].apply(lambda x: isinstance(x, str) and x.strip() != '')
    mentions = mentions[is_valid_name]

    if orcid_lookup is None and author_records_df is not None:
        orcid_lookup = build_orcid_lookup(author_records_df)

    if orcid_lookup is not None:
        keys = pd.MultiIndex.from_arrays([mentions['bibcode'], mentions['author_position']])
        mentions['orcid_id'] = orcid_lookup.reindex(keys).to_numpy()
    else:
        mentions['orcid_id'] = None

    mentions['identity_type'] = np.where(mentions['orcid_id'].notna(), 'orcid', 'name_string')

    return mentions.reset_index(drop=True)


def disambiguate_mentions(
    mentions, chars2vec_model, specter_tokenizer, specter_model, nand_encoder,
    min_samples=5, year_eps=3, min_eps=MIN_EPS, max_eps=MAX_EPS,
    max_expected_block_size=MAX_EXPECTED_BLOCK_SIZE, device='cpu', executor=None,
    paper_embedding_cache=None, author_embedding_cache=None,
):
    """Adds a 'cluster_id' column to `mentions`. -1 means noise (excluded from Stage 3 ranking --
    not confidently resolved to any identity). Requires `mentions` to have a clean 0..n-1 index
    (get_candidate_mentions already resets it) -- used directly as positions into the embedding
    arrays below.

    executor: an optional concurrent.futures.Executor (ProcessPoolExecutor recommended -- dbscanc's
    per-block work is CPU-bound pure Python, which won't parallelize under threads due to the GIL).
    Each name block's clustering is fully independent of every other block, so they're dispatched
    concurrently instead of in a sequential loop. Create the executor ONCE and reuse it across many
    queries (e.g. across an evaluate_labels run) -- spinning up a fresh process pool per query would
    eat into the savings. If executor is None, blocks are clustered sequentially (unchanged
    behavior).

    paper_embedding_cache / author_embedding_cache: optional {key: embedding} dicts, e.g. from
    nand_embeddings.precompute_paper_embeddings() / precompute_author_name_embeddings() run once
    across many queries -- candidate pools drawn from the same corpus overlap heavily across
    queries, so without a shared cache the same paper/name gets re-embedded through SPECTER2/
    chars2vec every time it shows up in a different query's pool. Any bibcode/name not found in a
    provided cache is embedded on the fly and the cache is updated in place, so a partially
    populated (or entirely absent) cache never breaks correctness, only speed.
    """
    if author_embedding_cache is not None:
        missing_names = [n for n in mentions['author_name'].unique() if n not in author_embedding_cache]
        if missing_names:
            author_embedding_cache.update(
                nand_embeddings.precompute_author_name_embeddings(missing_names, chars2vec_model)
            )
        author_embeddings = np.stack([author_embedding_cache[n] for n in mentions['author_name']])
    else:
        author_embeddings = nand_embeddings.embed_author_names(chars2vec_model, mentions['author_name'])

    # A paper's title+abstract is identical for every one of its authors -- embed each unique paper
    # once through SPECTER2, not once per mention. A paper with 8 authors would otherwise push the
    # same text through the transformer 8 times; at a large candidate pool this dominates runtime.
    unique_papers = mentions.drop_duplicates(subset=['bibcode'])[['bibcode', 'title', 'abstract']]
    if paper_embedding_cache is not None:
        missing_papers = unique_papers[~unique_papers['bibcode'].isin(paper_embedding_cache)]
        if len(missing_papers):
            missing_embeddings = nand_embeddings.embed_papers(
                specter_tokenizer, specter_model, missing_papers['title'], missing_papers['abstract'], device=device
            )
            paper_embedding_cache.update(dict(zip(missing_papers['bibcode'], missing_embeddings)))
        bibcode_to_paper_embedding = {b: paper_embedding_cache[b] for b in unique_papers['bibcode']}
    else:
        unique_paper_embeddings = nand_embeddings.embed_papers(
            specter_tokenizer, specter_model, unique_papers['title'], unique_papers['abstract'], device=device
        )
        bibcode_to_paper_embedding = dict(zip(unique_papers['bibcode'], unique_paper_embeddings))

    paper_embeddings = np.stack([bibcode_to_paper_embedding[b] for b in mentions['bibcode']])

    mention_embeddings = nand_model.encode_mentions(nand_encoder, author_embeddings, paper_embeddings, device=device)

    blocks = mentions['author_name'].map(block_key)
    block_groups = list(mentions.groupby(blocks))  # [(block, group_df), ...] -- computed once, reused below

    block_args = [
        (block, mention_embeddings[group.index], group['year'].tolist(), group['author_name'].tolist(),
         adaptive_eps(len(group), min_eps=min_eps, max_eps=max_eps,
                      max_expected_block_size=max_expected_block_size), min_samples, year_eps)
        for block, group in block_groups
    ]

    if executor is None:
        block_results = [_cluster_one_block(*args) for args in block_args]
    else:
        futures = [executor.submit(_cluster_one_block, *args) for args in block_args]
        block_results = [future.result() for future in futures]

    block_index = dict(block_groups)  # block -> group's row index, for writing labels back below
    cluster_ids = pd.Series(-1, index=mentions.index, dtype=object)
    for block, local_labels in block_results:
        cluster_ids.loc[block_index[block].index] = [
            f'{block}::{label}' if label != -1 else -1 for label in local_labels
        ]

    out = mentions.copy()
    out['cluster_id'] = cluster_ids
    return out


def singleton_noise(mentions):
    """Gives every noise mention (cluster_id == -1) its own unique cluster_id instead of leaving it
    excluded from downstream ranking -- for evaluations that only care about RANKING quality, not
    disambiguation quality. Stage 2's job is to merge same-person mentions together; a mention that
    didn't confidently join anyone else's cluster is still real evidence of one specific person's
    (or at least one specific paper's) expertise, and dropping it from ranking just because Stage 2
    wasn't confident enough to merge it with something else conflates "unmerged" with "irrelevant."

    Each noise mention becomes its own singleton cluster (one mention -> one one-paper "identity"),
    not merged with any other noise mention -- we don't know two different noise mentions are the
    same person (that's exactly what Stage 2 couldn't confirm), so treating each as its own identity
    is the conservative choice, same as how a real person with only one confirmed mention already
    gets their own one-paper cluster.

    DO NOT use this when reporting disambiguation metrics: cluster_id == -1 is
    disambiguation_metrics()'s entire mechanism for excluding "not confidently resolved" mentions
    from precision/recall (see its docstring). Applying this first would make every mention
    "measurable," silently invalidating any precision/recall/F1 computed on the result.
    """
    out = mentions.copy()
    is_noise = out['cluster_id'] == -1
    out.loc[is_noise, 'cluster_id'] = [f'noise::{i}' for i in out.index[is_noise]]
    return out
