"""Evaluate both the ranking and the disambiguation against ground truth.

Ranking: for each label (a query paper + its true author, coauthors, and cited authors), runs the
pipeline with that paper's own year as the temporal cutoff, then reports what rank each evidence
type's orcid_id(s) landed at in the resulting cluster ranking. A good ranking should put authorship,
coauthorship, and citation-linked people near the top; comparing their rank distributions is how we
tell which evidence type the ranking is best/worst at surfacing.

An orcid_id maps to a rank via whichever cluster(s) its mentions ended up in (Stage 2's clustering
output), not directly -- if that orcid_id's mentions got split across multiple clusters (a
disambiguation error), the best (lowest) rank among them is used, which is optimistic and can hide
fragmentation. That's exactly what disambiguation_metrics() below checks directly.

Disambiguation: mentions with a known orcid_id (identity_type == 'orcid') are ground truth for
Stage 2's clustering itself, independent of ranking -- pairwise precision/recall/F1 of cluster_id
against orcid_id, scoped to one query's candidate pool at a time (clustering runs fresh per query,
so the same person's mentions can land in different clusters across different queries).
"""
import time
import warnings

import pandas as pd
from sklearn.metrics.cluster import pair_confusion_matrix
from tqdm.auto import tqdm

from . import nand_embeddings
from . import stage1_retrieval as stage1
from . import stage2_disambiguation as stage2
from .pipeline import find_experts


def rank_of(orcid_id, ranking, disambiguated_mentions):
    """Best (lowest) rank among the clusters containing a mention with this orcid_id, or None if
    none of its mentions resolved into any cluster (all noise, or not in the candidate pool)."""
    member_clusters = disambiguated_mentions.loc[
        (disambiguated_mentions['orcid_id'] == orcid_id) & (disambiguated_mentions['cluster_id'] != -1),
        'cluster_id',
    ].unique()
    if len(member_clusters) == 0:
        return None
    ranks = ranking.loc[ranking['cluster_id'].isin(member_clusters), 'rank']
    return int(ranks.min()) if len(ranks) else None


def diagnose_missing_candidate(candidate_id, query_year, candidates, mentions, author_records_df):
    """Explains WHY rank_of() came back None for a given (candidate_id, query) pair -- that happens
    whenever none of candidate_id's mentions resolved into a cluster, which can mean three
    structurally different things:

    'cold_start' -- this person has no papers in the corpus published before query_year at all.
    There's nothing earlier for Stage 1 to retrieve, no matter how good retrieval is -- not a
    pipeline failure, just an inherent limit of judging expertise from prior publication history
    when there isn't any yet (a legitimate case for a first-time author, but also happens for anyone
    whose only corpus papers happen to be recent, e.g. right at the edge of the loaded NUM_YEARS
    window).

    'retrieval_miss' -- they DO have earlier papers in the corpus, but Stage 1 didn't retrieve any of
    them into THIS query's candidate pool -- its TF-IDF cosine similarity didn't judge their prior
    work close enough to the query paper (or `top_n_papers` wasn't large enough to reach it). Worth
    checking similarity_score for their nearest miss if this comes up often.

    'noise' -- at least one of their earlier papers WAS retrieved (their mentions appear in
    `mentions`), but Stage 2's clustering marked every one of those mentions as noise
    (cluster_id == -1) -- a genuine disambiguation failure (too few same-block neighbors within eps,
    or a name block dominated by other people's mentions), not a retrieval gap.

    'resolved' -- their mentions DID land in a real (non-noise) cluster. rank_of() should have found
    this; if you see this case, something's inconsistent between `mentions` and `ranking` (e.g. stale
    ranking passed in) and is worth investigating directly rather than trusting this function further.

    author_records_df needs 'orcid_id', 'bibcode', 'year' columns (author_records.parquet has all
    three) -- used here as the CORPUS-WIDE record of this person's papers, independent of whether
    Stage 1 retrieved them for this specific query, to distinguish "no prior work exists" from
    "prior work exists but wasn't retrieved."
    """
    prior_papers = author_records_df[
        (author_records_df['orcid_id'] == candidate_id) & (author_records_df['year'] < query_year)
    ]
    n_prior_papers = prior_papers['bibcode'].nunique()

    retrieved_prior_bibcodes = sorted(set(prior_papers['bibcode']) & set(candidates['bibcode']))
    own_mentions = mentions[mentions['orcid_id'] == candidate_id]

    if n_prior_papers == 0:
        case = 'cold_start'
    elif len(retrieved_prior_bibcodes) == 0:
        case = 'retrieval_miss'
    elif (own_mentions['cluster_id'] == -1).all():
        case = 'noise'
    else:
        case = 'resolved'

    return {
        'candidate_id': candidate_id,
        'case': case,
        'n_prior_papers_in_corpus': n_prior_papers,
        'n_prior_papers_retrieved': len(retrieved_prior_bibcodes),
        'retrieved_prior_bibcodes': retrieved_prior_bibcodes,
        'n_own_mentions_in_pool': len(own_mentions),
        'n_own_mentions_clustered': int((own_mentions['cluster_id'] != -1).sum()),
    }


def disambiguation_metrics(mentions):
    """Pairwise precision/recall/F1 of cluster_id against orcid_id, for one query's candidate-pool
    mentions. Restricted to mentions with a known orcid_id, excluding noise (cluster_id == -1) --
    noise isn't a cluster, so two different noise mentions sharing label -1 must not be counted as
    "predicted together".

    Precision: of the mention-pairs Stage 2 merged into the same cluster, how many are truly the
    same person (catches false merges of different same-named people). Recall: of the truly-same-
    person mention-pairs, how many did Stage 2 actually merge together (catches missed merges /
    over-splitting).
    """
    labeled = mentions[mentions['orcid_id'].notna() & (mentions['cluster_id'] != -1)]
    if len(labeled) < 2:
        return {'n_labeled_mentions': len(labeled), 'precision': float('nan'),
                'recall': float('nan'), 'f1': float('nan')}

    # pair_confusion_matrix warns when unique-value-count is close to sample-count -- expected here
    # (orcid_id/cluster_id are per-person identifiers; small candidate pools naturally have most
    # people appearing only once or twice), not a sign anything's wrong. Suppressed narrowly rather
    # than globally so other, real warnings still surface.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=UserWarning)
        (_, fp), (fn, tp) = pair_confusion_matrix(labeled['orcid_id'], labeled['cluster_id'])
    precision = tp / (tp + fp) if (tp + fp) else float('nan')
    recall = tp / (tp + fn) if (tp + fn) else float('nan')
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else float('nan')
    return {'n_labeled_mentions': len(labeled), 'precision': precision, 'recall': recall, 'f1': f1}


def disambiguation_metrics_by_block(mentions):
    """Same precision/recall/F1 as disambiguation_metrics(), but computed and reported SEPARATELY per
    name block (surname + given-initial, e.g. 'j.smith') instead of pooled across a whole query's
    candidate pool. Pooling doesn't change the precision/recall VALUE -- cross-block pairs are always
    "predicted different" (different blocks -> different cluster IDs by construction) and, since
    they're also almost always "actually different" (different orcid_id), they land as true negatives
    in pair_confusion_matrix, which never enters the precision/recall formulas (tp/(tp+fp),
    tp/(tp+fn); no tn term) -- so a pooled per-paper score is not literally inflated by them.

    What pooling DOES hide is which blocks were ever actually tested. A block with only one labeled
    orcid_id trivially scores perfect precision/recall (there's no other same-block person to
    accidentally merge with) -- disambiguation_metrics()'s own "give up" condition (len(labeled) < 2
    LABELED MENTIONS) doesn't catch this: a single person's 10 own mentions all correctly clustered
    together still has len(labeled) == 10, so it proceeds to compute a trivially-perfect 1.0 rather
    than recognizing that a false merge was never actually possible here. PRECISION AND F1 ARE
    THEREFORE OVERRIDDEN TO NaN BELOW whenever n_labeled_orcid_ids < 2, so this table reports
    "untestable" honestly instead of a misleadingly confident 1.0 that a caller has to know to filter
    out themselves. n_clusters_among_labeled tells you if a single orcid_id's mentions got split
    across more than one cluster (a missed merge, hurting recall) within that block.

    n_papers -- distinct bibcodes among this block's LABELED mentions -- is a second, independent
    trivial-case check: a block whose labeled mentions all come from a single paper can't test
    whether the model correctly links the same person's mentions ACROSS papers (recall), even if it
    has 2+ labeled orcid_ids, since a single paper's author list only ever lists a given real person
    once -- there's nothing there to merge across papers. n_papers > 1 is what to filter on before
    trusting a block's recall as a real cross-paper test (recall itself is NOT overridden to NaN here
    the way precision is, since recall's own NaN condition in disambiguation_metrics -- zero true
    same-person pairs -- already catches most of this case correctly; n_papers remains block-level
    aggregate, not per-person, so treat recall_testable as a slightly looser signal than
    precision_testable, per disambiguation_metrics_by_size's docstring).

    Returns one row per block, sorted by n_labeled_orcid_ids descending -- the blocks worth looking
    at first are the ones at the top (actual same-name-collision cases), not the trivial ones.
    """
    blocks = mentions['author_name'].map(stage2.block_key)
    rows = []
    for block, group in mentions.groupby(blocks):
        metrics = disambiguation_metrics(group)
        labeled = group[group['orcid_id'].notna() & (group['cluster_id'] != -1)]
        n_labeled_orcid_ids = labeled['orcid_id'].nunique()
        if n_labeled_orcid_ids < 2:
            # No second labeled person in this block -- a false merge was never possible, so
            # precision (and f1, which depends on it) is untestable here, not genuinely 1.0.
            metrics = {**metrics, 'precision': float('nan'), 'f1': float('nan')}
        rows.append({
            'block': block,
            'n_mentions': len(group),
            'n_papers': labeled['bibcode'].nunique(),
            'n_labeled_orcid_ids': n_labeled_orcid_ids,
            'n_clusters_among_labeled': labeled['cluster_id'].nunique(),
            **metrics,
        })
    return pd.DataFrame(rows).sort_values('n_labeled_orcid_ids', ascending=False).reset_index(drop=True)


def summarize_block_metrics(block_metrics):
    """Macro-average of disambiguation_metrics_by_block()'s per-block precision/recall/f1 -- every
    GENUINELY TESTABLE block counts equally, regardless of how many pairs it contributed. This is a
    deliberately different number from disambiguation_metrics()'s (and summarize_disambiguation()'s)
    pooled/micro average, which sums TP/FP/FN across all blocks BEFORE dividing -- since pair count
    grows quadratically with a block's mention count, a handful of large, mostly-correct blocks can
    dominate that pooled sum and hide a majority of small blocks being entirely wrong (e.g. a
    2-labeled-person block with one false merge contributes exactly 1 pair and a precision of 0.0,
    but a 19-mention block that's almost entirely correct contributes ~171 pairs -- pooled together,
    the second block's correct pairs can numerically bury the first block's error). Macro-averaging
    instead answers "of the blocks we could actually test, what fraction were handled correctly,"
    weighting a small false merge exactly as heavily as a large mostly-correct one.

    IMPORTANT: "precision is not NaN" is NOT the same thing as "precision was genuinely tested," and
    averaging over the former (an earlier version of this function did) silently pulls in every block
    with just ONE labeled orcid_id whose mentions happen to span 2+ papers -- disambiguation_metrics()
    only returns NaN precision when a block has fewer than 2 LABELED MENTIONS total, not fewer than 2
    DISTINCT labeled people, so a single person with several mentions still gets a non-NaN, trivially-
    perfect precision (there's no one else to falsely merge with). That's a real bug, not a subtlety:
    it can inflate "n_blocks measurable" and mean_precision by an order of magnitude by counting
    hundreds of blocks that were never actually testing for a false merge. Precision is therefore
    restricted here to n_labeled_orcid_ids >= 2 (a false merge was actually possible) and recall to
    n_papers > 1 (there was actually something to link across papers) -- matching
    disambiguation_metrics_by_block()'s own precision_testable/recall_testable criteria exactly.
    f1 is restricted to blocks meeting BOTH (a genuine, full test), since it combines both quantities.
    """
    precision_testable = block_metrics[block_metrics['n_labeled_orcid_ids'] >= 2]
    recall_testable = block_metrics[block_metrics['n_papers'] > 1]
    both_testable = block_metrics[
        (block_metrics['n_labeled_orcid_ids'] >= 2) & (block_metrics['n_papers'] > 1)
    ]
    return pd.Series({
        'n_blocks': len(block_metrics),
        'n_blocks_precision_testable': len(precision_testable),
        'mean_precision': precision_testable['precision'].mean(),
        'n_blocks_recall_testable': len(recall_testable),
        'mean_recall': recall_testable['recall'].mean(),
        'n_blocks_f1_testable': len(both_testable),
        'mean_f1': both_testable['f1'].mean(),
    })


def disambiguation_metrics_for_query(mentions):
    """Per-query-paper precision/recall/f1 -- a macro-average over THIS query's own name blocks,
    not the flat, pooled disambiguation_metrics(mentions) result.

    Pooling every block's mentions into one pair_confusion_matrix call reintroduces the exact
    trivial-inflation bug already fixed for the by-block table (disambiguation_metrics_by_block() /
    summarize_block_metrics()): a query's candidate pool is typically dominated by single-identity
    blocks (one ORCID-confirmed person, all their mentions correctly self-clustered) that contribute
    large numbers of GUARANTEED-correct pairs -- a 16-mention single-identity block alone contributes
    C(16,2)=120 true-positive pairs -- while the query's only genuinely testable block (2+ different
    labeled people, a real false-merge opportunity) might contribute just 1 pair. Pooled together, a
    single real false merge can be completely buried under the trivial blocks' bulk of correct pairs
    (this is exactly what was observed for one query: a pooled precision of 0.9997 despite its one
    precision-testable block scoring 0.0).

    Reuses summarize_block_metrics()'s macro-average -- every genuinely testable block within this
    query counts equally regardless of its pair count -- just scoped to one query's blocks instead of
    a whole batch's. NaN precision/recall/f1 if this query has zero testable blocks of that kind
    (same convention as disambiguation_metrics()).
    """
    block_metrics = disambiguation_metrics_by_block(mentions)
    summary = summarize_block_metrics(block_metrics)
    return {
        'n_labeled_mentions': int(block_metrics['n_labeled_mentions'].sum()),
        'precision': summary['mean_precision'],
        'recall': summary['mean_recall'],
        'f1': summary['mean_f1'],
        'n_blocks_precision_testable': int(summary['n_blocks_precision_testable']),
        'n_blocks_recall_testable': int(summary['n_blocks_recall_testable']),
        'n_blocks_f1_testable': int(summary['n_blocks_f1_testable']),
    }


def disambiguation_metrics_by_size(mentions, size_threshold=None):
    """Pooled disambiguation_metrics(), computed SEPARATELY for 'small' vs 'big' name blocks, split
    by total block size (all mentions with that block_key, not just the labeled subset) -- the same
    quantity adaptive_eps() uses to set each block's DBSCAN radius (tighter for big blocks, looser
    for small ones, on the assumption that a small block has little false-merge risk to begin with).
    This checks whether that assumption actually holds: if small blocks turn out to have WORSE
    precision than big ones, the looser radius given to small blocks is misjudging their risk, not
    correctly reflecting it.

    size_threshold: block size at or below which a block counts as 'small'. Defaults to the median
    block size across all blocks present in `mentions` (not weighted by mentions -- the median block,
    not the median mention).

    Returns a DataFrame with one row each for 'small' and 'big', reusing disambiguation_metrics()'s
    pooled precision/recall/f1 within each group -- so, like that function, still a pair-count-
    weighted average WITHIN each size group (small blocks are still pooled together among themselves,
    not macro-averaged) -- this splits summarize_disambiguation()'s single pooled number into two,
    it doesn't switch to macro-averaging (see summarize_block_metrics() for that).
    """
    blocks = mentions['author_name'].map(stage2.block_key)
    block_sizes = blocks.value_counts()
    mention_block_size = blocks.map(block_sizes)

    if size_threshold is None:
        size_threshold = block_sizes.median()

    is_small = mention_block_size <= size_threshold
    rows = []
    for group_name, group_mask in [('small', is_small), ('big', ~is_small)]:
        group_mentions = mentions[group_mask]
        group_blocks = blocks[group_mask]
        rows.append({
            'group': group_name,
            'size_threshold': size_threshold,
            'n_blocks': group_blocks.nunique(),
            'n_mentions': len(group_mentions),
            **disambiguation_metrics(group_mentions),
        })
    return pd.DataFrame(rows).set_index('group')


def block_paper_counts(mentions):
    """For every name block present in `mentions`, how many distinct candidate papers its LABELED
    mentions span. A block whose labeled mentions all come from a single paper contributes no real
    cross-paper disambiguation signal -- a paper's author list only ever lists a given real person
    once, so there's nothing there to correctly (or incorrectly) link across papers. Most of a typical
    candidate pool's blocks are exactly this trivial case (see filter_testable_blocks()), which is why
    a naive size-based split can end up comparing mostly-empty groups instead of a meaningful one.

    Returns one row per block: (block, n_papers). Blocks with zero labeled mentions at all get
    n_papers = 0, not silently dropped -- this is meant to cover every block actually present in the
    candidate pool, e.g. for reporting what fraction of blocks were even evaluable to begin with.
    """
    blocks = mentions['author_name'].map(stage2.block_key)
    labeled = mentions[mentions['orcid_id'].notna() & (mentions['cluster_id'] != -1)]
    labeled_blocks = blocks.loc[labeled.index]
    n_papers = labeled.groupby(labeled_blocks)['bibcode'].nunique()
    n_papers = n_papers.reindex(blocks.unique(), fill_value=0)
    return n_papers.rename_axis('block').rename('n_papers').reset_index()


def filter_testable_blocks(mentions, min_papers=None, min_labeled_orcid_ids=None):
    """Restricts `mentions` to only the name blocks that clear given testability thresholds (ANDed
    together if both given), dropping the rest entirely rather than just flagging them -- so a pooled
    metric computed on the result isn't diluted by blocks that were never really being tested. At
    least one of the two must be given.

    min_papers: keep only blocks whose LABELED mentions span at least this many distinct candidate
    papers -- the RECALL-testability criterion (see block_paper_counts()). A block below this can
    only ever score a trivial recall: nothing to link across papers.

    min_labeled_orcid_ids: keep only blocks with at least this many DISTINCT labeled orcid_ids -- the
    PRECISION-testability criterion (see disambiguation_metrics_by_block()). A block below this can
    only ever score a trivial precision: with at most one real person in it, there's no one else to
    falsely merge with, however many mentions or papers that one person has.

    These are NOT interchangeable, and picking the wrong one silently produces a misleading result --
    e.g. checking whether false-merge risk (precision) varies with block size requires
    min_labeled_orcid_ids, not min_papers; filtering on min_papers alone still lets in hundreds of
    single-person, multi-paper blocks that trivially contribute perfect precision and dilute the real
    signal from genuine same-block collisions (this was a real bug in an earlier version of the "small
    vs. big blocks" notebook cell).

    Note on min_papers specifically: it also drops legitimate SAME-paper precision tests (two
    different labeled people who happen to co-occur on one paper -- a real false-merge opportunity
    even with only 1 paper involved). That's a deliberate simplification for a single, easy-to-explain
    "linked/separated across papers" evaluation universe when min_papers is the active criterion; use
    min_labeled_orcid_ids (or disambiguation_metrics_by_block() directly, unfiltered) if same-paper
    collisions matter for what you're checking.
    """
    if min_papers is None and min_labeled_orcid_ids is None:
        raise ValueError('filter_testable_blocks: give at least one of min_papers, min_labeled_orcid_ids')

    block_metrics = disambiguation_metrics_by_block(mentions)
    keep = pd.Series(True, index=block_metrics.index)
    if min_papers is not None:
        keep &= block_metrics['n_papers'] >= min_papers
    if min_labeled_orcid_ids is not None:
        keep &= block_metrics['n_labeled_orcid_ids'] >= min_labeled_orcid_ids
    testable_blocks = set(block_metrics.loc[keep, 'block'])

    blocks = mentions['author_name'].map(stage2.block_key)
    return mentions[blocks.isin(testable_blocks)]


def diagnose_naive_paper_filter(mentions, min_papers=2, min_labeled_orcid_ids=2):
    """Quantifies the selection bias from filtering to 'orcid appears as a label on 2+ papers' as a
    gate for evaluating PRECISION, not just recall (see summarize_block_metrics() for why these need
    to be separate criteria).

    Takes the raw `mentions` (not block_metrics) because this needs a PER-PERSON paper count, which
    block_metrics's own 'n_papers' column does NOT give you: that column is a block-AGGREGATE count
    (distinct papers across every labeled person sharing the block combined), so a block with 2
    different people who each published exactly 1 paper still shows n_papers == 2 there -- it would
    incorrectly pass a naive 'n_papers >= 2' filter despite neither person individually meeting that
    bar. This computes n_papers PER (block, orcid_id) instead, to find blocks where every individual
    person has < min_papers labeled papers -- these are exactly the single-paper-per-person collision
    blocks (e.g. the x.li/l.ji cases: several distinct real people, one paper each, merged into one
    cluster) where essentially every observed false merge was concentrated. A naive per-person
    2+-papers filter would strip these people's labels entirely, since none of them individually
    qualifies -- disproportionately removing the hard cases and keeping prolific, easily-
    disambiguated authors whose many topically-similar papers cluster together easily. That's a
    selection-bias explanation for suspiciously high pooled precision, not evidence the model is
    actually doing better on the hard cases.

    Returns counts of precision-testable blocks (n_labeled_orcid_ids >= min_labeled_orcid_ids) that
    would survive vs. be entirely dropped by a naive per-person min_papers filter (survives if AT
    LEAST ONE person in the block individually has >= min_papers), and each half's mean precision --
    if the dropped half scores much worse, that confirms the naive filter cherry-picks easy cases.
    """
    blocks = mentions['author_name'].map(stage2.block_key)
    labeled = mentions[mentions['orcid_id'].notna() & (mentions['cluster_id'] != -1)].copy()
    labeled['block'] = blocks.loc[labeled.index]

    per_person_papers = labeled.groupby(['block', 'orcid_id'])['bibcode'].nunique()
    n_orcids_per_block = per_person_papers.groupby('block').size()
    precision_testable_blocks = set(n_orcids_per_block[n_orcids_per_block >= min_labeled_orcid_ids].index)

    max_papers_per_block = per_person_papers.groupby('block').max()
    survives_naive_filter = max_papers_per_block >= min_papers

    kept_blocks = [b for b in precision_testable_blocks if survives_naive_filter.get(b, False)]
    dropped_blocks = [b for b in precision_testable_blocks if not survives_naive_filter.get(b, False)]

    block_metrics = disambiguation_metrics_by_block(mentions).set_index('block')

    return pd.Series({
        'n_precision_testable_blocks': len(precision_testable_blocks),
        'n_kept_by_naive_filter': len(kept_blocks),
        'n_dropped_by_naive_filter': len(dropped_blocks),
        'pct_dropped': (100 * len(dropped_blocks) / len(precision_testable_blocks)
                         if precision_testable_blocks else float('nan')),
        'mean_precision_all_testable': block_metrics.loc[list(precision_testable_blocks), 'precision'].mean(),
        'mean_precision_kept_by_naive_filter': (
            block_metrics.loc[kept_blocks, 'precision'].mean() if kept_blocks else float('nan')
        ),
        'mean_precision_dropped_by_naive_filter': (
            block_metrics.loc[dropped_blocks, 'precision'].mean() if dropped_blocks else float('nan')
        ),
    })


def evaluate_disambiguation_by_block(
    labels, query_papers, vectorizer, matrix, papers, bibcode_to_row, author_records_df,
    chars2vec_model, specter_tokenizer, specter_model, nand_encoder,
    top_n_papers=50, dbscan_min_samples=5, dbscan_year_eps=3,
    dbscan_min_eps=stage2.MIN_EPS, dbscan_max_eps=stage2.MAX_EPS,
    dbscan_max_expected_block_size=stage2.MAX_EXPECTED_BLOCK_SIZE,
    device='cpu', executor=None, similarity_executor=None, n_similarity_chunks=8,
    candidates_cache=None, orcid_lookup=None,
    paper_embedding_cache=None, author_embedding_cache=None,
):
    """Block-level counterpart to evaluate_labels, built for TUNING adaptive_eps's parameters
    (dbscan_min_eps / dbscan_max_eps / dbscan_max_expected_block_size), not for reporting final
    results -- use evaluate_labels for that. Runs Stage 1/2 once per query paper (grouped by bibcode,
    same as evaluate_labels), and returns disambiguation_metrics_by_block()'s per-block rows
    concatenated across every query paper in `labels`, tagged with bibcode, so
    summarize_block_metrics() can be computed once over the whole set -- see tune_adaptive_eps(),
    which wraps this in a parameter sweep.

    paper_embedding_cache / author_embedding_cache: pass PREBUILT dicts in (e.g. reused across many
    calls with different eps parameters) to skip re-running chars2vec/SPECTER2 -- only clustering
    actually depends on the eps parameters being tuned here, so a sweep should embed once and
    re-cluster many times, not re-embed from scratch on every grid point. If not given, fresh empty
    dicts are used (and populated in place, but not returned -- pass your own in if you intend to
    reuse them across repeated calls, e.g. from tune_adaptive_eps).
    """
    if orcid_lookup is None:
        orcid_lookup = stage2.build_orcid_lookup(author_records_df)
    if paper_embedding_cache is None:
        paper_embedding_cache = {}
    if author_embedding_cache is None:
        author_embedding_cache = {}

    labels_by_bibcode = {}
    for label in labels:
        labels_by_bibcode.setdefault(label['bibcode'], []).append(label)

    block_frames = []
    for bibcode in labels_by_bibcode:
        query_row = query_papers.loc[query_papers['bibcode'] == bibcode]
        if query_row.empty:
            continue
        query_row = query_row.iloc[0]

        cached_candidates = candidates_cache.get(bibcode) if candidates_cache is not None else None
        candidates = cached_candidates if cached_candidates is not None else stage1.find_similar_papers(
            query_row['title'], query_row['abstract'], query_row['year'],
            vectorizer, matrix, papers, top_n=top_n_papers,
            similarity_executor=similarity_executor, n_similarity_chunks=n_similarity_chunks,
        )
        if candidates.empty:
            continue

        mentions = stage2.get_candidate_mentions(
            candidates['bibcode'], papers, bibcode_to_row=bibcode_to_row, orcid_lookup=orcid_lookup
        )
        if mentions.empty:
            continue

        disambiguated = stage2.disambiguate_mentions(
            mentions, chars2vec_model, specter_tokenizer, specter_model, nand_encoder,
            min_samples=dbscan_min_samples, year_eps=dbscan_year_eps,
            min_eps=dbscan_min_eps, max_eps=dbscan_max_eps,
            max_expected_block_size=dbscan_max_expected_block_size,
            device=device, executor=executor,
            paper_embedding_cache=paper_embedding_cache, author_embedding_cache=author_embedding_cache,
        )
        block_metrics = disambiguation_metrics_by_block(disambiguated)
        block_metrics['bibcode'] = bibcode
        block_frames.append(block_metrics)

    if not block_frames:
        return pd.DataFrame(columns=[
            'block', 'n_mentions', 'n_papers', 'n_labeled_orcid_ids', 'n_clusters_among_labeled',
            'n_labeled_mentions', 'precision', 'recall', 'f1', 'bibcode',
        ])
    return pd.concat(block_frames, ignore_index=True)


def tune_adaptive_eps(
    labels, query_papers, vectorizer, matrix, papers, bibcode_to_row, author_records_df,
    chars2vec_model, specter_tokenizer, specter_model, nand_encoder, param_grid,
    top_n_papers=50, dbscan_min_samples=5, dbscan_year_eps=3, device='cpu', executor=None,
    similarity_executor=None, n_similarity_chunks=8, candidates_cache=None, orcid_lookup=None,
):
    """Grid (or random) search over adaptive_eps's (min_eps, max_eps, max_expected_block_size),
    scored by summarize_block_metrics()'s MACRO-averaged F1 on genuinely testable blocks
    (n_labeled_orcid_ids >= 2 AND n_papers > 1) across `labels` -- deliberately not the pooled/micro
    F1, which can look good even when most real name collisions are handled badly (see
    summarize_block_metrics's docstring for why: a handful of large, easy blocks can numerically bury
    many small blocks that are entirely wrong).

    param_grid: list of (min_eps, max_eps, max_expected_block_size) tuples to try. Keep it small
    enough to be tractable -- each grid point re-clusters every query paper in `labels`.

    Run this against a VALIDATION set of query papers/labels, held out from whatever you use for
    final reported results -- tuning against the same data you evaluate on overstates real
    performance. Not enforced here; that split is the caller's responsibility.

    Embeddings (chars2vec/SPECTER2, the expensive part) are computed once, on the first grid point,
    and reused for every subsequent one via a shared cache -- only clustering depends on these
    parameters, so there's no reason to re-embed per grid point.

    Returns one row per grid point (the swept parameters plus summarize_block_metrics()'s columns),
    sorted by mean_f1 descending -- read off the top row's (min_eps, max_eps, max_expected_block_size)
    as the tuned values to use going forward.
    """
    paper_embedding_cache, author_embedding_cache = {}, {}
    rows = []
    for grid_i, (min_eps, max_eps, max_expected_block_size) in enumerate(param_grid):
        t0 = time.time()
        block_metrics = evaluate_disambiguation_by_block(
            labels, query_papers, vectorizer, matrix, papers, bibcode_to_row, author_records_df,
            chars2vec_model, specter_tokenizer, specter_model, nand_encoder,
            top_n_papers=top_n_papers, dbscan_min_samples=dbscan_min_samples, dbscan_year_eps=dbscan_year_eps,
            dbscan_min_eps=min_eps, dbscan_max_eps=max_eps,
            dbscan_max_expected_block_size=max_expected_block_size,
            device=device, executor=executor, similarity_executor=similarity_executor,
            n_similarity_chunks=n_similarity_chunks, candidates_cache=candidates_cache, orcid_lookup=orcid_lookup,
            paper_embedding_cache=paper_embedding_cache, author_embedding_cache=author_embedding_cache,
        )
        # First grid point pays for chars2vec/SPECTER2 (populates the caches above); every later grid
        # point should reuse them and only pay for re-clustering -- printed per grid point so a slow
        # run shows WHICH of those two it's actually stuck on, instead of just "it's slow".
        print(f'grid point {grid_i + 1}/{len(param_grid)} '
              f'(min_eps={min_eps}, max_eps={max_eps}, max_expected_block_size={max_expected_block_size}): '
              f'{time.time() - t0:.1f}s -- caches now hold {len(paper_embedding_cache)} papers, '
              f'{len(author_embedding_cache)} names')
        if block_metrics.empty:
            summary = pd.Series({
                'n_blocks': 0, 'n_blocks_precision_testable': 0, 'mean_precision': float('nan'),
                'n_blocks_recall_testable': 0, 'mean_recall': float('nan'),
                'n_blocks_f1_testable': 0, 'mean_f1': float('nan'),
            })
        else:
            summary = summarize_block_metrics(block_metrics)
        rows.append({
            'min_eps': min_eps, 'max_eps': max_eps, 'max_expected_block_size': max_expected_block_size,
            **summary.to_dict(),
        })

    return pd.DataFrame(rows).sort_values('mean_f1', ascending=False).reset_index(drop=True)


def merge_diagnostics(mentions):
    """For each labeled orcid_id's own (non-noise) cluster, breaks down what else Stage 2 pulled into
    that cluster. A broader, heuristic merge-risk signal than disambiguation_metrics_by_block's
    precision -- that one needs 2+ DIFFERENT labeled people in a block to ever be informative (see its
    docstring); this one is informative from a single labeled person, by treating any OTHER mention
    sharing that person's cluster as a candidate over-merge, labeled or not.

    LIMITATION -- read before treating this as a real error rate, not just a diagnostic: an unlabeled
    mention (orcid_id is null) sharing a cluster with a labeled orcid_id is NOT confirmed to be a
    different person -- it's only unconfirmed as the SAME person, since ORCID coverage on any given
    author position is incomplete (a real person's own papers routinely lack an ORCID link on that
    specific mention). Some of what n_unlabeled_merged counts below is therefore a CORRECT merge that
    just isn't ORCID-confirmed, not a real error. This is an upper bound on the true false-merge rate,
    useful for flagging clusters worth manually auditing -- not a number to report as precision.
    n_confirmed_other_people, by contrast, IS a rigorous false-merge count (a different confirmed
    orcid_id sharing the cluster is unambiguous ground truth) -- it's the same signal
    disambiguation_metrics' precision already captures, just broken out per identity here instead of
    pooled per block.

    Returns one row per (block, orcid_id, cluster_id):
      n_own_mentions            -- this orcid_id's own mentions in this cluster
      n_own_papers              -- distinct papers among them
      n_confirmed_other_people  -- mentions with a DIFFERENT non-null orcid_id in this cluster (a real,
                                    rigorous false merge)
      n_unlabeled_merged        -- mentions with no orcid_id sharing this cluster (the heuristic,
                                    upper-bound signal -- see limitation above)
      cluster_size              -- total mentions in this cluster, any label status
    """
    blocks = mentions['author_name'].map(stage2.block_key)
    labeled_clustered = mentions[mentions['orcid_id'].notna() & (mentions['cluster_id'] != -1)]

    rows = []
    for (orcid_id, cluster_id), group in labeled_clustered.groupby(['orcid_id', 'cluster_id']):
        cluster_all = mentions[mentions['cluster_id'] == cluster_id]
        is_other_confirmed = cluster_all['orcid_id'].notna() & (cluster_all['orcid_id'] != orcid_id)
        rows.append({
            'block': blocks.loc[group.index[0]],
            'orcid_id': orcid_id,
            'cluster_id': cluster_id,
            'n_own_mentions': len(group),
            'n_own_papers': group['bibcode'].nunique(),
            'n_confirmed_other_people': int(is_other_confirmed.sum()),
            'n_unlabeled_merged': int(cluster_all['orcid_id'].isna().sum()),
            'cluster_size': len(cluster_all),
        })
    return pd.DataFrame(rows).sort_values('n_unlabeled_merged', ascending=False).reset_index(drop=True)


def _score_label(label, ranking, mentions):
    """Scores one label against an already-computed (ranking, mentions) pair for its query paper --
    the expensive part (Stage 1/2/3) has already run; this is just rank_of()/disambiguation_metrics()
    lookups. Split out from evaluate_label so evaluate_labels can run the pipeline once per query
    paper and reuse the same (ranking, mentions) across every label that shares that paper's bibcode,
    instead of re-running clustering/ranking once per label (a paper with 5 confirmed authors used to
    mean 5 redundant full pipeline runs against the identical candidate pool).
    """
    return {
        'bibcode': label['bibcode'],
        'n_ranked_identities': len(ranking),
        'author_rank': rank_of(label['author'], ranking, mentions),
        'coauthor_ranks': [rank_of(orcid_id, ranking, mentions) for orcid_id in label['coauthors']],
        'cited_author_ranks': [rank_of(orcid_id, ranking, mentions) for orcid_id in label['cited_authors']],
        'disambiguation': disambiguation_metrics_for_query(mentions),
    }


def evaluate_label(
    label, query_papers, vectorizer, matrix, papers, bibcode_to_row, author_records_df,
    chars2vec_model, specter_tokenizer, specter_model, nand_encoder,
    top_n_papers=50, dbscan_min_samples=5, dbscan_year_eps=3,
    dbscan_min_eps=stage2.MIN_EPS, dbscan_max_eps=stage2.MAX_EPS,
    dbscan_max_expected_block_size=stage2.MAX_EXPECTED_BLOCK_SIZE, device='cpu', executor=None,
    paper_embedding_cache=None, author_embedding_cache=None,
    similarity_executor=None, n_similarity_chunks=8, candidates_cache=None, orcid_lookup=None,
    treat_noise_as_singletons=False,
):
    """label: one entry from the labels list, e.g.
    {'bibcode': ..., 'author': orcid_id, 'coauthors': [orcid_id, ...], 'cited_authors': [orcid_id, ...]}

    treat_noise_as_singletons: passed straight through to find_experts() -- see its docstring. Makes
    ranking never drop a candidate just because Stage 2 couldn't confidently merge their mention with
    anything else, at the cost of making disambiguation_metrics() on the result meaningless (every
    mention becomes "resolved"). Only set this True when you're not also reporting disambiguation
    quality from the same run.

    query_papers: [bibcode, title, abstract, year] for the query papers the labels reference.

    candidates_cache: optional {bibcode: Stage 1 result} dict, e.g. a `candidates_by_bibcode`
    precomputed once for a batch of queries -- if this label's bibcode is in it, Stage 1 is skipped
    entirely for this label instead of recomputing it. Cache miss just falls back to computing it.

    orcid_lookup: optional stage2.build_orcid_lookup(author_records_df) result -- avoids rebuilding
    that lookup from author_records_df fresh on every call (see build_orcid_lookup's docstring).

    Runs the full pipeline (Stage 1/2/3) for this one label's query paper -- evaluate_labels doesn't
    call this (it batches multiple labels per query paper through find_experts once, see _score_label
    above); this standalone function exists for evaluating a single label in isolation.

    Returns None if the query paper isn't in query_papers, or has no candidate papers published
    before it (both expected for very old query papers, since the TF-IDF index only covers the last
    15 years).
    """
    query_row = query_papers.loc[query_papers['bibcode'] == label['bibcode']]
    if query_row.empty:
        return None
    query_row = query_row.iloc[0]

    cached_candidates = candidates_cache.get(label['bibcode']) if candidates_cache is not None else None

    result = find_experts(
        query_title=query_row['title'], query_abstract=query_row['abstract'], query_year=query_row['year'],
        vectorizer=vectorizer, matrix=matrix, papers=papers, bibcode_to_row=bibcode_to_row,
        author_records_df=author_records_df,
        chars2vec_model=chars2vec_model, specter_tokenizer=specter_tokenizer,
        specter_model=specter_model, nand_encoder=nand_encoder,
        top_n_papers=top_n_papers, dbscan_min_samples=dbscan_min_samples, dbscan_year_eps=dbscan_year_eps,
        dbscan_min_eps=dbscan_min_eps, dbscan_max_eps=dbscan_max_eps,
        dbscan_max_expected_block_size=dbscan_max_expected_block_size, device=device, executor=executor,
        paper_embedding_cache=paper_embedding_cache, author_embedding_cache=author_embedding_cache,
        similarity_executor=similarity_executor, n_similarity_chunks=n_similarity_chunks,
        candidates=cached_candidates, orcid_lookup=orcid_lookup,
        treat_noise_as_singletons=treat_noise_as_singletons,
    )
    ranking, mentions = result['ranking'], result['mentions']
    if ranking.empty:
        return None

    return _score_label(label, ranking, mentions)


def evaluate_labels(
    labels, query_papers, vectorizer, matrix, papers, bibcode_to_row, author_records_df,
    chars2vec_model, specter_tokenizer, specter_model, nand_encoder,
    top_n_papers=50, dbscan_min_samples=5, dbscan_year_eps=3,
    dbscan_min_eps=stage2.MIN_EPS, dbscan_max_eps=stage2.MAX_EPS,
    dbscan_max_expected_block_size=stage2.MAX_EXPECTED_BLOCK_SIZE, device='cpu', executor=None,
    similarity_executor=None, n_similarity_chunks=8, candidates_cache=None, orcid_lookup=None,
    return_timings=False, treat_noise_as_singletons=False,
    paper_embedding_cache=None, author_embedding_cache=None,
    progress_every=None, progress_label='',
):
    """Runs evaluate_label over a list of labels. Returns (rank_results, disambiguation_results), or
    (rank_results, disambiguation_results, paper_timings) if return_timings=True:

    treat_noise_as_singletons: passed straight through to find_experts() for every query paper here
    -- see its docstring. Setting this True means disambiguation_results coming out of this call is
    meaningless (every mention becomes "resolved," since noise is never left as -1); only turn it on
    when you're not reporting disambiguation quality from this same call.

    rank_results -- one row per (bibcode, evidence_type, candidate_id) rank, long format, for
    summarize(). Deduped at that grain, NOT one row per label -- query_records.json's labels are per
    (paper, confirmed-author): a coauthor pair {A, B} on the same paper appears in both A's and B's
    label, and a paper's cited authors are attached identically to every one of its confirmed
    authors' labels (see author_records.ipynb). Scoring per-label instead of per-(paper, candidate)
    would evaluate the exact same candidate against the exact same ranking multiple times (once per
    confirmed author whose label happens to mention them) with an identical result every time --
    inflating row counts and skewing summarize()'s hit_rate/MRR toward papers with many confirmed
    authors, without adding real evaluation signal.
    disambiguation_results -- one row per query PAPER (not per label), with that paper's
    disambiguation_metrics() -- also doesn't depend on which label it's attached to, same reasoning.
    paper_timings -- one row per query paper (bibcode, n_labels, n_mentions, seconds), for sizing a
    batch run (e.g. extrapolating a 1000-paper ETA from a small sample) -- per-paper cost varies
    hugely with author-list size (a small paper vs. a mega-collaboration paper can differ 10x+), so
    a single mean/it-per-second isn't enough to size a real run; the actual distribution is. Default
    off since most callers (run_eval_batch.py, the "Evaluate" notebook cell) don't need it and it's
    an extra positional slot to plumb through if unused.

    executor: passed straight through to disambiguate_mentions() for every label in this run --
    create it once (e.g. a ProcessPoolExecutor) and pass it in here rather than per-label, so its
    startup cost is paid once for the whole evaluation instead of once per query.

    similarity_executor: passed straight through to Stage 1's find_similar_papers() (both the
    pre-pass below and the one inside every evaluate_label call) -- a ThreadPoolExecutor, distinct
    from `executor`. Create it once and reuse it here too.

    candidates_cache: optional {bibcode: Stage 1 result} dict -- e.g. a `candidates_by_bibcode`
    already precomputed for these query papers elsewhere (a notebook's single-query prototyping
    cell, say). When a label's bibcode is present, Stage 1 is skipped for it in BOTH the pre-pass
    below and the per-label evaluate_label call -- otherwise this function recomputes Stage 1 twice
    per label even though the caller may have already paid for it once.

    orcid_lookup: optional stage2.build_orcid_lookup(author_records_df) result. If not given, it's
    built once here (not per label) -- author_records_df is often the entire ORCID-matched corpus,
    and rebuilding a lookup against it inside get_candidate_mentions on every single call (as used
    to happen, once in the pre-pass below and again per label) scales with author_records_df's size,
    not with any one query's tiny candidate pool, and dominates runtime at scale.

    Before evaluating anything, runs Stage 1 retrieval for every label up front to collect the union
    of candidate papers/author names across the WHOLE label list, and embeds that union through
    chars2vec/SPECTER2 exactly once (nand_embeddings.precompute_*). Candidate pools drawn from the
    same corpus overlap heavily across queries -- without this, the same paper gets re-embedded
    through SPECTER2 once per query it happens to show up in, in many small batches instead of one
    large one. The extra Stage 1 pass this costs is TF-IDF/cosine-similarity only (no GPU), cheap
    relative to what it saves -- cheaper still, or free, when candidates_cache covers it.

    Labels are grouped by bibcode and the pipeline (Stage 1/2/3 -- retrieval, clustering, ranking)
    runs once PER QUERY PAPER, not once per label -- a paper with several ORCID-confirmed authors
    produces several label records that all share the same query paper, so scoring them separately
    used to mean redoing the identical clustering/ranking once per label. Every label sharing a
    bibcode is scored (_score_label) against that one shared (ranking, mentions) result instead.
    """
    if orcid_lookup is None:
        orcid_lookup = stage2.build_orcid_lookup(author_records_df)

    t0 = time.time()
    candidate_bibcodes, candidate_author_names = set(), set()
    # Candidate pools are query-paper-level, not label-level. A paper may have many labels;
    # repeating this extraction once per label adds needless pandas work before the real pipeline.
    unique_bibcodes = list(dict.fromkeys(label['bibcode'] for label in labels))
    for bibcode in tqdm(unique_bibcodes, desc='Pre-pass: collecting candidate pool'):
        query_row = query_papers.loc[query_papers['bibcode'] == bibcode]
        if query_row.empty:
            continue
        query_row = query_row.iloc[0]
        cached_candidates = candidates_cache.get(bibcode) if candidates_cache is not None else None
        candidates = cached_candidates if cached_candidates is not None else stage1.find_similar_papers(
            query_row['title'], query_row['abstract'], query_row['year'],
            vectorizer, matrix, papers, top_n=top_n_papers,
            similarity_executor=similarity_executor, n_similarity_chunks=n_similarity_chunks,
        )
        if candidates.empty:
            continue
        mentions = stage2.get_candidate_mentions(
            candidates['bibcode'], papers, bibcode_to_row=bibcode_to_row, orcid_lookup=orcid_lookup
        )
        candidate_bibcodes.update(mentions['bibcode'])
        candidate_author_names.update(mentions['author_name'].dropna())

    print(f'Pre-pass done: {time.time() - t0:.1f}s -> {len(candidate_bibcodes)} unique papers, '
          f'{len(candidate_author_names)} unique author names')

    if paper_embedding_cache is None:
        paper_embedding_cache = nand_embeddings.precompute_paper_embeddings(
            candidate_bibcodes, papers, specter_tokenizer, specter_model, device=device
        )
    else:
        missing = candidate_bibcodes.difference(paper_embedding_cache)
        if missing:
            raise ValueError(f'paper_embedding_cache is missing {len(missing)} candidate papers')
        print(f'Reusing shared paper embedding cache: {len(paper_embedding_cache)} papers')

    if author_embedding_cache is None:
        author_embedding_cache = nand_embeddings.precompute_author_name_embeddings(
            candidate_author_names, chars2vec_model
        )
    else:
        missing = candidate_author_names.difference(author_embedding_cache)
        if missing:
            raise ValueError(f'author_embedding_cache is missing {len(missing)} author names')
        print(f'Reusing shared author embedding cache: {len(author_embedding_cache)} names')

    labels_by_bibcode = {}
    for label in labels:
        labels_by_bibcode.setdefault(label['bibcode'], []).append(label)

    t1 = time.time()
    rank_rows = []
    disambiguation_rows = []
    paper_timings = []
    progress = tqdm(labels_by_bibcode.items(), desc='Evaluating query papers')
    for paper_i, (bibcode, bibcode_labels) in enumerate(progress, start=1):
        paper_t0 = time.time()
        query_row = query_papers.loc[query_papers['bibcode'] == bibcode]
        if query_row.empty:
            continue
        query_row = query_row.iloc[0]

        cached_candidates = candidates_cache.get(bibcode) if candidates_cache is not None else None

        result = find_experts(
            query_title=query_row['title'], query_abstract=query_row['abstract'], query_year=query_row['year'],
            vectorizer=vectorizer, matrix=matrix, papers=papers, bibcode_to_row=bibcode_to_row,
            author_records_df=author_records_df,
            chars2vec_model=chars2vec_model, specter_tokenizer=specter_tokenizer,
            specter_model=specter_model, nand_encoder=nand_encoder,
            top_n_papers=top_n_papers, dbscan_min_samples=dbscan_min_samples, dbscan_year_eps=dbscan_year_eps,
            dbscan_min_eps=dbscan_min_eps, dbscan_max_eps=dbscan_max_eps,
            dbscan_max_expected_block_size=dbscan_max_expected_block_size, device=device, executor=executor,
            paper_embedding_cache=paper_embedding_cache, author_embedding_cache=author_embedding_cache,
            similarity_executor=similarity_executor, n_similarity_chunks=n_similarity_chunks,
            candidates=cached_candidates, orcid_lookup=orcid_lookup,
            treat_noise_as_singletons=treat_noise_as_singletons,
        )
        ranking, mentions = result['ranking'], result['mentions']

        paper_seconds = time.time() - paper_t0
        progress.set_postfix(bibcode=bibcode, s=f'{paper_seconds:.1f}')
        tqdm.write(f'  {bibcode}: {paper_seconds:6.1f}s -> {len(bibcode_labels)} labels, {len(mentions)} mentions')
        paper_timings.append({
            'bibcode': bibcode, 'n_labels': len(bibcode_labels), 'n_mentions': len(mentions),
            'seconds': paper_seconds,
        })
        if progress_every and (paper_i % progress_every == 0 or paper_i == len(labels_by_bibcode)):
            elapsed = time.time() - t1
            rate = elapsed / paper_i
            print(f'{progress_label} progress: {paper_i}/{len(labels_by_bibcode)} papers '
                  f'complete; {paper_seconds:.1f}s last, {rate:.1f}s mean', flush=True)

        # One disambiguation row per PAPER, not per label -- disambiguation_metrics_for_query(mentions)
        # only depends on this query paper's own clustering result, not on which of its confirmed
        # authors' labels it's attached to, so computing/appending it once per label (as this used to
        # do) just wrote the same row N times for a paper with N confirmed authors. Uses the per-block
        # macro-average (not the flat pooled disambiguation_metrics()) so a query dominated by trivial
        # single-identity blocks can't bury a real false merge in one of its few testable blocks.
        disambiguation_rows.append({'bibcode': bibcode, **disambiguation_metrics_for_query(mentions)})

        if ranking.empty:
            continue

        # Dedupe ground-truth candidates by (evidence_type, candidate_id) BEFORE scoring, not after --
        # coauthorship/citation ground truth is a property of the (paper, candidate) pair, not of
        # which confirmed author's label happened to mention them. query_records.json's labels are
        # per-(paper, confirmed-author): a coauthor pair {A, B} appears in both A's and B's label, and
        # a paper's cited authors are attached identically to every one of its confirmed authors' labels
        # (see author_records.ipynb's coauthor_labels/citation_labels construction). Scoring per-label
        # like the old code did would call rank_of() on, and add a row for, the exact same candidate
        # multiple times with an identical result each time (ranking/mentions are the same for every
        # label sharing this bibcode) -- inflating n_labels and skewing summarize()'s hit_rate/MRR
        # toward papers with many confirmed authors, without adding any real evaluation signal.
        n_ranked = len(ranking)
        candidates_by_evidence_type = {'authorship': set(), 'coauthorship': set(), 'citation': set()}
        for label in bibcode_labels:
            candidates_by_evidence_type['authorship'].add(label['author'])
            candidates_by_evidence_type['coauthorship'].update(label['coauthors'])
            candidates_by_evidence_type['citation'].update(label['cited_authors'])

        for evidence_type, candidate_ids in candidates_by_evidence_type.items():
            for candidate_id in candidate_ids:
                rank_rows.append({
                    'bibcode': bibcode, 'evidence_type': evidence_type, 'candidate_id': candidate_id,
                    'n_ranked_identities': n_ranked, 'rank': rank_of(candidate_id, ranking, mentions),
                })

    paper_timings_df = pd.DataFrame(paper_timings)
    total_seconds = time.time() - t1
    print(f'Evaluating query papers: {total_seconds:.1f}s -> {len(labels_by_bibcode)} query papers, '
          f'{len(rank_rows)} rank rows, {len(disambiguation_rows)} disambiguation rows')
    if not paper_timings_df.empty:
        print(f'  per-paper seconds: mean={paper_timings_df["seconds"].mean():.1f}  '
              f'median={paper_timings_df["seconds"].median():.1f}  '
              f'min={paper_timings_df["seconds"].min():.1f}  max={paper_timings_df["seconds"].max():.1f}')

    if return_timings:
        return pd.DataFrame(rank_rows), pd.DataFrame(disambiguation_rows), paper_timings_df
    return pd.DataFrame(rank_rows), pd.DataFrame(disambiguation_rows)


def summarize(rank_results):
    """Mean reciprocal rank and hit rate (found anywhere in the ranking) per evidence type."""
    def _summarize_group(g):
        found = g['rank'].notna()
        reciprocal_rank = (1 / g.loc[found, 'rank']).sum() / len(g) if len(g) else float('nan')
        return pd.Series({
            'n_labels': len(g),
            'hit_rate': found.mean(),
            'mean_reciprocal_rank': reciprocal_rank,
        })

    return rank_results.groupby('evidence_type').apply(_summarize_group, include_groups=False)


def summarize_disambiguation(disambiguation_results):
    """Mean precision/recall/F1 across queries, each averaged independently over its own non-null
    rows (pandas .mean() skips NaN by default) -- a query can have a defined precision with an
    undefined recall (e.g. two different people and no true-same-person pairs to recall at all), so
    dropping a row from every metric just because one of the three is NaN would throw away a real
    signal on the other two.

    Each row already IS a macro-average (disambiguation_metrics_for_query() averages over that one
    query's own testable blocks), so this is a macro-of-macros: every query counts equally here too,
    regardless of how many testable blocks it had."""
    return pd.Series({
        'n_queries': len(disambiguation_results),
        'n_measurable_precision': disambiguation_results['precision'].notna().sum(),
        'mean_precision': disambiguation_results['precision'].mean(),
        'n_measurable_recall': disambiguation_results['recall'].notna().sum(),
        'mean_recall': disambiguation_results['recall'].mean(),
        'n_measurable_f1': disambiguation_results['f1'].notna().sum(),
        'mean_f1': disambiguation_results['f1'].mean(),
    })
