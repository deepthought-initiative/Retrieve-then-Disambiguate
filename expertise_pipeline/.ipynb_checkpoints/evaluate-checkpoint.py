"""Evaluate the ranking against the coauthorship/citation labels.

For each label (a query paper + its true author, coauthors, and cited authors), runs the pipeline
with that paper's own year as the temporal cutoff, then reports what rank each evidence type's
orcid_id(s) landed at in the resulting cluster ranking. A good ranking should put authorship,
coauthorship, and citation-linked people near the top; comparing their rank distributions is how we
tell which evidence type the ranking is best/worst at surfacing.

An orcid_id maps to a rank via whichever cluster(s) its mentions ended up in (Stage 2's clustering
output), not directly -- if that orcid_id's mentions got split across multiple clusters (a
disambiguation error), the best (lowest) rank among them is used, which is optimistic and can hide
fragmentation. Worth checking separately whether an orcid_id's mentions stayed in one cluster before
trusting this at scale.
"""
import pandas as pd

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


def evaluate_label(
    label, query_papers, vectorizer, matrix, papers, bibcode_to_row, author_records_df,
    chars2vec_model, specter_tokenizer, specter_model, nand_encoder,
    top_n_papers=50, dbscan_eps=0.5, dbscan_min_samples=5, dbscan_year_eps=3,
):
    """label: one entry from the labels list, e.g.
    {'bibcode': ..., 'author': orcid_id, 'coauthors': [orcid_id, ...], 'cited_authors': [orcid_id, ...]}

    query_papers: [bibcode, title, abstract, year] for the query papers the labels reference.

    Returns None if the query paper isn't in query_papers, or has no candidate papers published
    before it (both expected for very old query papers, since the TF-IDF index only covers the last
    15 years).
    """
    query_row = query_papers.loc[query_papers['bibcode'] == label['bibcode']]
    if query_row.empty:
        return None
    query_row = query_row.iloc[0]

    result = find_experts(
        query_title=query_row['title'], query_abstract=query_row['abstract'], query_year=query_row['year'],
        vectorizer=vectorizer, matrix=matrix, papers=papers, bibcode_to_row=bibcode_to_row,
        author_records_df=author_records_df,
        chars2vec_model=chars2vec_model, specter_tokenizer=specter_tokenizer,
        specter_model=specter_model, nand_encoder=nand_encoder,
        top_n_papers=top_n_papers,
        dbscan_eps=dbscan_eps, dbscan_min_samples=dbscan_min_samples, dbscan_year_eps=dbscan_year_eps,
    )
    ranking, mentions = result['ranking'], result['mentions']
    if ranking.empty:
        return None

    return {
        'bibcode': label['bibcode'],
        'n_ranked_identities': len(ranking),
        'author_rank': rank_of(label['author'], ranking, mentions),
        'coauthor_ranks': [rank_of(orcid_id, ranking, mentions) for orcid_id in label['coauthors']],
        'cited_author_ranks': [rank_of(orcid_id, ranking, mentions) for orcid_id in label['cited_authors']],
    }


def evaluate_labels(
    labels, query_papers, vectorizer, matrix, papers, bibcode_to_row, author_records_df,
    chars2vec_model, specter_tokenizer, specter_model, nand_encoder,
    top_n_papers=50, dbscan_eps=0.5, dbscan_min_samples=5, dbscan_year_eps=3,
):
    """Runs evaluate_label over a list of labels and returns one row per (label, evidence_type,
    orcid_id) rank -- long format, easy to summarize with a groupby."""
    rows = []
    for label in labels:
        result = evaluate_label(
            label, query_papers, vectorizer, matrix, papers, bibcode_to_row, author_records_df,
            chars2vec_model, specter_tokenizer, specter_model, nand_encoder,
            top_n_papers, dbscan_eps, dbscan_min_samples, dbscan_year_eps,
        )
        if result is None:
            continue

        rows.append({'bibcode': result['bibcode'], 'evidence_type': 'authorship',
                      'n_ranked_identities': result['n_ranked_identities'], 'rank': result['author_rank']})
        for rank in result['coauthor_ranks']:
            rows.append({'bibcode': result['bibcode'], 'evidence_type': 'coauthorship',
                         'n_ranked_identities': result['n_ranked_identities'], 'rank': rank})
        for rank in result['cited_author_ranks']:
            rows.append({'bibcode': result['bibcode'], 'evidence_type': 'citation',
                         'n_ranked_identities': result['n_ranked_identities'], 'rank': rank})

    return pd.DataFrame(rows)


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
