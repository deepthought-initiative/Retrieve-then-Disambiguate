"""Stage 3: rank disambiguated identities (clusters) by TF-IDF similarity to the query paper.

Each cluster's "expertise document" is the concatenation of every candidate-pool paper's title +
abstract attributed to that cluster. A fresh TfidfVectorizer is fit per query, over the query
document plus every cluster's expertise document together (small -- one document per candidate
identity, not the whole corpus) so they all land in the same, query-specific vector space.
"""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def build_expertise_documents(disambiguated_mentions):
    """One row per cluster_id (excluding -1 = noise -- not a confidently resolved identity), with
    the concatenated title+abstract text of every distinct paper attributed to that cluster."""
    clustered = disambiguated_mentions[disambiguated_mentions['cluster_id'] != -1]

    def _agg(group):
        papers = group.drop_duplicates(subset=['bibcode'])
        document = ' '.join(f'{title} {abstract}' for title, abstract in zip(papers['title'], papers['abstract']))
        return pd.Series({
            'expertise_document': document,
            'paper_count': len(papers),
            'display_name': group['author_name'].mode().iloc[0],
        })

    return clustered.groupby('cluster_id').apply(_agg, include_groups=False).reset_index()


def rank_experts(query_title, query_abstract, expertise_documents):
    """Returns [cluster_id, display_name, paper_count, expertise_score, rank], most expert first."""
    if expertise_documents.empty:
        return expertise_documents.assign(expertise_score=[], rank=[])

    query_document = f'{query_title} {query_abstract}'
    documents = [query_document] + expertise_documents['expertise_document'].tolist()

    vectorizer = TfidfVectorizer(stop_words='english')
    matrix = vectorizer.fit_transform(documents)
    scores = cosine_similarity(matrix[0], matrix[1:])[0]

    ranking = expertise_documents[['cluster_id', 'display_name', 'paper_count']].copy()
    ranking['expertise_score'] = scores
    ranking = ranking.sort_values('expertise_score', ascending=False).reset_index(drop=True)
    ranking['rank'] = range(1, len(ranking) + 1)
    return ranking
