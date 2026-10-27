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
MAX_EPS = 0.4
MAX_EXPECTED_BLOCK_SIZE = 95632  # calibrated against the training data's block-size distribution


def block_key(author_name):
    """given-name initial + surname, e.g. 'Smith, John A.' -> 'j.smith'."""
    if not isinstance(author_name, str):
        return '.'
    tokens = author_name.lower().split()
    if not tokens:
        return '.'
    family = re.sub(r'\W+', '', tokens[0])
    given_initial = re.sub(r'\W+', '', tokens[1])[:1] if len(tokens) > 1 else ''
    return f'{given_initial}.{family}'


def adaptive_eps(block_size, min_eps=MIN_EPS, max_eps=MAX_EPS):
    """Larger blocks (more same-initial mentions to tell apart) get a tighter radius; small blocks
    get a looser one, since there's little room for a false merge anyway. Log-scaled between
    max_eps (block_size == 1) and min_eps (block_size >= MAX_EXPECTED_BLOCK_SIZE)."""
    scale = (np.log1p(block_size) - np.log1p(1)) / (np.log1p(MAX_EXPECTED_BLOCK_SIZE) - np.log1p(1))
    eps = max_eps - scale * (max_eps - min_eps)
    return np.clip(eps, min_eps, max_eps)


def get_candidate_mentions(candidate_bibcodes, author_records_df, papers_df):
    """All author mentions on papers in the Stage 1 candidate pool, with each mention's paper's
    title/abstract/year attached -- everything the NAND model needs per mention."""
    candidate_bibcodes = set(candidate_bibcodes)
    mentions = author_records_df[author_records_df['bibcode'].isin(candidate_bibcodes)][
        ['bibcode', 'author_name', 'author_position', 'orcid_id', 'identity_type']
    ].copy()
    mentions = mentions.merge(papers_df[['bibcode', 'title', 'abstract', 'year']], on='bibcode', how='left')
    return mentions.reset_index(drop=True)


def disambiguate_mentions(
    mentions, chars2vec_model, specter_tokenizer, specter_model, nand_encoder,
    min_samples=1, year_eps=5, min_eps=MIN_EPS, max_eps=MAX_EPS, device='cpu',
):
    """Adds a 'cluster_id' column to `mentions`. -1 means noise (excluded from Stage 3 ranking --
    not confidently resolved to any identity). Requires `mentions` to have a clean 0..n-1 index
    (get_candidate_mentions already resets it) -- used directly as positions into the embedding
    arrays below.
    """
    author_embeddings = nand_embeddings.embed_author_names(chars2vec_model, mentions['author_name'])
    paper_embeddings = nand_embeddings.embed_papers(
        specter_tokenizer, specter_model, mentions['title'], mentions['abstract'], device=device
    )
    mention_embeddings = nand_model.encode_mentions(nand_encoder, author_embeddings, paper_embeddings, device=device)

    blocks = mentions['author_name'].map(block_key)
    cluster_ids = pd.Series(-1, index=mentions.index, dtype=object)

    for block, group in mentions.groupby(blocks):
        block_embeddings = mention_embeddings[group.index]
        eps = adaptive_eps(len(group), min_eps=min_eps, max_eps=max_eps)

        _, local_labels = dbscanc(
            block_embeddings, group['year'].tolist(), group['author_name'].tolist(),
            eps=eps, min_samples=min_samples, year_eps=year_eps,
        )

        cluster_ids.loc[group.index] = [
            f'{block}::{label}' if label != -1 else -1 for label in local_labels
        ]

    out = mentions.copy()
    out['cluster_id'] = cluster_ids
    return out
