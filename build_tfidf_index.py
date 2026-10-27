#!/usr/bin/env python3
"""Fit the retrieval TF-IDF representation and build a searchable index.

The vocabulary is fit only on the released query-paper table. The candidate-paper
table is then transformed with that same fitted representation. Keeping these two
tables separate makes the scientific procedure explicit: the query papers define
the vocabulary, while the candidate corpus defines what can be retrieved.

Both input tables must contain ``title``, ``abstract``, ``year``, and ``bibcode``.
The candidate table must also contain ``author`` and ``aff`` lists because Stage 2
uses them to construct author mentions.
"""

import argparse
from pathlib import Path

import joblib
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from expertise_pipeline.stage1_retrieval import (
    MATRIX_FILE,
    PAPERS_FILE,
    VECTORIZER_FILE,
)


REQUIRED_QUERY_COLUMNS = {"bibcode", "title", "abstract", "year"}
REQUIRED_CANDIDATE_COLUMNS = REQUIRED_QUERY_COLUMNS | {"author", "aff"}


def document_text(papers):
    return (
        papers["title"].fillna("").astype(str)
        + " "
        + papers["abstract"].fillna("").astype(str)
    ).tolist()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fit TF-IDF on query papers and transform a candidate-paper corpus"
    )
    parser.add_argument(
        "--vocabulary-papers",
        type=Path,
        required=True,
        help="Parquet table used to fit the TF-IDF vocabulary",
    )
    parser.add_argument(
        "--candidate-papers",
        type=Path,
        required=True,
        help="Parquet table to transform into the searchable candidate index",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tfidf_index"),
        help="Directory for the fitted vectorizer, matrix, and metadata",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    vocabulary_papers = pd.read_parquet(args.vocabulary_papers)
    candidate_papers = pd.read_parquet(args.candidate_papers)

    missing = REQUIRED_QUERY_COLUMNS - set(vocabulary_papers.columns)
    if missing:
        raise ValueError(f"Vocabulary table is missing columns: {sorted(missing)}")

    missing = REQUIRED_CANDIDATE_COLUMNS - set(candidate_papers.columns)
    if missing:
        raise ValueError(f"Candidate table is missing columns: {sorted(missing)}")

    vocabulary_papers = vocabulary_papers.drop_duplicates("bibcode")
    candidate_papers = candidate_papers.drop_duplicates("bibcode").reset_index(drop=True)

    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    vectorizer.fit(document_text(vocabulary_papers))
    matrix = vectorizer.transform(document_text(candidate_papers))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, args.output_dir / VECTORIZER_FILE)
    sparse.save_npz(args.output_dir / MATRIX_FILE, matrix)
    candidate_papers[
        ["bibcode", "title", "abstract", "author", "aff", "year"]
    ].to_parquet(args.output_dir / PAPERS_FILE)

    print(f"Vocabulary papers: {len(vocabulary_papers)}")
    print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")
    print(f"Candidate papers: {len(candidate_papers)}")
    print(f"TF-IDF matrix shape: {matrix.shape}")
    print(f"Saved index to {args.output_dir}")


if __name__ == "__main__":
    main()
