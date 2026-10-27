#!/usr/bin/env python3
"""Find reviewer candidates for one paper.

Examples
--------
Find a paper already present in the TF-IDF paper index::

    python run_reviewer_search.py --bibcode 2020ApJ...123..456A

Provide a new query paper as JSON::

    python run_reviewer_search.py --query query.json --top-k 20

The JSON file must contain ``title``, ``abstract``, and ``year``.  The script
uses the research pipeline directly; it does not use ORCID labels or other
evaluation-only data.
"""

import argparse
import json
from pathlib import Path

from expertise_pipeline import find_experts
from expertise_pipeline import nand_embeddings, nand_model
from expertise_pipeline.stage1_retrieval import load_tfidf_index

PROJECT_DIR = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description="Rank reviewer candidates for one paper")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--bibcode",
        help="Bibcode of a paper in the TF-IDF paper index",
    )
    source.add_argument(
        "--query",
        type=Path,
        help="JSON file containing title, abstract, and year",
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=PROJECT_DIR / "tfidf_index",
        help="Directory containing the saved TF-IDF index",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_DIR / nand_model.CHECKPOINT_PATH,
        help="Trained NAND checkpoint",
    )
    parser.add_argument(
        "--top-n-papers",
        type=int,
        default=500,
        help="Number of earlier papers used to build the candidate pool",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Number of reviewer candidates to print",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device, for example cpu or cuda",
    )
    return parser.parse_args()


def load_query(args, papers):
    if args.bibcode:
        matches = papers[papers["bibcode"] == args.bibcode]
        if matches.empty:
            raise ValueError(
                f"Bibcode {args.bibcode!r} was not found in {args.index_dir}"
            )
        row = matches.iloc[0]
        return row["title"], row["abstract"], int(row["year"])

    with args.query.open() as handle:
        query = json.load(handle)

    missing = {"title", "abstract", "year"} - set(query)
    if missing:
        raise ValueError(f"Query JSON is missing required fields: {sorted(missing)}")
    return query["title"], query["abstract"], int(query["year"])


def main():
    args = parse_args()

    vectorizer, matrix, papers, bibcode_to_row = load_tfidf_index(str(args.index_dir))
    title, abstract, year = load_query(args, papers)

    print("Loading name and paper-embedding models...", flush=True)
    chars2vec_model = nand_embeddings.load_chars2vec_model()
    specter_tokenizer, specter_model = nand_embeddings.load_specter2_model(
        device=args.device
    )
    nand_encoder = nand_model.load_nand_model(
        checkpoint_path=str(args.checkpoint), device=args.device
    )

    result = find_experts(
        title,
        abstract,
        year,
        vectorizer,
        matrix,
        papers,
        bibcode_to_row,
        author_records_df=None,
        chars2vec_model=chars2vec_model,
        specter_tokenizer=specter_tokenizer,
        specter_model=specter_model,
        nand_encoder=nand_encoder,
        top_n_papers=args.top_n_papers,
        device=args.device,
    )

    ranking = result["ranking"].head(args.top_k)
    reviewers = [
        {
            "rank": int(record["rank"]),
            "name": record["display_name"],
            "expertise_score": float(record["expertise_score"]),
            "paper_count": int(record["paper_count"]),
        }
        for record in ranking.to_dict(orient="records")
    ]

    print(json.dumps(reviewers, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
