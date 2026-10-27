# Retrieve-then-Disambiguate

This repository contains the inference code for the reviewer-expertise pipeline:

1. Retrieve earlier papers with TF–IDF cosine similarity.
2. Extract every author mention from the retrieved papers.
3. Embed names and papers, then disambiguate mentions with constrained DBSCAN.
4. Rank the resulting identities by TF–IDF similarity to the query paper.

The retrieval cutoff is strict: only papers published before the query year are eligible. The
pipeline does not require ORCID records for ordinary queries; ORCID data are used only by the
evaluation code as ground truth.

## Installation

Use Python 3.10 and install the dependencies:

```bash
python -m pip install -r requirements.txt
```

For GPU inference, install a PyTorch build matching the machine's CUDA version. The SPECTER2
base model and adapter, and the `chars2vec` `eng_50` model, are downloaded by the model-loading
functions on first use.

## Required files

Place these files in `tfidf_index/`:

```text
tfidf_index/
├── tfidf_matrix.npz
├── tfidf_papers.parquet
└── tfidf_vectorizer.joblib
```

`tfidf_papers.parquet` must contain `bibcode`, `title`, `abstract`, `year`, `author`, and `aff`
columns. The `author` and `aff` columns contain the author and affiliation lists used by Stage 2.

The trained NAND checkpoint is included at the repository root:

```text
1_specter_chars2vec_nceepoch=67-val_loss=7.47.ckpt
```

The index and checkpoint must be redistributed only under licenses that permit their use.

## Run a query

For a query paper already present in the index:

```bash
python run_reviewer_search.py \
  --bibcode 2020ApJ...123..456A \
  --top-n-papers 500 \
  --top-k 20 \
  --device cpu
```

For a new query, create a JSON file with `title`, `abstract`, and `year`:

```json
{
  "title": "Example paper title",
  "abstract": "Example paper abstract.",
  "year": 2020
}
```

Then run:

```bash
python run_reviewer_search.py --query query.json --top-k 20 --device cpu
```

The command prints a JSON list containing each candidate's rank, name, expertise score, and
number of supporting papers.

## Reproducing the study

To reproduce study results, distribute the exact retrieval index, query papers, labels or other
ground-truth files, model checkpoint, dependency versions, and experiment settings. The
evaluation utilities in `expertise_pipeline/evaluate.py` and `multi_gpu_eval.py` are separate
from the simple single-query command above.
