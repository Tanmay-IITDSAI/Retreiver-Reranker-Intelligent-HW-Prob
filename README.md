# Rewriter‑Retriever‑ReRanker — Homework Problem

**Natural Language Processing (NLP)**

> A compact system for generating, retrieving, rewriting and re-ranking homework questions using modern NLP models. Built as part of the Indian Institute of Technology, Bhilai project (Nov 23, 2024).

---

## Table of contents

* [Project overview](#project-overview)
* [Highlights / Features](#highlights--features)
* [Pipeline / Architecture](#pipeline--architecture)
* [Models & Libraries used](#models--libraries-used)
* [Installation](#installation)
* [Quickstart](#quickstart)
* [Notebooks & files](#notebooks--files)
* [Experiments & evaluation](#experiments--evaluation)
* [Results & notes](#results--notes)
* [Contributing](#contributing)
* [License & contact](#license--contact)

---

## Project overview

This project builds an end‑to‑end system to improve question retrieval and query relevance for a homework/problem dataset. It combines **data augmentation** (synthetic question generation and paraphrasing), **semantic retrieval** (dense embeddings + FAISS), **query rewriting** (Flan‑T5), and **re‑ranking** (cross‑encoder) to surface higher quality, more relevant question recommendations.

(Report source: project report PDF.)

---

## Highlights / Features

* Synthetic question generation using GPT‑style / generative LMs.
* Paraphrasing (T5 based) for dataset augmentation.
* Dense semantic search with `sentence-transformers` + FAISS for extremely fast nearest‑neighbour retrieval.
* Query rewriting via Flan‑T5 for better semantic matching.
* Re‑ranking with a cross‑encoder to improve final ranking quality.
* Evaluation using multiple metrics: nDCG, SacreBLEU, cosine precision and an LLM‑based precision check.

---

## Pipeline / Architecture

1. **Data augmentation**

   * Synthetic question generation (GPT‑2 / T5 variants).
   * Paraphrasing via a T5 paraphrase model to expand the dataset.

2. **Indexing & retrieval**

   * Encode documents and queries with a sentence‑transformer (e.g. `all‑MiniLM‑L6‑v2`).
   * Build a FAISS index over question embeddings.
   * Retrieve top‑K candidates using cosine / inner product similarity.

3. **Query rewriting**

   * Use Flan‑T5 to rewrite user queries to improve semantic match with indexed questions.

4. **Re‑ranking**

   * Feed query–document pairs into a cross‑encoder (BERT/CrossEncoder style) to re‑score and sort candidates.

5. **Evaluation**

   * Compute nDCG, SacreBLEU, cosine precision and LLM‑based precision for retrieved results.

---

## Models & libraries (recommended)

* Python 3.8+
* `torch` (CPU/GPU as available)
* `transformers` (Hugging Face)
* `sentence-transformers`
* `faiss-cpu` (or `faiss-gpu` if you have CUDA)
* `scikit-learn`, `numpy`, `pandas`
* `sacrebleu`
* `nltk`, `tqdm`
* (Optional) `accelerate` for distributed/GPU convenience

Example `pip` install:

```bash
python -m pip install torch transformers sentence-transformers faiss-cpu scikit-learn pandas sacrebleu nltk tqdm
```

---

## Installation

1. Clone the repository

```bash
git clone <your-repo-url>
cd <your-repo>
```

2. Create a virtualenv and install dependencies

```bash
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
```

3. (Optional) If you have a GPU and CUDA, install `faiss-gpu` and a CUDA‑enabled `torch`.

---

## Quickstart

### Run the main notebook

Open the provided notebook to reproduce experiments and walkthrough the pipeline:

```bash
jupyter notebook Copy_of_NLP_PROJECT_IHWP.ipynb
# or
jupyter lab
```

### Example commands (illustrative)

* Build embeddings & FAISS index

```python
# inside a script or notebook
from sentence_transformers import SentenceTransformer
import faiss

model = SentenceTransformer('all-MiniLM-L6-v2')
embs = model.encode(list_of_questions, show_progress_bar=True)
index = faiss.IndexFlatIP(embs.shape[1])  # inner product (cosine with normalized vectors)
index.add(embs)
# save index
faiss.write_index(index, 'faiss_index.bin')
```

* Retrieve & rerank

```python
# encode query (and rewritten query if using Flan-T5)
q_emb = model.encode(query)
D, I = index.search(q_emb.reshape(1,-1), k=50)
# take top-k IDs and feed pairs to cross-encoder for re-ranking
```

---

## Notebooks & files

* `Copy_of_NLP_PROJECT_IHWP.ipynb` — main notebook with end‑to‑end pipeline and experiments.
* `NLP_Project (3).pdf` — project report (used as the basis for this README). fileciteturn0file0

---

## Experiments & evaluation

Metrics used in the report:

* **nDCG** (Normalized Discounted Cumulative Gain)
* **SacreBLEU** for similarity between query and retrieved/generated text
* **Cosine Precision** (presence of relevant doc in top‑k by cosine)
* **LLM‑based Precision**: LLM (GPT‑style) is used to judge retrieved result relevance

Key reported values from the project report:

* nDCG: **0.9574**
* LLM‑based Precision: **1.00**
* SacreBLEU: **3.73**

These values demonstrate strong ranking behaviour on the dataset used. See the notebook for experiment code and evaluation scripts. fileciteturn0file0

---

## Results & notes

* The pipeline shows high ranking quality (nDCG near 0.96) and strong LLM‑labelled precision on the evaluation set.
* Main performance bottleneck: CPU‑bound operations when GPU is not available. Indexing and re‑ranking benefit greatly from GPU acceleration.
* SacreBLEU is sensitive to exact lexical matches — low BLEU does not necessarily mean poor semantic relevance.

---

## Reproducibility tips

* Set random seeds for all libraries (numpy, torch, etc.) when running experiments.
* Normalize embeddings before using cosine similarity or use FAISS metric types consistently.
* Save trained paraphrase/synthetic outputs so augmentation is deterministic between runs.

---

## Contributing

Contributions welcome. Suggested ways to help:

* Add unit tests for the retrieval/re‑ranking components.
* Integrate an evaluation suite to compare different sentence‑transformer models.
* Add GPU support and CI checks for core scripts.

Please open issues or PRs on the repository.
-..
