
# LLM_RAG_Project

# Amazon Financial Research RAG

This project starts with a framework-light PDF ingestion step for Amazon financial documents.

## Corpus found in `data/raw`

The 83 current PDFs include Amazon annual reports, quarterly 10-Q filings, earnings releases,
shareholder letters, earnings slides, and one business/financial update. The filenames span
2019–2026, with quarterly materials primarily covering 2020–2026.

## Set up and run

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 src/ingest.py
```

The ingester reads every PDF in `data/raw/` and creates:

- `data/processed/<source-name>.txt`: extracted text with page markers.
- `data/processed/<source-name>.json`: source filename, company, ticker, year, quarter,
  document type, page count, empty-page list, and character count.
- `data/processed/manifest.json`: a combined inventory of all processed documents.

Re-running the command safely replaces outputs that have the same source filename.

## Create fixed-token chunks

```bash
python3 src/chunk.py
```

The chunker uses `tiktoken`'s `cl100k_base` encoding and a baseline of 500 tokens with
75-token overlap. It reads the ingestion manifest and processed text files, then writes
`data/processed/chunks/chunks.jsonl` and a per-document `summary.json`. Original processed
text and metadata files are not changed. You can experiment with alternatives such as
`--chunk-size 800 --overlap 100`.

## Build embeddings and retrieve evidence

```bash
python3 src/embed.py
python3 src/retrieve.py "What was AWS sales growth in Q2 2026?" --top-k 5
```

`embed.py` uses `BAAI/bge-small-en-v1.5` locally to turn every saved chunk into a
normalized embedding vector. It stores a cosine-similarity FAISS index, a row-aligned
metadata mapping, and the model configuration in `data/processed/vector_index/`.
`retrieve.py` embeds a question with that same model and prints the highest-scoring chunks,
including their document IDs, periods, types, chunk numbers, and text snippets.

The first run downloads the embedding model. Embeddings are numeric representations used
to find similar text; they do not generate prose. A generative LLM can be added later to
write an answer from the retrieved evidence.

## Generate a grounded answer

Install dependencies and provide your API key through an environment variable. Never save
the key in this repository:

```bash
python3 -m pip install -r requirements.txt
export OPENAI_API_KEY="your-key-here"
python3 src/generate.py "What was AWS sales growth in Q2 2025?" --top-k 5
```

Optionally set `OPENAI_MODEL` or pass `--model`. To inspect retrieval and the exact prompt
without making an API call, add `--show-prompt`.

`generate.py` first calls the existing FAISS retriever, labels each result `[S1]`, `[S2]`,
and so on, and only then sends the question and retrieved text to the LLM. The prompt tells
the model to rely only on that evidence, cite source labels, and say when the context is
insufficient. The command prints the retrieved sources next to the answer and reports
latency and token usage. Exact cost is left to the provider dashboard because it depends on
the selected model and current pricing.

## Generate locally with Ollama

The OpenAI path above remains available as the hosted-model backup. To run generation
locally, install [Ollama](https://ollama.com), start its service, and pull the default model:

```bash
ollama serve
ollama pull qwen2.5:3b
python3 src/generate_ollama.py "What was AWS sales growth in Q2 2025?" --top-k 5
```

No API key is needed. Choose another installed model with `--model`, or set
`OLLAMA_MODEL`. You can also use `--show-prompt` to validate retrieval and prompt
construction without Ollama running. The local script reuses the same retriever,
grounding prompt, source labels, and chunk metadata as `generate.py`.

## Compare with OpenAI generation

The separate OpenAI comparison keeps the Ollama files and results unchanged. It uses the
same winning `800/100` index, `top_k=8`, bounded eight-source prompt, and citation labels as
the full local evaluation. The cost-conscious default is `gpt-5-mini`; override it only
with a model available to your API project. The program never silently substitutes a model.

```bash
export OPENAI_API_KEY="your-key-here"
python3 src/generate_openai.py \
  "What was AWS sales growth in Q2 2025?"
```

Inspect the retrieval and exact prompt without an API call:

```bash
python3 src/generate_openai.py \
  "What was AWS sales growth in Q2 2025?" --show-prompt
```

Run one benchmark question as a paid smoke test, then all 12 questions:

```bash
python3 src/evaluate_full_rag_openai.py --limit 1 --fail-fast
python3 src/evaluate_full_rag_openai.py
```

OpenAI artifacts are saved under `evals/results/full_rag/openai/`, including a separate
`latest_summary.json`, so Ollama artifacts and summaries are not overwritten. Missing keys,
invalid keys, and inaccessible models produce explicit errors.

## Evaluate the baseline

`evals/questions.json` is a small, human-readable benchmark. Each question names one or
more `expected_documents`; multi-document questions only pass the stricter check when
every expected document appears in the top-k results.

Run the complete local baseline with the current defaults (500-token chunks, 75-token
overlap, BGE embeddings, FAISS, top-k 5, and Qwen 2.5 3B):

```bash
python3 src/evaluate.py
```

For a fast retrieval-only run that does not call a generative model:

```bash
python3 src/evaluate.py --generator none
```

Each run writes a timestamped JSON artifact under `evals/results/`. It records per-question
retrieved chunks, document hit@k, expected-document recall@k, the stricter all-documents
check, retrieval latency, generated answers, generation latency and token counts when the
backend supplies them. Answer quality is deliberately a separate manual review field in
V1: review the answer against `answer_notes` and retrieved evidence, then change its status
to `pass`, `partial`, or `fail`. Retrieval scores never depend on that judgment.

## Run controlled retrieval experiments

To compare the fixed retrieval grid without changing the baseline chunks, index, or results:

```bash
python3 src/run_retrieval_experiments.py
```

This evaluates chunk/overlap pairs `300/50`, `500/75`, and `800/100` at top-k values
`3`, `5`, and `8`. Each chunk/index build is saved under its own artifact directory in
`evals/results/retrieval_experiments/`. The runner writes detailed per-question JSON plus
ranked `comparison.csv` and `comparison.json` files. Ranking prioritizes document hit rate,
then multi-document completeness, mean expected-document recall, and retrieval latency.
Use `--reuse-existing` to rerun evaluation and timing against already-built experiment indexes.

## Run hybrid retrieval

The hybrid experiment preserves the dense baseline and combines its winning `800/100`
FAISS index with a lightweight BM25 lexical index. Each retriever contributes 15 candidates,
reciprocal rank fusion (RRF) combines and deduplicates them, and the final eight chunks are
evaluated with the same retrieval metrics and 12 questions. No reranker is used.

```bash
python3 src/retrieve_hybrid.py "How did Amazon's net sales change from 2023 to 2024?"
python3 src/evaluate_retrieval_hybrid.py
```

The first run builds a reusable BM25 cache. Timestamped evaluation details and an explicit
comparison with the dense 75% hit-rate baseline are saved separately under
`evals/results/retrieval_hybrid/`.

## Run local cross-encoder reranking

The reranking experiment preserves both earlier baselines. It asks the hybrid retriever
for 20 deduplicated RRF candidates, scores each question/chunk pair with the lightweight
local `cross-encoder/ms-marco-MiniLM-L-6-v2` model, and returns the best eight chunks.
Dense, BM25, and RRF scores/ranks remain attached alongside the reranker score and rank.

```bash
python3 src/retrieve_hybrid_rerank.py \
  "How did Amazon's net sales change from 2023 to 2024?"
python3 src/evaluate_retrieval_rerank.py
```

The first run may download the reranker model. The retrieval-only evaluator uses the same
12 questions and metrics, records end-to-end retrieval latency including reranking, and
compares every question directly with the dense 75% and hybrid 83.3% artifacts. Timestamped
results and `latest_summary.json` are written under `evals/results/retrieval_rerank/`.
This step does not run either full-RAG benchmark.

## Evaluate full RAG with the winning retrieval configuration

The full-RAG stage fixes retrieval at the best tested setting: 800-token chunks,
100-token overlap, and `top_k=8`. It reuses the saved experiment index rather than
rebuilding or replacing the baseline:

```bash
python3 src/evaluate_full_rag.py --generator ollama --model qwen2.5:3b
```

Results are written under `evals/results/full_rag/`. Retrieval metrics, generation
latency/token metrics, and answer-quality metrics are reported separately. Correctness
uses the benchmark reference answers, completeness uses reference-answer recall, and
citation/groundedness heuristics are labeled as heuristics. For semantic review, add an
optional structured 1–5 judge without changing deterministic retrieval scores:

```bash
python3 src/evaluate_full_rag.py --judge ollama --judge-model qwen2.5:3b
```

The hosted generator remains available with `--generator openai` when
`OPENAI_API_KEY` is configured. Use `--judge openai` for a hosted judge.

## Project layout

```text
LLM_Project/
├── data/
│   ├── raw/
│   └── processed/
├── evals/
├── notebooks/
├── src/
│   ├── ingest.py
│   ├── chunk.py
│   ├── embed.py
│   ├── retrieve.py
│   ├── retrieve_hybrid.py
│   ├── retrieve_hybrid_rerank.py
│   ├── generate.py
│   ├── generate_ollama.py
│   ├── generate_openai.py
│   ├── evaluate.py
│   ├── evaluate_retrieval_hybrid.py
│   ├── evaluate_retrieval_rerank.py
│   ├── evaluate_full_rag.py
│   └── evaluate_full_rag_openai.py
├── tests/
├── .gitignore
├── README.md
└── requirements.txt
```
