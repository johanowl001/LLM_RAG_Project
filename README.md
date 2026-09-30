# LLM_RAG_Project

# Amazon Financial Research RAG

Amazon Financial Research RAG is an evaluation-driven retrieval-augmented generation system built over 83 Amazon financial documents spanning 2019–2026. The project benchmarks dense retrieval, chunking strategies, hybrid BM25 + semantic retrieval, cross-encoder reranking, and local versus hosted LLM generation, with separate measurement of retrieval quality, answer correctness, completeness, grounding, latency, and token usage.

The goal is not simply to produce answers from financial documents, but to measure which retrieval and generation choices improve end-to-end RAG quality, identify failure modes, and reject components that do not add value.

## Key results

### Retrieval experiments

| Retrieval approach | Document Hit@K | Avg. latency | Key finding |
|---|---:|---:|---|
| Initial dense baseline | 33.3% | — | Weak baseline before tuning |
| Tuned dense retrieval | 75.0% | 10.3 ms | Best dense setup: 800-token chunks, 100-token overlap, `top_k=8` |
| Hybrid FAISS + BM25 | **83.3%** | 45 ms | Improved retrieval with no regressions versus tuned dense |
| Hybrid + cross-encoder reranker | 66.7% | 169 ms | Rejected: worse retrieval quality and higher latency |

### Generation experiments

| Generator | Correctness | Completeness | Groundedness / citations | Mean latency |
|---|---:|---:|---:|---:|
| Qwen 2.5 3B (local Ollama) | 0.203 | 0.334 | 0.000 | **2.59 s** |
| GPT-5 mini (OpenAI API) | **0.422** | **0.628** | **0.792** | 6.70 s |

The generation comparison used the same retrieval configuration and benchmark, isolating the effect of model capability. GPT-5 mini materially improved correctness, completeness, and citation compliance, while the smaller local model remained faster.

## Key findings

- Retrieval tuning materially improved performance: the best dense configuration increased document Hit@K from 33.3% to 75.0%.
- Hybrid retrieval combining semantic FAISS search with BM25 lexical search improved Hit@K further to 83.3%.
- Cross-encoder reranking did not help this benchmark. It reduced Hit@K to 66.7% and increased latency, so it was rejected rather than retained for complexity alone.
- Model capability was a major generation bottleneck. Using GPT-5 mini roughly doubled correctness and substantially improved completeness and citation behavior compared with local Qwen 2.5 3B.
- Retrieval and generation are evaluated separately so failures can be traced to candidate retrieval, ranking, or answer generation rather than being collapsed into one end-to-end score.

## Architecture

```text
Amazon PDFs
   ↓
PDF ingestion
   ↓
Processed text + metadata
   ↓
Fixed-token chunking
   ↓
Embedding model
   ↓
FAISS dense index
   ↓
BM25 lexical index
   ↓
Hybrid retrieval / RRF
   ↓
Top-k evidence chunks
   ↓
LLM generation
   ├── Local Ollama / Qwen
   └── OpenAI API
   ↓
Answer + source citations
   ↓
Evaluation
   ├── Retrieval quality
   ├── Correctness
   ├── Completeness
   ├── Grounding / citations
   ├── Latency
   └── Token usage
```

## Failure analysis

The benchmark is intentionally small and human-readable so retrieval failures can be inspected directly.

Two retrieval cases remain unresolved in the current hybrid setup:

- `q03`: the expected evidence is present in the processed Amazon 2024 Annual Report, but the correct document does not enter the hybrid candidate pool. This points to a candidate-generation / query-document wording mismatch rather than an ingestion failure.
- `q12`: the expected document is present in the candidate pool but ranks below the final cutoff, indicating a ranking rather than parsing problem.

The reranker did not solve these issues. In candidate analysis, it also pushed previously relevant documents down the ranking for other questions, which is why it was rejected.

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

## Corpus

The 83 current PDFs include Amazon annual reports, quarterly 10-Q filings, earnings releases, shareholder letters, earnings slides, and one business/financial update. The filenames span 2019–2026, with quarterly materials primarily covering 2020–2026.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 src/ingest.py
```

The ingester reads every PDF in `data/raw/` and creates:

- `data/processed/<source-name>.txt`: extracted text with page markers.
- `data/processed/<source-name>.json`: source filename, company, ticker, year, quarter, document type, page count, empty-page list, and character count.
- `data/processed/manifest.json`: a combined inventory of all processed documents.

Re-running the command safely replaces outputs that have the same source filename.

## Create fixed-token chunks

```bash
python3 src/chunk.py
```

The chunker uses `tiktoken`'s `cl100k_base` encoding. The original baseline was 500 tokens with 75-token overlap. Controlled experiments later found that 800-token chunks with 100-token overlap performed best on the current benchmark.

The chunker reads the ingestion manifest and processed text files, then writes `data/processed/chunks/chunks.jsonl` and a per-document `summary.json`. Original processed text and metadata files are not changed.

To reproduce the winning dense chunk configuration:

```bash
python3 src/chunk.py --chunk-size 800 --overlap 100
```

## Build embeddings and retrieve evidence

```bash
python3 src/embed.py
python3 src/retrieve.py "What was AWS sales growth in Q2 2026?" --top-k 8
```

`embed.py` uses `BAAI/bge-small-en-v1.5` locally to turn every saved chunk into a normalized embedding vector. It stores a cosine-similarity FAISS index, a row-aligned metadata mapping, and the model configuration in `data/processed/vector_index/`.

`retrieve.py` embeds a question with that same model and prints the highest-scoring chunks, including their document IDs, periods, types, chunk numbers, and text snippets.

The first run downloads the embedding model. Embeddings are numeric representations used to find similar text; they do not generate prose.

## Generate a grounded answer

```bash
python3 -m pip install -r requirements.txt
export OPENAI_API_KEY="your-key-here"
python3 src/generate.py "What was AWS sales growth in Q2 2025?" --top-k 8
```

Optionally set `OPENAI_MODEL` or pass `--model`. To inspect retrieval and the exact prompt without making an API call, add `--show-prompt`.

`generate.py` first calls the existing FAISS retriever, labels each result `[S1]`, `[S2]`, and so on, and only then sends the question and retrieved text to the LLM. The prompt tells the model to rely only on that evidence, cite source labels, and say when the context is insufficient.

The command prints the retrieved sources next to the answer and reports latency and token usage. Exact cost is left to the provider dashboard because it depends on the selected model and current pricing.

## Generate locally with Ollama

The OpenAI path remains available as the hosted-model comparison. To run generation locally, install Ollama, start its service, and pull the default model:

```bash
ollama serve
ollama pull qwen2.5:3b
python3 src/generate_ollama.py "What was AWS sales growth in Q2 2025?" --top-k 8
```

The local path allows the RAG system to run without external generation API calls, although model capability and hardware limits can materially affect answer quality and latency.

## Compare with OpenAI generation

The separate OpenAI comparison keeps the Ollama files and results unchanged. It uses the same winning `800/100` index, `top_k=8`, bounded eight-source prompt, and citation labels as the full local evaluation.

The tested hosted model was `gpt-5-mini`.

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

OpenAI artifacts are saved under `evals/results/full_rag/openai/`, including a separate `latest_summary.json`, so Ollama artifacts and summaries are not overwritten. Missing keys, invalid keys, and inaccessible models produce explicit errors.

## Evaluate the baseline

`evals/questions.json` is a small, human-readable benchmark. Each question names one or more `expected_documents`; multi-document questions only pass the stricter check when every expected document appears in the top-k results.

Run the original local baseline:

```bash
python3 src/evaluate.py
```

For a fast retrieval-only run that does not call a generative model:

```bash
python3 src/evaluate.py --generator none
```

Each run writes a timestamped JSON artifact under `evals/results/`. It records per-question retrieved chunks, document hit@k, expected-document recall@k, the stricter all-documents check, retrieval latency, generated answers, generation latency, and token counts when the backend supplies them.

Answer quality remains separate from deterministic retrieval scoring.

## Run controlled retrieval experiments

To compare the fixed retrieval grid without changing the baseline chunks, index, or results:

```bash
python3 src/run_retrieval_experiments.py
```

This evaluates chunk/overlap pairs `300/50`, `500/75`, and `800/100` at top-k values `3`, `5`, and `8`.

Each chunk/index build is saved under its own artifact directory in `evals/results/retrieval_experiments/`. The runner writes detailed per-question JSON plus ranked `comparison.csv` and `comparison.json` files.

Ranking prioritizes document hit rate, then multi-document completeness, mean expected-document recall, and retrieval latency.

Use `--reuse-existing` to rerun evaluation and timing against already-built experiment indexes.

The best tested dense configuration was:

```text
chunk size: 800
overlap: 100
top_k: 8
document Hit@K: 75.0%
mean expected-document recall: 75.0%
multi-document all-found rate: 66.7%
average retrieval latency: 10.3 ms
```

## Run hybrid retrieval

The hybrid experiment preserves the dense baseline and combines the winning `800/100` FAISS index with a lightweight BM25 lexical index.

Each retriever contributes 15 candidates, reciprocal rank fusion (RRF) combines and deduplicates them, and the final eight chunks are evaluated with the same retrieval metrics and 12 questions.

```bash
python3 src/retrieve_hybrid.py "How did Amazon's net sales change from 2023 to 2024?"
python3 src/evaluate_retrieval_hybrid.py
```

The first run builds a reusable BM25 cache. Timestamped evaluation details and an explicit comparison with the dense 75% hit-rate baseline are saved separately under `evals/results/retrieval_hybrid/`.

Hybrid retrieval improved document Hit@K to **83.3%**, fixed `q11`, and introduced no regressions versus the tuned dense baseline.

## Run local cross-encoder reranking

The reranking experiment preserves both earlier baselines. It asks the hybrid retriever for 20 deduplicated RRF candidates, scores each question/chunk pair with the lightweight local `cross-encoder/ms-marco-MiniLM-L-6-v2` model, and returns the best eight chunks.

Dense, BM25, and RRF scores/ranks remain attached alongside the reranker score and rank.

```bash
python3 src/retrieve_hybrid_rerank.py \
  "How did Amazon's net sales change from 2023 to 2024?"
python3 src/evaluate_retrieval_rerank.py
```

The reranker was **rejected** for the current benchmark because it reduced Hit@K from **83.3% to 66.7%** and increased average retrieval latency from **45 ms to 169 ms**.

It also introduced regressions on `q04` and `q11`, while `q03` and `q12` remained unresolved.

The first run may download the reranker model. Timestamped results and `latest_summary.json` are written under `evals/results/retrieval_rerank/`.

## Evaluate full RAG

The full-RAG stage fixes retrieval at the best tested dense setting for the original generation comparison: 800-token chunks, 100-token overlap, and `top_k=8`.

### Local Ollama evaluation

```bash
python3 src/evaluate_full_rag.py --generator ollama --model qwen2.5:3b
```

Results are written under `evals/results/full_rag/`.

Retrieval metrics, generation latency/token metrics, and answer-quality metrics are reported separately. Correctness uses benchmark reference answers, completeness uses reference-answer recall, and citation/groundedness heuristics are labeled as heuristics.

For semantic review, add an optional structured 1–5 judge without changing deterministic retrieval scores:

```bash
python3 src/evaluate_full_rag.py --judge ollama --judge-model qwen2.5:3b
```

### OpenAI evaluation

```bash
python3 src/evaluate_full_rag_openai.py
```

The OpenAI benchmark completed successfully with `gpt-5-mini` and materially improved generation quality relative to Qwen 2.5 3B while using the same retrieval setup.

## Reproducibility and design choices

This project intentionally keeps retrieval, generation, and evaluation modular:

- retrieval can be benchmarked without paying for generation,
- local and hosted LLMs can be compared with the same evidence,
- failed experiments are preserved rather than hidden,
- deterministic retrieval metrics are not replaced by LLM judge scores,
- all major experiment outputs are written to separate timestamped result folders.

The project is designed as an evaluation-driven RAG study rather than a chatbot demo.
