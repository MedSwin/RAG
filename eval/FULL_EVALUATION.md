# MedSwin Complete TREC-CDS 2016 Evaluation

This document defines the strict publication-evaluation path implemented by `scripts/start-local.sh full-eval`. It is deliberately separate from the ordinary eval UI and the sampled corpus utilities. A smoke run is useful for development, but it must never be reported as the complete TREC-CDS benchmark described here.

## Completion contract

A run is "complete" only when all of the following hold.

### 1. Source data and model warmup

The startup warmup must successfully verify:

- Hugging Face model: `MedSwin/MedSwin-DaRE-TIES-KD-0.7`.
- Dataset: `pmc/v2/trec-cds-2016`.
- Exactly 1,255,260 PMC collection documents are readable.
- Exactly 30 TREC-CDS 2016 topics are readable.
- Exactly 37,707 qrels are readable.
- Azure AI Foundry GPT deployment: `gpt-5.4` by default.
- Foundry embedding deployment: `embed-v-4-0` by default.
- Foundry reranker deployment: `Cohere-rerank-v4.0-fast` by default.
- Query and document embedding probes both return the configured vector dimension.
- The reranker returns one score per supplied passage.
- GPT returns non-empty completion content.

Warmup writes machine-readable evidence under `data/eval-warmup/` and fails instead of silently falling back when a required cloud service is unavailable.

### 2. Complete corpus preparation

`eval/scripts/prepare_full_trec_runtime.py` must:

- read every TREC-CDS PMC collection document;
- construct document text from title, abstract, and the complete body rather than a fixed opening excerpt;
- apply the repository's production `app.medswin.chunking.section_chunks` chunker;
- embed every resulting corpus chunk through Cohere Embed v4 with `input_type=document`;
- persist embedding model, space, and dimension metadata on every chunk;
- materialize all 30 TREC case notes as patient-scoped EMR chunks in the same vector space;
- build SQLite FTS5 `bm25()` over every persisted benchmark chunk;
- build HNSW incrementally over every active vector;
- store the HNSW label mapping in SQLite so millions of mappings do not have to be loaded into Python memory;
- preserve all positive qrels in the benchmark case JSONL without the sampled-preparation qrel cap.

The publication path intentionally does not accept a corpus sample-size argument.

### 3. Independent persisted-runtime verification

`eval/scripts/verify_full_trec_runtime.py` runs after preparation and does not trust the ingestion checkpoint alone. It independently verifies the current persisted state:

- exactly 1,255,260 distinct `LIT` document IDs exist for the benchmark org;
- all 30 benchmark patient contexts exist;
- every persisted benchmark chunk has a non-empty active `embed-v-4-0` vector with the configured dimension;
- every positive-qrel document referenced by the 30 cases exists;
- SQLite BM25 row count equals the Mongo benchmark chunk count;
- HNSW element count equals the Mongo benchmark chunk count;
- SQLite HNSW mapping count equals the Mongo benchmark chunk count;
- runtime manifest reports all 30 topics and 37,707 qrels;
- corpus embedding intent is recorded as `document`.

This check catches a completed checkpoint whose persisted corpus or index was later deleted or corrupted.

## Evaluation matrix

The strict matrix always evaluates all 30 cases. It reuses the same prepared corpus/index and swaps only the requested pipeline/generator dimensions. The default (`--pipeline both`) is the four-cell publication matrix. `--pipeline naive_rag` or `--pipeline medswin` still runs both generators and all 30 topics, but the resulting artifact is a subset: `strict_pass` means the selected cells passed architecture checks, while `publication_complete` is `true` only for the full 2x2.

| Cell | Retrieval/system | Generator |
| --- | --- | --- |
| A | naive RAG | local `MedSwin/MedSwin-DaRE-TIES-KD-0.7` |
| B | naive RAG | Foundry `gpt-5.4` |
| C | full MedSwin | local `MedSwin/MedSwin-DaRE-TIES-KD-0.7` |
| D | full MedSwin | Foundry `gpt-5.4` |

All online query embeddings are generated with `input_type=query` against the corpus created with `input_type=document`.

### Naive-RAG contract

A naive cell fails when any case does not satisfy all of these conditions:

- response pipeline is `naive_rag`;
- retrieval backend is ANN, not Mongo fallback or an empty/error backend;
- retrieved evidence is non-empty;
- generated answer is non-empty;
- the trace contains `retrieval.naive_dense`;
- there are no reranker traces;
- there are no MAC agent tool calls;
- there are no sufficiency checks;
- the response is not degraded.

This makes the baseline specifically `query embedding -> ANN dense top-k -> generation`. It does not accidentally inherit BM25, reranking, MAC, or the evidence gate.

### Full MedSwin contract

A full-system cell fails when any case does not satisfy all of these conditions:

- response pipeline is `medswin`;
- hybrid retrieval traces are present;
- reranker traces are present and contain scores;
- `retrieval.rerank` was executed and did not return the repository's `rerank-error` fail-open marker;
- sufficiency checks, policy decision, sufficiency decision, and facet matrix are present;
- the EMR, guideline, safety, quality, and critic specialist agents all have tool-call and message provenance;
- no required specialist reports degraded/error execution;
- the response is not degraded;
- answer content is non-empty.

The full runtime also contains bounded exact dense recovery for small source-scoped EMR/CPG/safety pools. This prevents a global literature-dominated ANN top-k from starving patient-scoped specialist evidence after metadata filtering.

## Running it

Create `.env` from `env.example` and supply at least the Azure Foundry account endpoint/API key plus any Hugging Face authentication required by the model repository.

```bash
./scripts/start-local.sh full-eval
./scripts/start-local.sh full-eval --pipeline naive_rag
```

`--pipeline` is forwarded into `eval/scripts/run_full_matrix.py`. Omitting it keeps the publication 2x2. A naive-only or MedSwin-only run still pays the complete-corpus warmup/index cost; it only skips the unselected chat cells.

To deliberately discard an existing benchmark org and rebuild every persisted chunk/index from zero:

```bash
./scripts/start-local.sh full-eval --reset-full-corpus
```

The command performs, in order:

1. strict model/dataset/Foundry warmup;
2. complete corpus chunking and cloud document embedding;
3. complete BM25/HNSW construction;
4. independent persisted-runtime verification;
5. local MedSwin generation-server validation;
6. the selected 30-case evaluation cells (four cells unless `--pipeline` filters them);
7. strict per-case architecture validation and aggregate audit output.

Per-cell and matrix JSON outputs are written under `RUN_STORE_DIR` (default `/tmp/medswin-audits`). Selected cells are acceptable only when `strict_pass` is `true`. Treat the run as the complete TREC system audit only when `publication_complete` is also `true`.

## Important distinction for publication

The strict matrix is a whole-system MedSwin audit: it proves the requested corpus, retrieval, reranking, MAC, gating, and generation paths actually executed and records the repository's system-level metrics. It should not be described as a replacement for the official TREC retrieval leaderboard protocol unless standard TREC ranked-run metrics (for example MAP/nDCG/P@k using the official qrels) are also generated and reported separately.

Likewise, code review, compilation, or GitHub CI success is not evidence that the million-document cloud evaluation itself completed. Publication evidence is the persisted-runtime verification artifact plus a four-cell matrix whose `strict_pass` and `publication_complete` are both `true`, produced on the target machine with real Azure credentials, storage, network access, and sufficient model compute.
