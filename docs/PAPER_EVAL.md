# Paper evaluation

This is the governed publication path. Official TREC CDS 2016 retrieval (T1/T2) is separate from the clinician study (T3) and the automatic gate/MAC ablation (T4).

The MSAS eval portal (`eval/`, `:8200`, `eval --run`, homemade composites) is removed. Do not publish numbers from that path.

Executable contract: [`scripts/start-local.sh`](../scripts/start-local.sh) `paper-eval`, [`benchmarks/trec_cds2016/`](../benchmarks/trec_cds2016/), [`benchmarks/expert/`](../benchmarks/expert/).  
Pre-registration: [`benchmarks/trec_cds2016/PREREG.md`](../benchmarks/trec_cds2016/PREREG.md) (append-only after the first scored run).  
Clinician protocol: [`benchmarks/expert/PROTOCOL.md`](../benchmarks/expert/PROTOCOL.md).  
Metric table: [`../metrics.md`](../metrics.md).

---

## 1. What you may claim

| Table | Instrument | What it measures | What it does **not** measure |
| --- | --- | --- | --- |
| **T1** | Official NIST IR | Ranked PMC retrieval: infNDCG (primary), infAP, residual P@10, R-prec | Gate, MAC, generation, “CDSS SOTA” |
| **T2** | Same T1 runs | Diagnosis / test / treatment split (topics 1–10 / 11–20 / 21–30) | New scores or a confirmatory test |
| **T3** | Human study | Accept/abstain and citation support on packed product-path bundles | Official TREC relevance |
| **T4** | Automatic only | Action / missing-facet / contradiction rates for full vs `-gate` vs `-MAC` | Human confirmation of the ablation |

T1 is a **retriever** table. n=30 is track-defined and underpowered. Star only the three pre-registered infNDCG contrasts after BH-FDR.

Historical comparator: 2016 overview **Table 8** note-automatic medians (infNDCG 0.1228, P@10 0.1833), not Table 6. Summary-field runs are appendix sensitivity vs Table 10.

---

## 2. Operator

Default `paper-eval --stage all` is **T1 only** (warmup → prepare → emit → score). T3/T4 are opt-in.

```bash
./scripts/start-local.sh warmup
./scripts/start-local.sh paper-eval
./scripts/start-local.sh paper-eval --systems bm25,dense
./scripts/start-local.sh paper-eval --systems rrf,cascade --topic-field note
./scripts/start-local.sh paper-eval --stage emit --systems cascade
./scripts/start-local.sh paper-eval --stage score
./scripts/start-local.sh paper-eval --pipeline both --generator cloud --stage t3
./scripts/start-local.sh paper-eval --pipeline medswin --stage t4
./scripts/start-local.sh paper-eval --reset-full-corpus
```

`full-eval` is a deprecated alias that prints the new command and runs `paper-eval --stage all`.  
`eval` and `eval --run` exit 1 with the replacement text.

| Flag | Applies to | Notes |
| --- | --- | --- |
| `--stage warmup\|prepare\|emit\|score\|t3\|t4\|all` | paper-eval | Default `all` = T1 |
| `--systems bm25,dense,rrf,cascade\|all` | T1 emit | Illegal with `--stage t3` or `t4` |
| `--topic-field note\|summary` | T1 | Primary is `note` |
| `--pipeline naive\|medswin\|both` | T3/T4 | Illegal with `--stage emit` or `score` |
| `--generator cloud\|medswin\|both` | T3 | Frozen paper generator is Foundry GPT. Local 7B needs `--allow-local-t3` |
| `--reset-full-corpus` | prepare | Rebuild the isolated 1.25M corpus |
| `--allow-local-t3` | T3 | Exploratory 7B packs only |
| `--with-local-llm` | warmup / T3 | Download/serve MedSwin 7B |
| `--with-compose-api` | T3/T4 | Start `rag_api` with benchmark mounts |
| `--force-eval-warmup` / `--skip-eval-warmup` | warmup | Score still requires NIST files |

Warmup is **off** for ordinary `serve` / `console` / `ask` (`EVAL_WARMUP_ON_START=false`).

---

## 3. T1 — official retrieval

### Query (not `/chat`)

- One official field per run. Primary: **note** + Ely type question.
  - diagnosis: `What is the patient's diagnosis?`
  - test: `What tests should the patient receive?`
  - treatment: `How should the patient be treated?`
- Never concatenate note+description+summary.
- Never ingest the topic note as a literature document. T1 retrieve is `LIT_ONLY`.
- Exporter: `DenseRetriever` / `LexicalRetriever` / `EmbeddingClient(input_type=query)` / `RerankerClient`.

### Ranking

- Document unit: bare PMCID (max-pool chunks)
- Depth: 1000 unique PMCIDs
- \(k_{\mathrm{BM25}}=k_{\mathrm{dense}}=4000\)
- Hybrid: RRF \(k=60\)
- Cascade: rerank top **300** unique PMCIDs (best chunk each); fill 301–1000 from the unreranked RRF tail without reordering the head
- Production fusion / EBM / safety weights are not T1 scores

Run names (≤12 chars): `msbm25note`, `msdensnote`, `msrrfnote`, `mscascnote`. Summary variants: `msbm25summ`, …

### Official scoring

NIST files are SHA256-pinned in `PREREG.md` and `benchmarks/trec_cds2016/nist.py`. Do not score `ir_datasets` qrels as inferred measures.

```text
perl sample_eval.pl -q qrels-sampleval-2016.txt RUN
trec_eval -q -c -M1000 qrels-treceval-2016.txt RUN
```

Report infNDCG, infAP, residual P@10, R-prec. Do not report `iP10` as P@10. Do not use `trec_eval -J` as official. Keep topic 22.

### Confirmatory tests

Three paired infNDCG contrasts only: BM25 vs RRF, dense vs RRF, RRF vs cascade. Primary p = paired randomization; also paired t. 95% CI = bootstrap over 30 topics. FDR = Benjamini–Hochberg over those three. P@10 / R-prec / infAP and T2 are descriptive.

---

## 4. T3 — clinician study (separate experiment)

Product path, not the T1 exporter:

1. Ingest the official **note** as EMR (`patient_id=trec-cds-{n}`)
2. Query = type question only
3. `source_policy=ANY`, `include_patient_context_in_query=false`
4. Frozen generator: Foundry GPT
5. Persist the immutable pack before rating

| Task | Stimulus | Label |
| --- | --- | --- |
| A | Note + type question + packed snippets. No system name, no answer | `answer` / `abstain` for *this* bundle |
| B | Atomic claims from the clinician-visible answer (cap 8; abstain rationale cap 3) vs **cited** snippets | supported / unsupported / contradictory |

TREC has no abstain gold. Naive always answers. Two clinicians + one adjudicator; report Cohen κ (and Randolph κ if prevalence is extreme).

---

## 5. T4 — automatic ablation

Same T3 packs under `full`, `no_gate` (`constraints.disable_gate`), and `no_mac` (`constraints.disable_mac`). Report action / missing-facet / contradiction rates + FDR. Human T4 is not confirmatory: those conditions change the bundle, so Task A gold is not reusable.

---

## 6. Corpus and runtime

- Dataset: `pmc/v2/trec-cds-2016` (1,255,260 documents)
- Tenant: `BENCHMARK_ORG_ID=bench-org`
- Isolated artifacts under `FULL_EVAL_DATA_DIR` (default `./data/full-trec-benchmark`): HNSW, SQLite mapping, BM25 FTS5
- Mongo: prefer Compose `medswin-mongodb` / `mongo:7.0`
- Paper API for T3/T4: `FULL_EVAL_API_PORT` (default 8110)
- `trec_eval`: `TREC_EVAL_BIN`, PATH, or build v9.0.8 into `benchmarks/trec_cds2016/bin/`
- NIST dir: `NIST_DIR` or `benchmarks/trec_cds2016/nist/`

Prepare is fail-closed: exact document count, chunker SHA, embedding space, BM25, and HNSW identity fingerprints. See `benchmarks/trec_cds2016/prepare/`.

---

## 7. Modules (one-word Python names)

| Path | Role |
| --- | --- |
| `benchmarks/trec_cds2016/emit.py` | T1 run-file exporter |
| `benchmarks/trec_cds2016/validate.py` | qid 1–30, bare PMCID, depth, scores |
| `benchmarks/trec_cds2016/score.py` | Official two-tool scoring + T2 |
| `benchmarks/trec_cds2016/contrast.py` | Pre-registered infNDCG contrasts |
| `benchmarks/trec_cds2016/prepare/mongo.py` | Tenant-safe Mongo preflight |
| `benchmarks/trec_cds2016/prepare/runtime.py` | 1.25M corpus builder |
| `benchmarks/trec_cds2016/prepare/materialize.py` | Document metadata layer |
| `benchmarks/trec_cds2016/prepare/verify.py` | Fail-closed corpus verifier |
| `benchmarks/expert/t3_packs.py` | Product-path packs (name exception) |
| `benchmarks/expert/t4_automatic.py` | Automatic T4 rates (name exception) |

`benchmarks/trec_cds2016/runtime.py` binds isolated artifact env. It is not the corpus builder.

---

## 8. Honesty checklist

1. Record `git rev-parse HEAD` and non-secret paper-eval env (`CLOUD_MODE`, embedding, `FULL_EVAL_*`, sufficiency thresholds).
2. Verify NIST SHA256 pins before scoring.
3. T1 run files pass `validate.py`. Score only with the two official commands above.
4. Do not write “official TREC shows the gated multi-agent CDSS is SOTA.”
5. Do not publish MedQuAD ROUGE or HealthBench lexical overlap as the primary system result.
6. T3 packs must use the product path (note as EMR, type question as query).
7. T4 is automatic; do not treat a human −gate/−MAC rating as confirmatory.
