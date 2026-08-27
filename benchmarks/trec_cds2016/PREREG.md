# TREC CDS 2016 pre-registration

This file is append-only after the first scored paper run.

## Track

- TREC Clinical Decision Support 2016 only
- Corpus: 28 March 2016 PMC OA snapshot (`pmc/v2`, 1,255,260 documents)
- Automatic runs. `type` is used programmatically and stays automatic.
- No tuning on 2016 qrels.

## T1 query

- One official field per run.
- Primary: **note** + Ely type question
  - diagnosis: `What is the patient's diagnosis?`
  - test: `What tests should the patient receive?`
  - treatment: `How should the patient be treated?`
- Appendix sensitivity only: summary + same type question vs overview Table 10.
- Never concatenate note+description+summary.
- Never ingest the topic note as a literature document. T1 retrieve is `LIT_ONLY`.

## Ranking

- Run-file validator: qid 1-30, bare PMCID, ≤1000 unique, monotonic score, run name ≤12
- Document unit: bare PMCID
- Chunk → document: **max-pool**
- Depth: 1000 unique PMCIDs
- BM25 list depth \(k_{\mathrm{BM25}}=4000\)
- Dense list depth \(k_{\mathrm{dense}}=4000\)
- Hybrid: Reciprocal Rank Fusion, \(k=60\) (Cormack, Clarke, Buettcher, SIGIR 2009)
- Cascade: RRF to 1000; pointwise rerank top **300** unique PMCIDs (best chunk each); fill 301–1000 from the unreranked RRF tail without reordering the head
- Production fusion / EBM / safety weights are **not** T1 scores

## Official files (SHA256)

| File | SHA256 |
| --- | --- |
| `topics2016.xml` | `167541d16ab0986fd36045fb4e1104fccb6c8df18e1842b7ee448e7639479767` |
| `qrels-treceval-2016.txt` | `285fcf088b81ea3ad926b054d92aaddf24b148287f9f574c8edccbf21b5ed3ac` |
| `qrels-sampleval-2016.txt` | `f3617dcdd37b00aae48a943e437a2059fe629c11fb737fc39530a29f41dd82e2` |
| `sample_eval.pl` | `8af44fab50fed7ae8d00c75cc0358ccd7242274f213d9dfa805fa64c74c069f4` |

## Official commands

```text
perl sample_eval.pl -q qrels-sampleval-2016.txt RUN
trec_eval -q -c -M1000 qrels-treceval-2016.txt RUN
```

Report: infNDCG (primary), infAP, residual P@10, R-prec. Do not report `iP10` as P@10. Do not use `trec_eval -J` as official. Historical comparator: overview **Table 8** note automatic medians (infNDCG 0.1228, P@10 0.1833), not Table 6.

## Confirmatory IR tests

Family of three paired contrasts on **infNDCG only**:

1. BM25 vs RRF
2. dense vs RRF
3. RRF vs RRF+cascade

Test: paired randomization (primary p) and paired t. 95% CI = bootstrap over 30 topics. FDR = Benjamini–Hochberg over those three. P@10 / R-prec / infAP and the diagnosis/test/treatment split are descriptive.

Keep topic 22.

## T3 (separate experiment)

- Generator frozen: Foundry GPT
- Stimulus: note ingested as EMR; query = type question only
- Task A: evidence-conditioned accept/abstain on the packed bundle (no answer shown)
- Task B: atomic claims (cap 8) vs cited snippets
- Human T4 is not confirmatory
