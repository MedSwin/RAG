Locked publication metrics. Operator: [`docs/PAPER_EVAL.md`](docs/PAPER_EVAL.md). Pre-registration: [`benchmarks/trec_cds2016/PREREG.md`](benchmarks/trec_cds2016/PREREG.md).

| Title | Metrics | Explanation | Cites |
|---|---|---|---|
| **T1 Official retrieval** | infNDCG (primary), infAP, P@10, R-prec | Ranked PMC retrieval on TREC CDS 2016. Score 1000-PMCID run files with NIST `sample_eval` / `trec_eval`. Not packed chat context. | Roberts et al., TREC 2016 CDS overview; Yilmaz et al., SIGIR 2008; Zhang et al., *BMC Bioinformatics* 2023 |
| **T2 Topic split** | Same four, by diagnosis / test / treatment | Official 2016 topic types (topics 1–10 / 11–20 / 21–30). Same runs, no new scores. | Roberts et al., TREC 2016 CDS overview |
| **T3 Clinician decisions** | Accept/abstain accuracy; false-answer; citation support; κ | Humans label whether the system should answer or abstain, and whether each cite is supported. TREC has no abstain gold. | Abdallah et al., *Nat Commun* 2026 (TrialMatchAI); Jin et al., *Nat Commun* 2024 (TrialGPT); Cohen 1960 / Randolph κ in Singhal et al., *Nat Med* 2024 |
| **T4 Gate / MAC ablation** | T3 metrics only (−gate, −MAC, full) | Agents and the sufficiency gate are claimed here, not on infNDCG. | TrialMatchAI ablation + FDR; Med-PaLM axes only if you fund a small rubric (Singhal et al., *Nature* 2023) |