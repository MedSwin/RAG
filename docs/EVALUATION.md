# System evaluation

Six cells execute the real v2 API under isolated cloud (`gpt-5.4`) and local MedSwin profiles. Every cell uses cloud `embed-v-4-0`; two-stage/full use `Cohere-rerank-v4.0-fast`. The runner validates `/health`, restarts between profiles, preserves artifacts, retries transient failures, and excludes degraded observations from aggregate quality statistics while reporting degradation rates.

TREC CDS 2016 is LIT-only with complete corpus/qrels and a fixed full-note plus diagnosis/test/treatment-question query. BioASQ uses authenticated BioASQ 13b data, a deterministic 30-per-type sample, and a closed LIT-only corpus. Retrieval uses Precision/Recall/nDCG at 5/10/20 and MRR@20; primary TREC endpoint is nDCG@10. BioASQ primary answer endpoint is ROUGE-SU4 F1 on 200-word ideal answers, with ROUGE-2 and type-specific exact metrics as diagnostics.

Paired comparisons use replicate-matched per-case means, 10,000-draw bootstrap intervals, paired sign-randomization, Holm correction, and 90% clean-pair coverage. Reports are automatic and do not include or imply clinician validation.

Sources: [TREC CDS overview](https://trec.nist.gov/pubs/trec25/papers/Overview-CL.pdf), [BioASQ datasets](https://participants-area.bioasq.org/datasets/), [TrialMatchAI](https://www.nature.com/articles/s41467-026-70509-w), [TrialGPT](https://www.nature.com/articles/s41467-024-53081-z), [BioASQ evaluation](https://pmc.ncbi.nlm.nih.gov/articles/PMC4450488/).
