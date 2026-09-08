# Metric registry

Precision@K is relevant documents in top K divided by K; Recall@K divides recovered relevant documents by all known relevant documents; nDCG@K uses `2^rel−1` and log2 rank discount; MRR@20 is the reciprocal first relevant rank.

BioASQ ideal answers use ROUGE-2 and ROUGE-SU4 after deterministic normalization and a 200-word limit. Type-specific yes/no, factoid, and list metrics are secondary. Automated NLI claim diagnostics are never clinician validation.
