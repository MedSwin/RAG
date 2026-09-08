# Setup

Set `CLOUD_MODE=true`, Foundry credentials, `CLOUD_EMBEDDING=embed-v-4-0`, and `CLOUD_RERANKER=Cohere-rerank-v4.0-fast`. Configure `MEDSWIN_LLM_URL` and `MEDSWIN_LLM_MODEL=MedSwin/MedSwin-DaRE-TIES-KD-0.7` for local-generator cells. Evaluation also requires `BIOASQ_USERNAME`, `BIOASQ_PASSWORD`, and `NCBI_EMAIL`; credentials are never stored in artifacts.

TREC uses the complete PMC collection. BioASQ preparation downloads the registered 13b golden-enriched source, deterministically samples 120 questions, and creates a separate closed LIT-only corpus. Keep ordinary, TREC, and BioASQ stores isolated.
