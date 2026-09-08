# Setup

Set `CLOUD_MODE=true`, Foundry credentials, `CLOUD_EMBEDDING=embed-v-4-0`, and `CLOUD_RERANKER=Cohere-rerank-v4.0-fast`. Configure `MEDSWIN_LLM_URL` and `MEDSWIN_LLM_MODEL=MedSwin/MedSwin-DaRE-TIES-KD-0.7` for local-generator cells. Evaluation also requires `BIOASQ_USERNAME`, `BIOASQ_PASSWORD`, and `NCBI_EMAIL`; credentials are never stored in artifacts.

TREC uses the complete PMC collection. BioASQ preparation downloads Training 13b from the participants area (ZIP), deterministically samples 120 questions (30 per type), and creates a separate closed LIT-only corpus via NCBI. Keep ordinary, TREC, and BioASQ stores isolated.

On memory-constrained hosts, start the API with `DISABLE_DATASET_PRELOAD=1` so Hugging Face dashboard preload does not OOM the process.
