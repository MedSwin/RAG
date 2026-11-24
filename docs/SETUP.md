# Setup

Set `CLOUD_MODE=true` and `CLOUD_PROVIDER=bedrock`. Chat uses `global.openai.gpt-6-luna`. Embeddings use `global.cohere.embed-v4:0` at 1536 dimensions. Rerank uses `cohere.rerank-v3-5:0`. Credentials come from the AWS chain. Do not put a static key in the repo.

Chat and embeddings use the global profiles in `AWS_BEDROCK_REGION`. Cohere Rerank 3.5 uses `AWS_BEDROCK_RERANK_REGION`, because that model is not in every Region. GPT-6 Luna does not accept a temperature field.

Set `CLOUD_PROVIDER=foundry` only for the older Azure profile. That profile uses `CLOUD_EMBEDDING=embed-v-4-0` and `CLOUD_RERANKER=Cohere-rerank-v4.0-fast`. Configure `MEDSWIN_LLM_URL` and `MEDSWIN_LLM_MODEL=MedSwin/MedSwin-DaRE-TIES-KD-0.7` for a local generator. A Bedrock index and a local MedEmbed index are different vector spaces. Rebuild the index after a provider change. Evaluation also requires `BIOASQ_USERNAME`, `BIOASQ_PASSWORD`, and `NCBI_EMAIL`. Credentials are never stored in artifacts.

TREC uses the complete PMC collection. BioASQ preparation downloads Training 13b from the participants area (ZIP), deterministically samples 120 questions (30 per type), and creates a separate closed LIT-only corpus via NCBI. Keep ordinary, TREC, and BioASQ stores isolated.

On memory-constrained hosts, start the API with `DISABLE_DATASET_PRELOAD=1` so Hugging Face dashboard preload does not OOM the process.
