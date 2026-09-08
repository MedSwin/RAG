# Administration

Ingest documents through `POST /api/v1/storage/chunks`, refresh embeddings through `/api/v1/storage/embeddings/refresh`, and build the ANN index through `/api/v1/storage/index/build`. Maintain separate Mongo organizations/index/data paths for ordinary use, TREC, and BioASQ evaluation. `/health` reports API contract `2.0`, supported pipelines, generator identity, cloud embedding/reranker identities, local revision, and evaluation-mode state.
