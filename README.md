# MedSwin 2.0

MedSwin is an evidence-grounded clinical decision-support RAG system with comparable naive dense RAG, two-stage hybrid/reranked/gated RAG, and full MAC-enabled pipelines.

Copy [`env.example`](env.example) to `.env`, configure MongoDB, Foundry, Cohere, the local MedSwin model, and BioASQ credentials, then run:

```bash
./scripts/start-local.sh serve
./scripts/start-local.sh system-eval
./scripts/start-local.sh system-eval --custom combination-september
```

| Pipeline | Cloud | Local MedSwin |
| --- | --- | --- |
| Naive | `--pipeline naive --generator cloud` | `--pipeline naive --generator medswin` |
| Two-stage | `--pipeline two-stage --generator cloud` | `--pipeline two-stage --generator medswin` |
| Full | `--pipeline full --generator cloud` | `--pipeline full --generator medswin` |

See [`docs/SETUP.md`](docs/SETUP.md), [`docs/CLI.md`](docs/CLI.md), [`docs/PIPELINES.md`](docs/PIPELINES.md), [`docs/API.md`](docs/API.md), and [`docs/EVALUATION.md`](docs/EVALUATION.md).
