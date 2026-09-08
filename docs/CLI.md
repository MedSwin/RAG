# CLI

```bash
./scripts/start-local.sh ask --mode all --question "..."
./scripts/start-local.sh system-eval
./scripts/start-local.sh system-eval --pipeline naive --generator cloud
./scripts/start-local.sh system-eval --phase prepare
./scripts/start-local.sh system-eval --custom combination-september
./scripts/start-local.sh system-eval --smoke
```

The September preset runs naive and two-stage under both generators. Smoke output is visibly marked non-benchmark output.
