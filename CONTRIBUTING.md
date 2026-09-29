# Contributing

Use a separate output directory for every run. Do not commit checkpoints, corpora,
raw trajectories, credentials, host names, or scheduler state. New experiments
must include a YAML manifest, an atomic JSON output, a configuration signature,
and a CPU-only test for any new analysis logic.

Before opening a pull request, run:

```bash
make compile
make test
make lint
```
