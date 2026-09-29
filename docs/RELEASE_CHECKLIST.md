# Release checklist

- [ ] Replace anonymous citation metadata after de-anonymization.
- [ ] Confirm the intended open-source license with every code owner.
- [ ] Tag the exact paper commit.
- [ ] Publish checkpoint and data hashes without redistributing restricted data.
- [ ] Run `make compile`, `make test`, and `make lint` in a clean environment.
- [ ] Run GPU canaries before the full reproduction matrix.
- [ ] Freeze a machine-readable artifact bundle and regenerate all tables.
- [ ] Scan Git history for credentials, internal hostnames, and large binaries.
