# Release checklist

- [ ] Replace anonymous citation metadata after de-anonymization.
- [ ] Confirm the intended open-source license with every code owner.
- [ ] Tag the exact paper commit.
- [ ] Publish checkpoint and data hashes without redistributing restricted data.
- [ ] Run `make check` and `make suite-dry-run` in a clean environment.
- [ ] Run `make smoke` with the real 125M checkpoint and PG-19 validation array.
- [ ] Run `python scripts/selfcheck.py --profile all --full` when publishing the optional WebShop path.
- [ ] Freeze a machine-readable artifact bundle and regenerate all tables.
- [ ] Scan Git history for credentials, internal hostnames, and large binaries.
