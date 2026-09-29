# Results

This source release intentionally separates code from mutable manuscript
numbers. Machine-readable outputs belong under `results/` and are ignored by
Git. A public artifact bundle should include raw JSON, its manifest, hashes,
and the analysis command that produced each table or figure.

No value should be copied into this file until it is regenerated from a frozen
artifact bundle and checked against the paper's protocol table.
