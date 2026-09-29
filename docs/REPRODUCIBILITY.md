# Reproducibility contract

Every released result must record:

- source commit and dirty-tree status;
- complete CLI or YAML manifest;
- checkpoint and dataset SHA256;
- model preset and numerical dtype;
- book/task identities and their selection rule;
- seeds, decoder, horizon, probe positions, and batch width;
- start/end timestamps and hardware model;
- per-book or per-task observations, not only an aggregate.

Generation is not assumed bitwise deterministic. Teacher-forced controls should
be deterministic; generated trajectories should be preserved when exact replay
is required. Atomic JSON writes prevent preemption from leaving a valid-looking
partial file. A reader must ignore metadata keys beginning with `_` when
iterating result cells and must reject mismatched configuration signatures.

Statistical intervals use books or tasks as the independent unit. Sampling
seeds characterize generation variability but do not create new books.
