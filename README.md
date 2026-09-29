# Writable Memory Needs Write Authority

Official research code for studying persistent test-time updates in long
self-generated streams. The repository contains the TTT-E2E PyTorch runtime,
controlled language-model experiments, Settlement variants, and the ALFWorld
agent evaluation used by the accompanying paper.

The release is organized around four reproducibility rules:

1. every result is produced by a versioned YAML manifest or an explicit CLI;
2. checkpoints and corpora are external, immutable inputs identified by hash;
3. probes are read-only and independent evaluation text never enters updates;
4. books or tasks, rather than repeated sampling seeds, are the statistical units.

## Repository layout

```text
ttt_pt/        TTT model, fast-weight state, streaming decoder, training code
scripts/       paper experiment entry points
configs/       versioned, inspectable experiment manifests
analysis/      aggregation and confidence-interval scripts
tests/         CPU unit tests
validation/    explicit GPU/checkpoint validation programs
cluster/       optional scheduler templates; no site credentials
docs/          protocol, data, and reproducibility documentation
schemas/       machine-readable result schema
```

## Installation

Python 3.11 or 3.12 is recommended. Install the PyTorch wheel matching the CUDA
runtime first, then install this repository:

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -e '.[language,dev]'
```

For ALFWorld experiments:

```bash
pip install -e '.[language,agents,dev]'
alfworld-download
```

Copy `.env.example` and export the paths in it. Checkpoints, PG-19 tokens,
ALFWorld data, and generated trajectories are intentionally not committed.

## Validate the installation

```bash
make compile
make test
python scripts/selfcheck.py
```

The GPU invariants should pass before any long run:

```bash
python scripts/gate_canaries.py --ckpt "$TTT_CKPT/125m-ext32k.pt" \
  --val "$TTT_DATA/pg19/val.npy"
python scripts/state_probe_audit.py --ckpt "$TTT_CKPT/125m-ext32k.pt" \
  --val "$TTT_DATA/pg19/val.npy"
python scripts/stream_parity.py --ckpt "$TTT_CKPT/125m-ext32k.pt" \
  --val "$TTT_DATA/pg19/val.npy"
```

## Reproduce the canonical experiment

Inspect a manifest before running it:

```bash
python scripts/run_config.py configs/canonical_closed.yaml --dry-run
python scripts/run_config.py configs/canonical_masked.yaml --dry-run
```

Then run the two arms:

```bash
python scripts/run_config.py configs/canonical_closed.yaml
python scripts/run_config.py configs/canonical_masked.yaml
```

The canonical protocol uses 8 books, 5 sampling seeds, 128 chunks, eight real
prefill writes, branch-only clean probes, temperature 1, and top-p 0.95. See
[`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) for all protocol definitions.

## Main experiment entry points

| Question | Entry point |
|---|---|
| Closed loop vs writes off and fixed generation | `scripts/horizon.py` |
| Per-write prospective transfer | `scripts/preq_obs.py` |
| Decoder and causal mechanism controls | `scripts/mechanism_pilot.py` |
| Real-text exposure-density boundary | `scripts/exposure_density_sweep.py` |
| Damage–adaptation update-strength frontier | `scripts/update_strength_sweep.py` |
| Mixed-source Settlement | `scripts/settlement_mixed.py` |
| Independent-source heavy tail | `scripts/acceptance_heavy_tail.py` |
| Hidden-provenance benchmark | `scripts/acceptance_hidden_provenance.py` |
| Long-stream retrieval | `scripts/acceptance_retrieval_3b.py` |
| ALFWorld online/prequential evaluation | `scripts/alfworld_agentbench.py` |
| WebShop agent policies | `scripts/ws_arms.py` |
| ScienceWorld agent policies | `scripts/sw_arms.py` |

## Artifact policy

Raw model weights and corpora are too large and may have upstream licenses, so
this repository publishes their required names, hashes, selection rules, and
schemas rather than redistributing them. A run is comparable only when its
checkpoint hash, corpus identity, book selection, probe schedule, decoder, and
result schema agree. Do not merge outputs based only on filenames.

## Citation and license

Citation metadata is in `CITATION.cff`. Code is released under the MIT License.
Dataset and checkpoint licenses remain those of their original providers.
