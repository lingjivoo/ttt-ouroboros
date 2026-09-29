# Self-Generated Feedback Destabilizes Test-Time Training

Reproduction code for **“Self-Generated Feedback Destabilizes Test-Time
Training: A Causal Decomposition of Long-Horizon Adaptation.”** The repository
contains the native TTT-E2E runtime, the controlled long-horizon interventions,
Settlement, the WebShop task experiment, audited per-book data and the scripts
used to build the submitted figures.

The release is intentionally narrow. ALFWorld, ScienceWorld, retrieval and
other exploratory experiments that are absent from the paper are excluded.

## What can be reproduced

| Result family | Runner | Status |
|---|---|---|
| 125M/760M/3B Closed Loop vs Writes Off | `scripts/horizon.py` | released |
| Fixed Generation and replay controls | `scripts/horizon.py`, `scripts/mechanism_pilot.py` | released |
| Single-update transfer and gradient conflict | `scripts/preq_obs.py` | released |
| Exposure-density boundary | `scripts/exposure_density_sweep.py` | released |
| aTTT, write-dose and anchor controls | `scripts/attt_closed_loop.py`, `scripts/update_strength_sweep.py`, `scripts/anchor_causal.py` | released |
| Language-model Settlement | `scripts/deferred.py`, `scripts/settlement_mixed.py` | released |
| WebShop causal comparison | `scripts/ws_arms.py` | released |
| Qwen3-4B real-text utility | `scripts/qwen_reviewer_p0c.py` | released |
| Qwen3-4B long-horizon raw grid | — | audited aggregate only; original launcher not recovered |
| Source-label corruption raw grid | — | audited aggregate only; original launcher not recovered |

The exact mapping from each paper table or figure to code and data is in
[`docs/PAPER_REPRODUCTION.md`](docs/PAPER_REPRODUCTION.md).

## Repository layout

```text
ttt_pt/       TTT model, fast-weight state, streaming decoder and training code
scripts/      experiment runners, release checks and suite orchestration
configs/      versioned manifests and ordered experiment suites
analysis/     paired aggregation and book-bootstrap summaries
figures/      final figure generators
data/         audited aggregate input for the canonical paper curves
validation/   conversion, streaming and state-restoration checks
tests/        CPU tests for mechanics, manifests and release workflows
```

## 1. Create the environment

The reference environment uses Python 3.11 and CUDA 12.4. PyTorch 2.5–2.9 is
supported; the release GPU smoke test also runs under PyTorch 2.9.

### Conda

```bash
conda env create -f environment.yml
conda activate ttt-feedback
pip install -e .
```

### Existing CUDA/PyTorch environment

Install the CUDA-matched PyTorch wheel first, then:

```bash
pip install -e '.[language,dev]'
```

For WebShop:

```bash
pip install -e '.[language,agents,dev]'
```

### Docker

```bash
docker build -t ttt-feedback .
docker run --gpus all --rm \
  -v /absolute/data:/data:ro \
  -v /absolute/checkpoints:/checkpoints:ro \
  -v "$PWD/results:/workspace/ttt-feedback/results" \
  -e TTT_DATA=/data -e TTT_CKPT=/checkpoints \
  ttt-feedback --suite configs/suites/main_125m.yaml --smoke
```

## 2. Configure artifacts

```bash
cp env.sh.example env.sh
# Edit absolute paths.
source env.sh
```

Expected language artifacts:

```text
$TTT_DATA/pg19/val.npy
$TTT_CKPT/125m-ext32k.pt
```

`val.npy` is the tokenized PG-19 validation stream with book boundaries. Model
weights and corpora are external because of size and upstream licenses. See
[`docs/DATA.md`](docs/DATA.md) for Qwen and WebShop preparation.

## 3. Validate before a long run

```bash
make check
python scripts/selfcheck.py --profile language --full
```

`--full` loads the actual checkpoint on CUDA and performs one 1024-token update.
Then run the three-arm integration smoke test:

```bash
python scripts/run_paper_suite.py \
  --suite configs/suites/main_125m.yaml \
  --smoke
```

Smoke outputs use one physical row, one seed and 16 chunks. They verify the full
load → prefill → generate → update → branch-probe → save path and are not paper
results. The completed release validation is recorded in
[`docs/VALIDATION.md`](docs/VALIDATION.md).

## 4. Run the canonical 125M experiment

```bash
python scripts/run_paper_suite.py \
  --suite configs/suites/main_125m.yaml
```

The full suite runs Closed Loop, Writes Off and Fixed Generation. It maintains
logical width 8 over physical rows/books 0–7, then reports screened books 2–7.
Each arm uses seeds 42, 1, 7, 2 and 3, 128 chunks and 16 branch-only probes.
Completed seeds are resumed safely from signed JSON outputs.

The scale-comparison manifests use the same physical width, books, seeds and
probe schedule:

```bash
python scripts/run_paper_suite.py --suite configs/suites/main_760m.yaml
python scripts/run_paper_suite.py --suite configs/suites/main_3b.yaml
```

They expect `760m-ext32k.pt` and `3b_128k_pt.pt` under `TTT_CKPT`. The 3B suite
is a true logical-width-8 run and therefore requires a high-memory GPU.

Aggregate the paired results:

```bash
python analysis/summarize_canonical.py \
  --closed "$TTT_OUT/canonical/closed.json" \
  --writes-off "$TTT_OUT/canonical/writes_off.json" \
  --fixed "$TTT_OUT/canonical/fixed_generation.json" \
  --out "$TTT_OUT/canonical/SUMMARY.md"
```

The summary first averages seeds within each book and bootstraps the six books.
It rejects outputs with the wrong width, book offset, probe count or protocol.

To inspect commands without launching GPU work:

```bash
make suite-dry-run
python scripts/run_config.py configs/canonical_closed.yaml --dry-run
```

## 5. Other paper experiments

Each runner exposes complete CLI documentation:

```bash
python scripts/exposure_density_sweep.py --help
python scripts/preq_obs.py --help
python scripts/deferred.py --help
python scripts/attt_closed_loop.py --help
python scripts/update_strength_sweep.py --help
python scripts/anchor_causal.py --help
```

The shared protocol and the reason some suites use separate long-book sets are
specified in [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md). Never pool canonical,
exposure-density and Settlement endpoints solely because all are measured in
nats.

## 6. WebShop

Set `WEBSHOP_DIR` to the upstream WebShop checkout and build the deterministic
catalogue described in [`docs/DATA.md`](docs/DATA.md). The paper arms are:

```bash
python scripts/ws_arms.py --policy none       --stream-seed 0 --out "$TTT_OUT/webshop/off_s0.json"
python scripts/ws_arms.py --policy uniform    --stream-seed 0 --out "$TTT_OUT/webshop/closed_s0.json"
python scripts/ws_arms.py --policy fixed      --stream-seed 0 --out "$TTT_OUT/webshop/fixed_s0.json"
python scripts/ws_arms.py --policy settlement --stream-seed 0 --out "$TTT_OUT/webshop/settlement_s0.json"
```

Repeat stream seeds 0–4. All arms process 900 training episodes and the same 150
held-out goals. Settlement evaluates a temporary 25-episode candidate on 10
disjoint validation goals.

## 7. Regenerate figures

```bash
make figures
```

`figures/make_paper_figures.py` reads the committed audited per-book JSON for the
unified canonical trajectory. Other figure scripts contain clearly marked final
audited aggregates when the original raw launcher was not recovered.

## Reproducibility rules

- Checkpoint, corpus, book selection, seed set, decoder and probe schedule are
  part of a result's identity.
- Probes snapshot and restore all carried state; evaluation text is read-only.
- Books or held-out tasks are independent units; repeated sampling seeds are
  averaged within them.
- JSON output is resumable only when its configuration signature matches.
- `SOURCE_MANIFEST.json` contains SHA-256 and byte size for every released file.

Run `make check` and `python scripts/source_manifest.py --check` before creating
a release. Code is MIT licensed; checkpoints and datasets retain their upstream
licenses.
