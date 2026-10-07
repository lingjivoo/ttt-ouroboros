# Paper reproduction map

This repository is scoped to the submitted paper **Self-Generated Feedback
Destabilizes Test-Time Training: A Causal Decomposition of Long-Horizon
Adaptation** ([arXiv:2610.05076](https://arxiv.org/abs/2610.05076)).

## Protocol families

Do not pool rows across these families.

| Suite | Statistical unit | Books/tasks | Seeds | Primary runner |
|---|---|---:|---:|---|
| Canonical TTT-E2E | PG-19 book | run rows 0–7; report books 2–7 | 42, 1, 7, 2, 3 | `scripts/horizon.py` |
| Exposure density | long PG-19 book | 8 fixed long books | 5 | `scripts/exposure_density_sweep.py` |
| Single update | PG-19 book/state pair | 8 books, positions 1/33/65/97 | paired | `scripts/preq_obs.py` |
| Gradient direction | PG-19 book/state pair | 8 books, positions 1/33/65/97 | 5 | `scripts/p1_direction_prediction.py` |
| Settlement | long PG-19 book | scale-specific long-book sets | paired | `scripts/deferred.py` |
| WebShop | held-out goal, paired within seed | 900 train / 150 test goals | 5 | `scripts/ws_arms.py` |

Canonical TTT-E2E uses 128 chunks of 1024 tokens, eight real prefill writes,
temperature 1, top-p .95, 16 branch-only clean probes and logical width 8. The
released manifests run eight physical rows and the analysis selects the six
screened books; this preserves width-dependent CUDA numerics.
The prefill-end baseline does not advance the real-text cursor: it and the first
scheduled probe score the same passage, matching the audited 25,601-token book
selection threshold.

## Main paper

| Paper item | Code or frozen input | Reproduction note |
|---|---|---|
| Table 1, scale comparison | `scripts/horizon.py` | Run the released 125M/760M/3B suites; all preserve width 8, the common books, seeds and probe schedule. |
| Figure 1a, canonical trajectory | `data/unified_perbook_data.json`, `figures/make_paper_figures.py` | Audited 6 books × 5 seeds × 16 probes; endpoint estimator matches Table 1. |
| Figure 1b, exposure boundary | `scripts/exposure_density_sweep.py`, `analysis/analyze_exposure_density.py`, `figures/make_paper_figures.py` | Separate eight-long-book suite. The panel is normalized to its own 0% result. |
| Table 2, Qwen real-text utility | `scripts/qwen_reviewer_p0c.py` | Updates the last four `down_proj` matrices with clipped Adam. |
| Figure 2, feedback-path intervention | `scripts/horizon.py`, `scripts/mechanism_pilot.py` | `open`/Fixed Generation uses a frozen generator and an adapting receiver. |
| Figure 3a, drift and endpoint damage | `scripts/horizon.py`, `figures/make_paper_figures.py` | Closed, Writes Off, Fixed Generation and Real-Text Learning. |
| Figure 3b, recorded replay | `scripts/horizon.py`, `scripts/mechanism_pilot.py` | Record once, then replay identical tokens read-only and read+write. |
| Tables 5–6, one-write transfer | `scripts/preq_obs.py` | Snapshot a common state and keep/discard one identical candidate update. |
| Gradient-direction diagnostic | `scripts/p1_direction_prediction.py`, `analysis/bootstrap_gradient_correlation.py` | Measures exact clean-gradient/candidate-update alignment and realized transfer damage; bootstrap books and seeds as clusters. |
| Figure 4a–b, heavy tail | `scripts/acceptance_heavy_tail.py`, `figures/make_paper_figures.py` | Passage-level fixed-receiver comparisons; source sequence is the independent source unit. |
| Figure 4c/Table 28, WebShop | `scripts/ws_arms.py`, `analysis/agent_stats.py`, `figures/make_agent_causal_success.py` | `none`, `uniform`, `fixed`, `settlement` map to the four paper arms. |
| Figure 5a, Settlement | `scripts/deferred.py`, `analysis/analyze_settlement_value.py`, `figures/make_provenance_noise.py` | Compare each scale only with its paired Writes Off trajectory. |
| Figure 5b, label corruption | `figures/make_provenance_noise.py` | Final audited aggregate is embedded in the figure source; the original corruption-grid launcher was not recovered in this source snapshot. |

## Scope

This compact release retains the main causal experiments and their runtime
dependencies. Auxiliary dose/anchor/aTTT grids, cluster templates and release
administration files are available in Git history at `d192991`.

The original Qwen long-horizon and source-label corruption launchers are not
included. Plotting their saved aggregates is not a from-scratch reproduction.

## Canonical commands

```bash
python scripts/run_config.py configs/canonical_closed.yaml
python scripts/run_config.py configs/canonical_writes_off.yaml
python scripts/run_config.py configs/canonical_fixed_generation.yaml
```

For 760M and 3B use `configs/suites/main_760m.yaml` and
`configs/suites/main_3b.yaml`. Never infer a preset from a filename.

## Exposure sweep

Run fractions 0, .05, .10, .20 and .31 with the same seed/book grid. At .31,
also run `bursty` while keeping the number of real slots fixed. Aggregate books
after first averaging seeds within each book.

## WebShop

The four paper policies are:

```bash
python scripts/ws_arms.py --policy none       --episodes 900 --stream-seed 0 --out results/webshop/off_s0.json
python scripts/ws_arms.py --policy uniform    --episodes 900 --stream-seed 0 --out results/webshop/closed_s0.json
python scripts/ws_arms.py --policy fixed      --episodes 900 --stream-seed 0 --out results/webshop/fixed_s0.json
python scripts/ws_arms.py --policy settlement --episodes 900 --stream-seed 0 --out results/webshop/settlement_s0.json
```

Repeat seeds 0–4. Settlement accumulates a temporary 25-episode candidate while
the committed state supplies actions, evaluates current and candidate states on
the same 10 validation goals, and commits only on higher mean reward.
Final evaluation uses the same 150 held-out goals with writes disabled.
Current runner outputs include ordered goal IDs and split hashes. Historical
formal JSONs lack explicit goal IDs; a reconstructed manifest alone does not
prove that validation and final evaluation had zero overlap. Audit the actual
goal sets before making that claim.

## 3B recorded replay

`scripts/reviewer_3b_replay.py` separates source and receiver books and writes
token hashes into every output. Run `--task record` once per source trajectory,
then run `--task replay --condition read` and `--condition write` against the
same record. Do not aggregate cells unless their source-record, checkpoint,
dataset and code hashes match.

The exact runner and three recorded source trajectories were recovered. The
original read/write result JSONs were not, so any new numbers from this runner
are rerun results.

## Figure regeneration

`figures/make_paper_figures.py` writes Figure 1 plus the causal, heavy-tail,
breadth and appendix repair panels. The other three scripts write the WebShop,
commitment and source-label-noise panels. Generated PDFs are ignored by Git.


# Installation

Use the Conda environment from the README, or install the package with pip.

## Conda

```bash
conda env create -f environment.yml
conda activate ttt-ouroboros
cp env.sh.example env.sh
# Edit the three TTT_* paths, then:
source env.sh
python scripts/selfcheck.py --profile language
```

## Pip

Use Python 3.11–3.12 and a CUDA-matched PyTorch 2.x build. The reported language
experiments used bf16 CUDA kernels; CPU execution is for unit tests and analysis.

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -e '.[language,dev]'
```

Copy `env.sh.example` and set `TTT_DATA`, `TTT_CKPT`, and
`TTT_OUT`. Validate a real checkpoint before launching a long run:

```bash
python scripts/selfcheck.py --profile language --full
python scripts/run_paper_suite.py --suite configs/suites/main_125m.yaml --smoke
```

For WebShop, install its upstream repository separately, set `WEBSHOP_DIR` and
`WEBSHOP_DATA`, then install this repository's agent extras:

```bash
pip install -e '.[agents]'
```

Run `python scripts/selfcheck.py --full` on a GPU host after exporting
`TTT_DATA`, `TTT_CKPT`, and `TTT_OUT`.



# Data and checkpoints

The repository does not redistribute corpora or weights. Expected layout:

```text
$TTT_DATA/
  pg19/val.npy
  records/self.pt
  records/degraded_replay.pt
$TTT_CKPT/
  125m-ext32k.pt
  760m-ext32k.pt
  3b_128k_pt.pt
```

`val.npy` is a one-dimensional integer token array. Book boundaries are encoded
by the protocol's BOS token and are selected by `find_books`; do not replace
book identities with row numbers alone. Record the byte range or stable book
ID, tokenization version, selection threshold, and SHA256 in every public
artifact manifest.

Checkpoints must be loaded with the matching preset in `ttt_pt/config.py`.
Never infer a preset from parameter count. The 3B 8K and 128K checkpoints are
distinct experimental objects.

Place an optional `<artifact>.sha256` sidecar beside large data and checkpoint
files when hashing them at every launch is too expensive. Its first whitespace
separated field must be the lowercase SHA256 digest. Otherwise the runners hash
the artifact directly and store the digest in `_config`.

The canonical 125M protocol loads physical books 0--7 at logical width eight
and reports books 2--7. This distinction is encoded in the manifests; a result
that ran only six physical rows is not canonical even if it labels them 2--7.

For Qwen3-4B experiments, retokenize immutable PG-19 streams and retain the
resulting `manifest.json`:

```bash
python scripts/qwen_prepare_pg19.py \
  --val "$TTT_DATA/pg19/val.npy" \
  --source-tokenizer "$TTT_SOURCE_TOKENIZER" \
  --qwen-tokenizer "$TTT_QWEN_MODEL" \
  --out "$TTT_DATA/qwen_pg19"
```

WebShop expects the upstream repository in `WEBSHOP_DIR` and the deterministic
catalogue made by `scripts/ws_build_catalogue.py` in `WEBSHOP_DATA`.


# Artifact and evidence inventory

This file states what the source tree can regenerate and what requires the
separate frozen artifact bundle. “Runner available” is not equivalent to “raw
result available.”

| Result family | Exact runner | Raw evidence required | Availability |
|---|---|---|---|
| Canonical 125M/760M/3B | `scripts/horizon.py` | checkpoint, PG-19 tokens, per-book JSON | runner and audited aggregate in tree; large inputs external |
| 125M recorded replay | `scripts/mechanism_pilot.py` | recorded `.pt`, read/write JSON | runner in tree; records external |
| 3B disjoint-source replay | `scripts/reviewer_3b_replay.py` | source `.pt`, read/write JSON | runner and three source records recovered; original result JSON missing |
| Gradient direction | `scripts/p1_direction_prediction.py` | 20 sample-level JSONs | runner recovered; raw JSONs in frozen audit bundle |
| WebShop causal/Settlement | `scripts/ws_arms.py` | five-seed JSONs, goal manifest | runner in tree; raw JSONs and deterministic goal manifest in frozen audit bundle |

## WebShop audit fields

Current runs write:

- goal ID for every training episode;
- ordered goal IDs for each evaluation and Settlement validation call;
- train/test split seed, counts and goal-set hash;
- per-goal reward and success;
- candidate/base validation scores and admission decisions;
- LoRA drift and number of written samples.

The paper's recovered formal bundle contains five 900-episode seeds for Closed
Loop, Fixed Generation and Settlement, plus the shared deterministic Writes Off
baseline. Each final evaluation contains 150 paired goal-level outcomes.

## Statistical units

- Canonical language experiments bootstrap books after averaging seeds within a
  book.
- Gradient direction resamples books and seeds as clusters; stream positions
  remain repeated measurements.
- WebShop resamples seeds and paired held-out goals. The same Writes Off vector
  is shared because it performs no stream updates and is deterministic under the
  fixed evaluation seed.

## Known missing evidence

The original `reviewer-p0b-replay-v1` 3B read/write JSONs were not present in
the recovered experiment and release snapshots.
Any newly generated files must be labelled as reruns and carry new code,
checkpoint, dataset and trajectory hashes.


## Experimental training harness

These runners study training cost and adaptation after training. They are
**not the experiments underlying the paper's existing tables**. All four arms
share the same prime architecture and outer AdamW optimizer:

| `--method` | Inner updates | Outer gradient |
|---|---|---|
| `adam` | None | Ordinary language-model gradient |
| `exact` | Original sequential K=1 | Second-order meta-gradient |
| `fo` | Fixed weights within K=2 blocks; average gradient then update | First order; full suffix KV graph |
| `shared` | Same K=2 forward update | Reused backward; truncated suffix KV graph, full prefix backward |

K=1 and K=2 differ in update frequency and dose. Their comparison does not
isolate the gradient approximation. None of these training arms includes a
Settlement gate. Use `python -m scripts.atomic_block_pilot --help` and
`python -m scripts.atomic_block_closed_pilot --help` for separate block-level
Settlement diagnostics.

### Train 125M from random initialization

Run from a repository checkout with `pip install -e '.[language,dev]'`.
The current GPU harness targets large-memory CUDA devices and waits for at
least **110 GiB free VRAM** before loading its model. This is a launcher guard,
not the measured peak allocation of every method. The documented configuration
has been smoke-tested on H200; smaller GPUs need a separately validated setup.

The reference is the [official TTT-E2E pretraining recipe](https://github.com/test-time-training/e2e/blob/a4fc4788ace38e29b5067916d4f4be33da894085/configs/training/125m/pretrain-8K.yaml):
8,192-token sequences, global batch 64, 4,800 optimizer steps, and
**2,516,582,400 tokens per arm**. Outer LR warms up from zero to 0.003 over
480 steps, then decays toward 1e-5. Inner LR warms up from 0.1 to 1 over
480 steps. Model seed is zero by default.

Prepare public DCLM with the pinned Llama-3 tokenizer:

```bash
python -m scripts.prepare_scratch_dclm \
  --out data/scratch_dclm \
  --workers 8 --tokenizer-threads 8 \
  --tokenizer-revision 315b20096dc791d381d514deb5f8bd9c8d6d3061
```

The preparer filters documents shorter than 8K tokens, removes exact text
duplicates, and holds out all documents whose content hash selects the validation
partition. It saves 64 validation documents, each evaluated over its first 8K
tokens. It pins the dataset revision in `plan.json` on first launch and records
source and token hashes. This is **re-tokenized public DCLM, not the official
pre-tokenized bucket**; source-shard ordering also differs from the official
Grain shuffle. The implementation is PyTorch rather than JAX. Near-duplicate
overlap is not ruled out by the exact-content hash split.

Once `ready.json` and `validation.npy` exist, training can consume the committed
prefix while preparation continues. Missing data causes a wait, never repeated
samples or zero padding. Run each command on its own GPU:

```bash
CUDA_VISIBLE_DEVICES=0 TTT_CKPT_PREFIX=1 python -m scripts.train_scratch_comparison \
  --method adam --data data/scratch_dclm --out results/scratch/adam --microbatch 2
CUDA_VISIBLE_DEVICES=1 TTT_CKPT_PREFIX=1 python -m scripts.train_scratch_comparison \
  --method exact --data data/scratch_dclm --out results/scratch/exact --microbatch 2
CUDA_VISIBLE_DEVICES=2 TTT_CKPT_PREFIX=1 python -m scripts.train_scratch_comparison \
  --method fo --data data/scratch_dclm --out results/scratch/fo --microbatch 2
CUDA_VISIBLE_DEVICES=3 TTT_CKPT_PREFIX=1 python -m scripts.train_scratch_comparison \
  --method shared --data data/scratch_dclm --out results/scratch/shared --microbatch 2
```

Microbatch two uses 32 accumulation steps to retain global batch 64. The
scratch entry point has **no pretrained-checkpoint option**. It hashes random
initialization before training; verify equal hashes across arms. For an
integration check, add `--smoke-steps 3` and use distinct `results/smoke/*`
output directories. A smoke run is not a completed pretraining run.

Each arm writes:

- `config.json`: initialization, code and data-plan hashes, schedules and method;
- `train.jsonl`: loss, gradient norm, LR, inner multiplier, step time and peak memory;
- `validation.jsonl`: fixed-document NLL under no writes, K=1 and K=2, initially
  and every 200 steps;
- `latest.pt`: atomic model/optimizer/RNG checkpoint every 100 steps;
- `model_*.pt`: model snapshots at steps 400, 1,200, 2,400 and 4,800;
- `complete.json`: written only when the run and its requested evaluations finish.

Resume with the same command and directory. The runner rejects changed config
or source hashes and refuses to overwrite completed runs. Logs may contain
replayed steps after recovery; retain the last record for each step when
aggregating. To collect optional PG-19 32K/128K transfer and short generation
diagnostics, add `--pg19-eval-data` pointing to the data prepared below. These
are not substitutes for Books context-extension training.

### Continue an existing checkpoint

This is a separate, short-budget experiment and must not be reported as
from-scratch pretraining:

```bash
python -m scripts.prepare_inner_comparison --out data/continuation
CUDA_VISIBLE_DEVICES=0 python -m scripts.inner_training_comparison \
  --method shared --preset 125m-e2e-ext32k \
  --ckpt "$TTT_CKPT/125m-ext32k.pt" --data data/continuation \
  --out results/continuation/shared --steps 120 --accum 4 --seq-length 16384
python -m scripts.summarize_inner_comparison --root results/continuation
```

Run `adam`, `exact`, and `fo` in matching sibling directories for a complete
comparison. The summary leaves missing arms pending. It reports common K=2
deployment alongside native-rule fields in JSON. Generated harm uses only a
16K diagnostic; it is not a canonical 128K generation result. This continuation
runner's Adam path uses the full suffix layout. Use
`python -m scripts.benchmark_adam_layout --help` to audit it against a chunked
layout that retains the full KV gradient. The scratch runner already uses that
optimized baseline. Gradient reductions in the layout audit use FP64.

# Release validation

Validation performed on 2026-09-29.

## Static and CPU checks

- all Python sources compiled;
- manifest and suite commands expanded without unresolved variables;
- four release-workflow tests passed;
- tensor-mechanics tests are executed by CI after CPU PyTorch is installed;
- Ruff passed on the package, tests and release utilities;
- the source distribution built as a wheel;
- all submitted vector figures regenerated.

## GPU integration smoke

The canonical three-arm smoke test ran with the released 125M checkpoint on an
NVIDIA H20 under PyTorch 2.9.0+cu128. It used physical book row 2, seed 42, 16
chunks and branch-only probes at positions 8 and 16.

| Arm | Status | Probe NLLs | Final drift |
|---|---|---|---:|
| Writes Off | passed | 3.199708, 3.297248 | 0.00455 |
| Closed Loop | passed | 3.199708, 3.377452 | 0.00609 |
| Fixed Generation | passed | 3.199708, 3.363771 | 0.00570 |

The identical first-probe value checks the common prefill state. These smoke
values only validate execution and must not be reported as experimental results.
