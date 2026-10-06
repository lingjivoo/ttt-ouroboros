# Paper reproduction map

This repository is scoped to the submitted paper **Self-Generated Feedback
Destabilizes Test-Time Training: A Causal Decomposition of Long-Horizon
Adaptation** (23-page version supplied on 2026-09-29).

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

## Appendix experiments

| Paper item | Runner |
|---|---|
| Decoder controls and reset | `scripts/mechanism_pilot.py`, `scripts/horizon.py` |
| Exposure-density table | `scripts/exposure_density_sweep.py` |
| Fixed recorded text / repetition targeting | `scripts/mechanism_e4.py` |
| aTTT and matched write-mass controls | `scripts/attt_closed_loop.py` |
| Update-strength frontier | `scripts/update_strength_sweep.py` |
| Anchor-content decomposition | `scripts/anchor_causal.py` |
| Settlement components, capacity and commitment | `scripts/deferred.py`, `scripts/settlement_mixed.py` |
| State restoration and stream equivalence | `scripts/state_probe_audit.py`, `scripts/stream_parity.py` |
| PyTorch/HF numerical checks | `validation/hf_ttt.py`, `validation/stream_parity.py`, `tests/test_remat.py` |

The paper's Qwen3-4B long-horizon self-writing launcher and the original
source-label corruption launcher were not present in the recovered experiment
snapshot. Their audited values and plotting code are preserved, but this release
does not claim that those two raw grids can be regenerated from scratch. This
is recorded explicitly so a similar script is not presented as the original.

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
the same 10 disjoint validation goals, and commits only on higher mean reward.
Final evaluation uses the same 150 held-out goals with writes disabled.
Current runner outputs include ordered goal IDs and split hashes. Historical
formal JSONs use the deterministic goal manifest distributed with the frozen
artifact bundle to establish the same pairing and zero overlap.

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
