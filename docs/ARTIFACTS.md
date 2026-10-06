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
the recovered H200, t2, release, unified-audit or local experiment snapshots.
Any newly generated files must be labelled as reruns and carry new code,
checkpoint, dataset and trajectory hashes.
