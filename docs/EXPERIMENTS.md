# Experiment protocols

## Canonical language protocol

- TTT-E2E 125M, 760M or 3B with the matching checkpoint preset.
- Run physical rows/books 0–7 to preserve logical width 8, then report the
  screened canonical books 2–7. Books, not seeds, are independent units.
- Seeds 42, 1, 7, 2 and 3; 128 chunks; 16 branch-only probes including the
  prefill-end reference.
- Eight real prefill writes, then generated chunks.
- The prefill-end reference and first scheduled trajectory probe reuse the same
  read-only clean passage; this preserves the audited 25,601-token selection
  threshold and isolates state change from probe-text change.
- Temperature 1 and top-p .95 unless the decoder is the intervention.
- Branch-only probes snapshot and restore the complete carried state.
- `closed` retains generated writes; `masked`/Writes Off drops them; `open`
  uses frozen-W0 generation while the receiver continues to update.

## Exposure density

This is a separate eight-long-book suite. Vary real-text write fraction over
0%, 5%, 10%, 20% and 31%; at 31%, compare even and burst schedules with the
same real-token budget. Report within-suite anchored harm and do not compare
its absolute 0% value with the canonical six-book endpoint.

## Update strength and content controls

`update_strength_sweep.py` scales the already clipped parameter delta. The
aTTT, random-dose and uniform-dose arms match write mass per row/chunk.
Anchor-content arms match slot count and distinguish skipped slots, read-only
self text, random tokens, shuffled real text, frozen-W0 text and real text.

## Settlement

A proposal remains outside the committed generation state. The next independent
validation evidence compares current and candidate states; only an improvement
commits. Validation and final evaluation text are disjoint. Report accepted,
rejected and pending proposals. Settlement uses a scale-specific long-book suite
and its endpoint gap must not be described as canonical H.

## WebShop

Qwen3-4B is frozen except rank-16 LoRA on `down_proj` in the final quarter of
layers. A written episode takes two Adam steps at 1e-5 over at most 16
action pairs. Each seed processes 900 training episodes and is evaluated on the
same 150 held-out goals. Settlement checks one 25-episode candidate on 10
disjoint validation goals. Run five paired stream seeds.
