# Experiment protocols

## Canonical language protocol

- Model: 125M TTT-E2E, 32K-extended checkpoint.
- Independent units: eight audited PG-19 books.
- Sampling seeds: 42, 1, 7, 2, 3.
- Stream: eight real prefill writes followed by generated chunks to 128K.
- Decoder: temperature 1, top-p 0.95 unless the decoder is the intervention.
- Probes: branch-only clean text; probe tokens do not enter cache or weights.
- Primary contrast: anchored first-to-last change in Closed minus Masked.

## Exposure density

Use the same model, book identities, horizon, decoder, and probe definition.
Vary the fraction of real-text write slots and the arrangement (`even` or
`bursty`). The manifest `configs/exposure_density.yaml` shows one cell; a full
sweep changes only fraction, arrangement, and seed.

## Update strength

Scale the already clipped parameter delta. Scaling gradients before Adam is not
a valid dose intervention because Adam largely cancels uniform gradient scale.
Report generated-stream harm, identical-real-stream benefit, realized update
norm, clipping frequency, diversity, repetition, and onset.

## Settlement

A proposal is held outside the committed generation state. The next independent
validation evidence compares the current and candidate states; only accepted
proposals become persistent. Validation text and final evaluation text must be
disjoint. Report admission counts and pending proposals in addition to endpoint
loss.

## ALFWorld online/prequential protocol

The adapter remains active in both seen and unseen evaluation. Each task is
observed, acted on, optionally written, and then the stream advances. The hard
pilot excludes task type 1 and uses task types 2–6 with a 50-step limit. Pair
policies on the same task order and seeds. Report exact task-level wins and
losses; one stream is not five independent seeds.
