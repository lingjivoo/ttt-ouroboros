#!/usr/bin/env python3
"""Frozen, paired book-clustered summary of the 90 reviewer follow-up shards."""
import argparse
import json
from pathlib import Path
import numpy as np

SEEDS = (42, 1, 7, 2, 3)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('root', type=Path)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    rng = np.random.default_rng(20260912)
    def stat(x):
        x = np.asarray(x, dtype=float)
        assert x.shape == (8,) and np.isfinite(x).all()
        b = x[rng.integers(0, 8, (20000, 8))].mean(1)
        lo, hi = np.quantile(b, [.025, .975])
        return f'{x.mean():.4f} [{lo:.4f}, {hi:.4f}]'
    lines = ['## Completed reviewer follow-ups — 2026-09-13', '',
             'All 90 scheduled shards completed: entropy/recovery 25/25, degeneration replay 5/5, and 3B 60/60. No horizon experiment process remains.', '',
             'Intervals below are 95% percentile book-cluster bootstrap intervals (20,000 draws, RNG seed 20260912). Average the five seeds within each book first; paired contrasts use the same books and seeds. Eight books are the resampling units.', '',
             '### 3B extension: eight books, five seeds, batch 2, 128 chunks', '',
             '| Condition | Final clean NLL [95% CI] | Excess NLL over masked [95% CI] |',
             '|---|---:|---:|']
    vals = {}
    for mode in ('masked', 'open', 'closed'):
        rows = []
        for seed in SEEDS:
            row = []
            for offset in (2, 4, 6, 8):
                d = json.loads((a.root/'3b'/f'{mode}_s{seed}_o{offset}.json').read_text())
                c = d['_config']
                assert c['mode'] == mode and c['book_offset'] == offset
                assert c['sampling_device'] == 'cuda' and c['n_chunks'] == 128 and c['n_seqs'] == 2
                k = next(k for k in d if not k.startswith('_'))
                assert k.endswith(f'_s{seed}')
                probes = d[k]['probes_book']
                assert len(probes) == 15 and probes[-1][0] == 127
                assert len(probes[-1][1]) == 2
                row.extend(probes[-1][1])
            rows.append(row)
        vals[mode] = np.array(rows).mean(0)
    for mode in vals:
        lines.append(f'| {mode} | {stat(vals[mode])} | {stat(vals[mode]-vals["masked"])} |')
    lines += ['', '`open` uses fixed-W0 generation; report the paired final clean-probe excess as H. These final endpoints do not estimate initial-to-final D. Raw shards: `out/reviewer_followups_20260912/3b` in the frozen artifact bundle. This extends scale evidence within TTT-E2E; it does not establish another native architecture.', '',
              '### Expanded 125M entropy/recovery', '',
              '| Condition | Final clean NLL [95% CI] | Excess over masked [95% CI] | Final distinct-2 | Final repeated-4 |',
              '|---|---:|---:|---:|---:|']
    ev = {}; metrics = {}; transitions = {}
    for mode in ('masked', 'fixed_w0', 'closed', 'kl_gate', 'closed_inject_g80'):
        nlls = []; d2 = []; r4 = []; con = []; rec = []
        for seed in SEEDS:
            d = json.loads((a.root/'entropy'/f'{mode}_s{seed}.json').read_text())
            assert d['status'] == 'passed'
            m = d['manifest']; assert m['seed'] == seed and m['n_seqs'] == 8
            assert m['sampling_device'] == 'cuda' and m['book_indices'] == list(range(8))
            assert m['inject_generated_index'] == (80 if mode == 'closed_inject_g80' else 0)
            assert d['probes'][-1]['schedule_index'] == 127
            nlls.append(d['probes'][-1]['nll_book'])
            generated = [r for r in d['rows'] if not r['injected_real']]
            d2.append(generated[-1]['distinct2_book']); r4.append(generated[-1]['repeated4_book'])
            tail = [r for r in generated if r['generated_index'] >= 80]
            con.append(np.mean([r['contraction_book'] for r in tail], axis=0))
            recovery = [r['recovery_book'] for r in tail if r.get('recovery_book') is not None]
            rec.append(np.mean(recovery, axis=0))
        ev[mode] = np.array(nlls).mean(0)
        metrics[mode] = (np.mean(d2), np.mean(r4))
        transitions[mode] = (np.array(con).mean(0), np.array(rec).mean(0))
    for mode in ev:
        d2, r4 = metrics[mode]
        lines.append(f'| {mode} | {stat(ev[mode])} | {stat(ev[mode]-ev["masked"])} | {d2:.4f} | {r4:.4f} |')
    lines += ['', '| Paired contrast | Final clean NLL difference [95% CI] |', '|---|---:|---:|']
    for mode in ('fixed_w0', 'kl_gate', 'closed_inject_g80'):
        lines.append(f'| {mode} minus closed | {stat(ev[mode]-ev["closed"])} |')
    lines += ['', '| Condition | Late mean entropy contraction [95% CI] | Late mean entropy recovery [95% CI] |', '|---|---:|---:|---:|']
    for mode, (con, rec) in transitions.items():
        lines.append(f'| {mode} | {stat(con)} | {stat(rec)} |')
    lines += ['', 'Late transition summaries average generated-index ≥80 within each seed and book, then average seeds. Contraction = entropy before minus after a generated chunk; recovery = next entropy before minus current entropy after. They are descriptive within each condition, whose generated text can differ, and do not isolate the causal effect of one write. Final diversity metrics are point estimates for the last generated chunk. Raw trajectories: `out/reviewer_followups_20260912/entropy` in the frozen artifact bundle.', '',
              '### Remaining scope', '',
              'The 90-shard scheduled follow-up suite is complete. The broader proposal still lacks a verified 760M checkpoint, another native TTT architecture, middle-stage replay, and cross-corpus clean probes; these were not part of this completed queue.', '',
              'Reproduce: `python3 analysis/analyze_reviewer_followups.py out/reviewer_followups_20260912 --out out/reviewer_followups_20260912/ANALYSIS.md`.']
    a.out.write_text('\n'.join(lines)+'\n')
    print(a.out.read_text())

if __name__ == '__main__':
    main()
