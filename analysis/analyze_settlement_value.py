"""Descriptive paired summary; the two-book pilot is not population evidence."""
import argparse
import json
from pathlib import Path
import numpy as np

def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--trial-only',action='store_true')
    a=p.parse_args();seeds=(42,1,7)
    trials=[json.loads((a.root/f'trial_s{s}.json').read_text()) for s in seeds]
    retention=[] if a.trial_only else [json.loads((a.root/f'retention_s{s}.json').read_text()) for s in seeds]
    assert all(d['status']=='passed' for d in trials+retention)
    assert all(d['config']['books']==2 for d in trials+retention)
    assert all(d['config']['book_bounds']==trials[0]['config']['book_bounds'] for d in trials)
    text=['# Settlement value pilot: completed descriptive results','',
          f'Transaction self-write burst length: {trials[0]["config"].get("burst_length",3)}.', '',
          '125M, 2 receiver books, 3 fixed-source seeds (42, 1, 7). Four transactions per policy; results average evaluation chunks within transaction, transactions within seed/book, then seeds and books. No confidence interval or generality claim is warranted from this two-book pilot.', '',
          '## Policy comparison','', '| Policy | Future clean NLL | Paired difference vs real-only mask |', '|---|---:|---:|']
    nll={}
    for policy in trials[0]['policies']:
        nll[policy]=np.array([np.mean([e['nll_book'] for e in d['policies'][policy]['evaluations']],axis=0) for d in trials])
    for policy,x in nll.items():
        text.append(f'| {policy} | {x.mean():.6f} | {(x-nll["mask_q1"]).mean():+.6f} |')
    text+=['','| Policy | Selected states (12 transaction decisions) |','|---|---|']
    for policy in nll:
        choices={}
        for d in trials:
            for e in d['policies'][policy]['selections']:
                choices[e['chosen']]=choices.get(e['chosen'],0)+1
        text.append(f'| {policy} | {choices} |')
    events=[e for d in trials for e in d['utility_and_rejection_diagnostics']]
    accepted=[e for e in events if e['q1_accepted']]
    text+=['','## Candidate utility persistence','',
           f'q1 accepts {len(accepted)}/{len(events)} candidates on the binary-q1 trajectories. These are correlated events on three state streams, not independent trials.', '',
           '| Horizon | Mean NLL advantage: keep minus full | Mean among q1-accepted candidates |','|---|---:|---:|']
    for h in ('1','2','4','8'):
        full=np.mean([e['utility_by_horizon'][h] for e in events])
        acc=f'{np.mean([e["utility_by_horizon"][h] for e in accepted]):+.6f}' if accepted else 'not estimable'
        text.append(f'| {h} | {full:+.6f} | {acc} |')
    if accepted:
        for h in ('2','4','8'):
            rev=sum(np.mean(e['utility_by_horizon'][h])<0 for e in accepted)
            text.append(f'At horizon {h}, {rev}/{len(accepted)} q1-accepted candidates have negative batch-mean utility.')
    else:
        text+=['','No candidate is accepted, so this pilot cannot estimate reversal after acceptance. Add early/mixed candidate streams before drawing a conclusion about accepted-update lifetime.']
    rejected=[e['rejected_components'] for e in events if e['rejected_components'] is not None]
    text+=['','## Rejected transaction components','',
           f'{len(rejected)} rejected transactions. Means below use only the eight held-out evaluation chunks, excluding both validation chunks. Shared attention follows the keep baseline.', '',
           '| Component | Future clean NLL | NLL difference vs keep |','|---|---:|---:|']
    if rejected:
        parts={k:np.array([np.mean(e[k][2:],axis=0) for e in rejected]) for k in rejected[0]}
        for k,v in parts.items():text.append(f'| {k} | {v.mean():.6f} | {(v-parts["keep"]).mean():+.6f} |')
    else:text.append('| No rejected transactions | not estimable | not estimable |')
    if a.trial_only:
        text += ['', f'Raw trial outputs: `{a.root}` in the frozen artifact bundle. Reproduce with `python3 analysis/analyze_settlement_value.py {a.root} --trial-only --out {a.out}`. No retention runs belong to this burst-16 condition.']
        a.out.write_text('\n'.join(text)+'\n');print(a.out.read_text());return
    text+=['','## Retention attribution','',
           'Each entry averages eight held-out real evaluation chunks, three seeds, and two receiver books. k denotes scheduled self writes, not certified harmful writes. Delta columns compare lambda=0.95 minus lambda=1 at the same k. The last column subtracts the k=0 decay effect to control removal of prefill adaptation.', '',
           '| Mode | k | NLL lambda=1 | NLL lambda=0.95 | Paired decay difference | Difference minus k=0 decay difference |',
           '|---|---:|---:|---:|---:|---:|']
    for mode in (() if a.trial_only else ('replay','fixed_attention','fixed_delta','online')):
        def get(k,lam):return np.array([d['retention'][f'{mode}_k{k}_lam{lam}']['nll_book'] for d in retention])
        baseline=get(0,0.95)-get(0,1.0)
        for k in (0,1,8):
            x,y=get(k,1.0),get(k,0.95);diff=y-x
            text.append(f'| {mode} | {k} | {x.mean():.6f} | {y.mean():.6f} | {diff.mean():+.6f} | {(diff-baseline).mean():+.6f} |')
    text+=['','Fixed replay preserves token input; fixed attention additionally shares reference caches. Fixed delta additionally freezes baseline-computed proposals. Online permits policy-dependent generation. Compare paired differences across these controls; drift magnitude alone does not identify predictive damage.', '',
           'Raw outputs and SHA256 signatures: `out/settlement_value_20260913` in the frozen artifact bundle. Reproduce with `python3 analysis/analyze_settlement_value.py out/settlement_value_20260913 --out out/settlement_value_20260913/ANALYSIS.md`. Parameters and input stages were fixed before final evaluation; do not retrospectively select lambda, margin, or validation length by these results.']
    a.out.write_text('\n'.join(text)+'\n');print(a.out.read_text())

if __name__=='__main__':main()
