"""Aggregate the four paired continuation runs without dropping missing arms."""
import argparse
import json
import statistics
from pathlib import Path


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def mean(x):
    return statistics.mean(x)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);args=ap.parse_args()
    root=Path(args.root);results={};configs=[]
    for method in ('adam','exact','fo','shared'):
        base=root/method;cfg=read(base/'config.json')
        if cfg:configs.append(cfg)
        rows=[]
        if (base/'train.jsonl').exists():
            # Resumed steps may be replayed from an earlier valid checkpoint.
            rows=list({r['step']:r for line in (base/'train.jsonl').read_text().splitlines() if line.strip() for r in [json.loads(line)]}.values())
            rows.sort(key=lambda r:r['step'])
        result=dict(status=read(base/'status.json'),completed_steps=max([r['step'] for r in rows],default=0))
        if len(rows)>2:
            result.update(median_step_s=statistics.median(r['step_s'] for r in rows[2:]),
                          mean_step_s=mean([r['step_s'] for r in rows[2:]]),
                          peak_gib=max(r['peak_gib'] for r in rows),
                          training_step_seconds=sum(r['step_s'] for r in rows))
        initial,short,long,gen=[read(base/f) for f in ('initial_32k.json','final_32k.json','final_128k.json','generated.json')]
        if long and long['status']=='complete':
            result['long_real']=long
            result['nll_off']=mean(long['policies']['0']['last32k_nll_per_book'])
            result['nll_k2']=mean(long['policies']['2']['last32k_nll_per_book'])
            result['benefit_k2']=mean(long['policies']['2']['last32k_benefit_per_book'])
            native='1' if method=='exact' else ('0' if method=='adam' else '2')
            result['native_k']=int(native)
            result['native_nll']=mean(long['policies'][native]['last32k_nll_per_book'])
        if initial and short and initial['status']==short['status']=='complete':
            result['training_change_32k_k2']=mean(short['policies']['2']['mean_nll_per_book'])-mean(initial['policies']['2']['mean_nll_per_book'])
        if gen and gen['status']=='complete':
            result['generated']=gen
            result['generated_harm']=mean(gen['anchored_harm_per_book'])
        results[method]=result
    if configs:
        for key in ('checkpoint_sha256','training_window_order','tokens','seed','seq_length','accumulation','lr','optimizer','code_sha256'):
            assert all(c[key]==configs[0][key] for c in configs), 'unpaired config: '+key
        assert all(c['dataset_manifest']==configs[0]['dataset_manifest'] for c in configs)
    complete=all(r['status'] and r['status'].get('status')=='complete' for r in results.values())
    obj=dict(status='complete' if complete else 'pending',paired_configs_checked=len(configs),results=results)
    (root/'comparison.json').write_text(json.dumps(obj,indent=2)+'\n')
    def fmt(v):return 'pending' if v is None else f'{v:.4f}'
    preset=configs[0].get('preset','125m-e2e-ext32k') if configs else 'TTT'
    lines=[f'# {preset} training comparison', '',
           'Paired short-budget continuation from one checkpoint; one seed. This is not a from-scratch pretraining result.', '',
           '| Method | Steps | Median full step (s) | Peak GiB | NLL without writes | NLL with K=2 | K=2 benefit | Generated harm |',
           '|---|---:|---:|---:|---:|---:|---:|---:|']
    for m,r in results.items():
        lines.append('| '+m+' | '+str(r['completed_steps'])+' | '+' | '.join(fmt(r.get(k)) for k in ('median_step_s','peak_gib','nll_off','nll_k2','benefit_k2','generated_harm'))+' |')
    if configs:
        c=configs[0]
        lines += ['', '## Fixed protocol', '',
                  f"- {c['steps']} AdamW steps; sequence {c['seq_length']}; microbatch 1; accumulation {c['accumulation']}; {c['tokens']:,} training tokens; seed {c['seed']}.",
                  '- Same prime architecture and parameters for all arms. Adam means no inner updates during training; all four outer optimizers are AdamW.',
                  '- Exact: original K=1 second-order. FO: K=2 full-prefix/full-suffix-graph first-order. Shared: K=2, suffix KV graph truncated at block boundaries; prefix backward retained through activation bridge.',
                  f"- Prefix checkpointing: {c.get('prefix_checkpointing','0')}; real-text evaluation batch: {c.get('eval_batch_size',4)}. Exact-rematerialization flags per arm: { {r['method']:r.get('exact_rematerialization',False) for r in configs} }.",
                  '- Inner LR 1.0, per-row clip 1.0; K=2 averages two chunk gradients then clips. K=1 vs K=2 is not a dose-matched ablation.',
                  '- Outer peak LR 4e-4, 12-step warmup, cosine to 1e-5, betas (0.9,0.95), weight decay 0.1, gradient clip 1.0. Fresh Adam state for every arm.',
                  '- Official PG-19 training books are disjoint by ID from four official test books. Starting-checkpoint pretraining overlap is not newly audited.',
                  '- Reported real-text NLL is the last 32 chunks of a 128-chunk teacher-forced stream. Benefit is per-book NLL(no writes) minus NLL(K=2). Both K=1 and K=2 raw results are retained.',
                  '- Generated harm: common K=2 deployment, first two test books, 4K shared real prefill, 16K self-generation, T=1/top-p=.95; independent probe at book offset 64K. Anchored Closed-minus-Off difference, paired seeds. A short diagnostic, not canonical 128K generation.',
                  '- Timings include transfer, zero_grad, forward, inner updates, backward, gradient clipping and AdamW; exclude checkpoint I/O and evaluation. First two steps excluded from speed median. Exact and FO use their supported attention backends.',
                  '- No Settlement gate in these four training arms. Values are descriptive; four books and one seed do not establish statistical equivalence.',
                  '', '## Provenance', '', f"Checkpoint SHA-256: `{c['checkpoint_sha256']}`", '',
                  'Training windows, test book IDs, tokenizer revision, source hashes and code hashes are in each `config.json`. Full per-book curves are in `comparison.json`.']
    (root/'COMPARISON.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(status=obj['status'],steps={m:r['completed_steps'] for m,r in results.items()})))


if __name__=='__main__':main()
