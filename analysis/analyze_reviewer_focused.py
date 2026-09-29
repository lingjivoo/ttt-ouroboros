"""Frozen analyses for reviewer P2 and P0-B raw outputs."""
import argparse,glob,json
from pathlib import Path
import numpy as np
SEEDS=(42,1,7,2,3)
def ci8(x,rng):
 x=np.asarray(x,float);b=x[rng.integers(0,8,(20000,8))].mean(1);return x.mean(),*np.quantile(b,[.025,.975])
def fmt(x):return f'{x[0]:.4f} [{x[1]:.4f},{x[2]:.4f}]'
def p2(root):
 rng=np.random.default_rng(20260920);v={};metrics={}
 for tag,top in [('095',.95),('100',1.0)]:
  for mode in ('masked','closed'):
   nll=[];d2=[];r4=[];fe=[]
   for seed in SEEDS:
    d=json.loads((root/f'{mode}_p{tag}_s{seed}.json').read_text());assert d['status']=='passed';m=d['manifest']
    assert m['mode']==mode and m['seed']==seed and m['n_seqs']==8 and m['n_chunks']==128 and m['temperature']==1 and m['top_p']==top
    assert m['book_indices']==list(range(8)) and m['fixed_context_diagnostic'].startswith('empty caches')
    nll.append(d['probes'][-1]['nll_book']);rows=[r for r in d['rows'] if not r['injected_real']]
    d2.append(rows[-1]['distinct2_book']);r4.append(rows[-1]['repeated4_book'])
    fe.append(np.asarray(rows[-1]['fixed_context_entropy_book'])-rows[0]['fixed_context_entropy_book'])
   v[top,mode]=np.mean(nll,axis=0);metrics[top,mode]=(float(np.mean(d2)),float(np.mean(r4)),np.mean(fe,axis=0))
 lines=['# P2 sampling result','', '8 books ×5 seeds; seed means are formed within book before a 20,000-draw paired book bootstrap (analysis seed 20260920).','',
 '| top-p | mode | Final clean NLL [95% CI] | H=closed−masked [95% CI] | final distinct-2 | final repeated-4 | fixed-context entropy last−first [95% CI] |','|---:|---|---:|---:|---:|---:|---:|']
 for top in (.95,1.0):
  h=v[top,'closed']-v[top,'masked']
  for mode in ('masked','closed'):
   d2,r4,fe=metrics[top,mode];lines.append(f'| {top:.2f} | {mode} | {fmt(ci8(v[top,mode],rng))} | {fmt(ci8(h,rng)) if mode=="closed" else "—"} | {d2:.4f} | {r4:.4f} | {fmt(ci8(fe,rng))} |')
 contrast=(v[.95,'closed']-v[.95,'masked'])-(v[1.0,'closed']-v[1.0,'masked'])
 lines+=['',f'Primary paired contrast `H_p=.95 − H_p=1`: **{fmt(ci8(contrast,rng))}**.','',
 'Fixed-context entropy uses identical held-out token IDs and empty attention caches for every arm. It isolates weight-state effects but is still a finite-state diagnostic, not a test that an ideal expected gradient is exactly zero.']
 return '\n'.join(lines)+'\n'
def p0b(root):
 rows=[];maxinit=0
 for seed in SEEDS:
  for g in range(3):
   for mapping in (0,1):
    a={}
    for cond in ('read','write'):
     d=json.loads((root/'replay'/f'{cond}_s{seed}_g{g}_m{mapping}.json').read_text());assert d['status']=='passed';a[cond]=d
    assert a['read']['manifest']['received_token_sha256']==a['write']['manifest']['received_token_sha256']
    assert a['read']['manifest']['source_indices']==a['write']['manifest']['source_indices']
    assert a['read']['manifest']['receiver_indices']==a['write']['manifest']['receiver_indices']
    for b,(src,rec) in enumerate(zip(a['read']['manifest']['source_indices'],a['read']['manifest']['receiver_indices'])):
     ir=a['read']['first_nll_book'][b];iw=a['write']['first_nll_book'][b];maxinit=max(maxinit,abs(ir-iw))
     rows.append((src,rec,seed,a['write']['change_nll_book'][b]-a['read']['change_nll_book'][b]))
 assert len(rows)==60 and maxinit<=1e-5
 cell={(s,r):np.mean([x[3] for x in rows if x[0]==s and x[1]==r]) for s,r,_,_ in rows}
 sources=sorted(set(x[0] for x in rows));receivers=sorted(set(x[1] for x in rows));rng=np.random.default_rng(20260920)
 boots=[]
 while len(boots)<20000:
  ss=rng.choice(sources,len(sources));rr=rng.choice(receivers,len(receivers));sc={x:int(np.sum(ss==x)) for x in sources};rc={x:int(np.sum(rr==x)) for x in receivers}
  vals=[];weights=[]
  for k,v in cell.items():
   w=sc[k[0]]*rc[k[1]]
   if w:vals.append(v);weights.append(w)
  if weights:boots.append(np.average(vals,weights=weights))
 vals=np.array(list(cell.values()));main=(vals.mean(),*np.quantile(boots,[.025,.975]))
 bs=[]
 for _ in range(20000):
  ss=rng.choice(sources,len(sources));bs.append(np.mean([np.mean([v for (s,_),v in cell.items() if s==x]) for x in ss]))
 one=(vals.mean(),*np.quantile(bs,[.025,.975]));loo={s:np.mean([v for (q,_),v in cell.items() if q!=s]) for s in sources}
 lines=['# P0-B fixed-text 3B replay result','',
 'Six source books, six disjoint receiver books, two preregistered cyclic mappings and five independent source recordings per source. Each source recording is reused by two receivers; source/receiver pairs are the 12 graph edges, not 60 independent rows.','',
 f'Primary `J=(write_last-write_first)−(read_last-read_first)`: **{fmt(main)}** (two-way source/receiver cluster bootstrap, 20,000 draws).',
 f'Source-only cluster bootstrap sensitivity: **{fmt(one)}**. Maximum read/write initial NLL mismatch: `{maxinit:.3g}` nats.','',
 '| source book index | mean J | leave-one-source-out J |','|---:|---:|---:|']
 for s in sources:lines.append(f'| {s} | {np.mean([v for (q,_),v in cell.items() if q==s]):+.4f} | {loo[s]:+.4f} |')
 lines+=['','| receiver book index | mean J |','|---:|---:|']
 for r in receivers:lines.append(f'| {r} | {np.mean([v for (_,q),v in cell.items() if q==r]):+.4f} |')
 lines+=['','Every paired read/write job verifies identical received-token SHA256 values. With only six source and receiver clusters, intervals are unstable and source-level sensitivity is part of the result.']
 return '\n'.join(lines)+'\n'
def main():
 p=argparse.ArgumentParser();p.add_argument('--p2',type=Path);p.add_argument('--p0b',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 text=[]
 if a.p2:text.append(p2(a.p2))
 if a.p0b:text.append(p0b(a.p0b))
 a.out.write_text('\n'.join(text));print(a.out.read_text())
if __name__=='__main__':main()
