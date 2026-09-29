#!/usr/bin/env python3
"""Frozen paired-book analysis and Pareto figure for update-strength sweep."""
import argparse,json
from pathlib import Path
import numpy as np
ALPHAS=(0.0,0.0625,0.125,0.25,0.5,1.0,2.0);SEEDS=(42,1,7,2,3)
def ci(x,rng):
 x=np.asarray(x,float);b=x[rng.integers(0,8,(20000,8))].mean(1);return float(x.mean()),*map(float,np.quantile(b,[.025,.975]))
def fmt(x):return f"{x[0]:.4f} [{x[1]:.4f}, {x[2]:.4f}]"
def load(root,mode,alpha):
 tag=str(alpha).replace(".","p");ds=[]
 for seed in SEEDS:
  d=json.loads((root/"raw"/f"{mode}_a{tag}_s{seed}.json").read_text());assert d["status"]=="passed";m=d["manifest"]
  assert m["alpha"]==alpha and m["mode"]==mode and m["book_indices"]==[0,1,27,28,29,43,44,46] and m["n_chunks"]==128
  ds.append(d)
 return ds
def main():
 p=argparse.ArgumentParser();p.add_argument("--root",type=Path,required=True);p.add_argument("--out",type=Path,required=True);a=p.parse_args();rng=np.random.default_rng(20260920)
 data={(m,x):load(a.root,m,x) for m in ("closed","real") for x in ALPHAS};base_c=data["closed",0.0];base_r=data["real",0.0];rows=[]
 for alpha in ALPHAS:
  c=data["closed",alpha];r=data["real",alpha]
  cn=np.mean([d["probes"][-1]["nll_book"] for d in c],0);c0=np.mean([d["probes"][-1]["nll_book"] for d in base_c],0)
  rn=np.mean([d["probes"][-1]["nll_book"] for d in r],0);r0=np.mean([d["probes"][-1]["nll_book"] for d in base_r],0)
  h=cn-c0;b=r0-rn
  updates=[u for d in c for u in d["updates"] if u["kind"]=="generated_scaled"]
  unorm=float(np.mean([x for u in updates for x in u["realized_update_norm_book"]]))
  clipfreq=float(np.mean([x for u in updates for x in u["clipped_book"]]))
  tail=[g for d in c for g in d["generation"][-5:]]
  d2=float(np.mean([x for g in tail for x in g["distinct2_book"]])) if tail else float("nan")
  r4=float(np.mean([x for g in tail for x in g["repeated4_book"]])) if tail else float("nan")
  curves=[]
  for k in range(len(c[0]["probes"])):
   curves.append(np.mean([np.asarray(d["probes"][k]["nll_book"])-np.asarray(z["probes"][k]["nll_book"]) for d,z in zip(c,base_c)],0))
  threshold=next((c[0]["probes"][k]["schedule_index"]+1 for k,x in enumerate(curves) if np.mean(x)>=.1),None)
  significant=next((c[0]["probes"][k]["schedule_index"]+1 for k,x in enumerate(curves) if ci(x,rng)[1]>0),None)
  rows.append((alpha,ci(h,rng),ci(b,rng),unorm,clipfreq,d2,r4,threshold,significant))
 lines=["# Update-strength damage–adaptation frontier","","125M TTT-E2E; canonical 8 books × 5 seeds; 128 chunks; batch 8; T=1; top-p=.95. Alpha multiplies the already-clipped parameter delta.","",
 "`H(alpha)=closed(alpha)-closed(0)` and `B(alpha)=real(0)-real(alpha)`; positive H is harm and positive B is real-text adaptation benefit. Seeds are averaged within book before the paired 20,000-draw book bootstrap.","",
 "| alpha | H [95% CI] | B [95% CI] | realized generated update norm | clipping frequency | tail distinct-2 | tail repeated-4 | onset H>=.1 | onset CI>0 |","|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
 for x,h,b,n,cf,d2,r4,o,s in rows:lines.append(f"| {x:g} | {fmt(h)} | {fmt(b)} | {n:.6g} | {cf:.4f} | {d2:.4f} | {r4:.4f} | {o if o else '—'} | {s if s else '—'} |")
 a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text("\n".join(lines)+"\n")
 try:
  import matplotlib.pyplot as plt
  xs=[r[2][0] for r in rows];ys=[r[1][0] for r in rows];xlo=[x-r[2][1] for x,r in zip(xs,rows)];xhi=[r[2][2]-x for x,r in zip(xs,rows)];ylo=[y-r[1][1] for y,r in zip(ys,rows)];yhi=[r[1][2]-y for y,r in zip(ys,rows)]
  fig,ax=plt.subplots(figsize=(5.3,4));ax.errorbar(xs,ys,xerr=[xlo,xhi],yerr=[ylo,yhi],fmt="o-",capsize=2)
  for x,y,r in zip(xs,ys,rows):ax.annotate(f"a={r[0]:g}",(x,y),xytext=(4,4),textcoords="offset points",fontsize=8)
  ax.axhline(0,color=".6",lw=1);ax.axvline(0,color=".6",lw=1);ax.set_xlabel("Real-text adaptation benefit B (nats)");ax.set_ylabel("Self-generated harm H (nats)");fig.tight_layout();fig.savefig(a.out.with_suffix(".pdf"));plt.close(fig)
 except ImportError:pass
 print(a.out.read_text())
if __name__=="__main__":main()
