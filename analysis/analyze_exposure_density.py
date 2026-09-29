#!/usr/bin/env python3
import argparse,json
from pathlib import Path
import numpy as np
FRACS=(0.0,.05,.15,.31);SEEDS=(42,1,7,2,3)
def ci(x,rng):
 x=np.asarray(x,float);b=x[rng.integers(0,8,(20000,8))].mean(1);return float(x.mean()),*map(float,np.quantile(b,[.025,.975]))
def fmt(x):return f"{x[0]:.4f} [{x[1]:.4f}, {x[2]:.4f}]"
def load(root,f,arr):
 tag=str(f).replace(".","p");z=[]
 for seed in SEEDS:
  use="even" if f==0 else arr;d=json.loads((root/"raw"/f"f{tag}_{use}_s{seed}.json").read_text());assert d["status"]=="passed";z.append(d)
 return z
def main():
 p=argparse.ArgumentParser();p.add_argument("--root",type=Path,required=True);p.add_argument("--out",type=Path,required=True);a=p.parse_args();rng=np.random.default_rng(20260920);rows=[]
 for f in FRACS:
  for arr in ("even","bursty"):
   ds=load(a.root,f,arr);m=ds[0]["manifest"];assert m["book_indices"]==[0,1,27,28,29,43,44,46]
   final={q:np.mean([d["policies"][q]["probes"][-1]["nll_book"] for d in ds],0) for q in ("all","mask","no_writes")}
   h=final["all"]-final["mask"];b=final["no_writes"]-final["mask"]
   curves=[]
   for k in range(len(ds[0]["policies"]["all"]["probes"])):
    curves.append(np.mean([np.asarray(d["policies"]["all"]["probes"][k]["nll_book"])-np.asarray(d["policies"]["mask"]["probes"][k]["nll_book"]) for d in ds],0))
   onset=next((ds[0]["policies"]["all"]["probes"][k]["position"] for k,x in enumerate(curves) if np.mean(x)>=.1),None)
   sig=next((ds[0]["policies"]["all"]["probes"][k]["position"] for k,x in enumerate(curves) if ci(x,rng)[1]>0),None)
   tail={}
   for q in ("all","mask"):
    gs=[g for d in ds for g in d["policies"][q]["generation"][-5:]];tail[q]=(float(np.mean([x for g in gs for x in g["distinct2_book"]])),float(np.mean([x for g in gs for x in g["repeated4_book"]])))
   rows.append((f,arr,ci(h,rng),ci(b,rng),m["real_slot_count"],m["max_generated_burst"],tail,onset,sig))
 lines=["# Real-text exposure phase boundary","","125M; audited 8 long books ×5 seeds; 128 live chunks (8 real prefill +120 exposure); T=1; top-p=.95. Probes are branch-only and do not interrupt the live stream.","",
 "`H=All Writes-Mask` measures generated-write harm. `B=No Writes-Mask` measures benefit from real-text writes; positive B is beneficial. Fractions count real slots among the 120 exposure chunks.","",
 "| real fraction | arrangement | real slots | max generated burst | H [95% CI] | B [95% CI] | all d2/r4 | mask d2/r4 | onset H>=.1 | onset CI>0 |","|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
 for f,arr,h,b,n,burst,t,o,s in rows:lines.append(f"| {100*f:.0f}% | {arr} | {n} | {burst} | {fmt(h)} | {fmt(b)} | {t['all'][0]:.3f}/{t['all'][1]:.3f} | {t['mask'][0]:.3f}/{t['mask'][1]:.3f} | {o if o else '—'} | {s if s else '—'} |")
 lines += ["","The 0% schedules are identical by construction and reuse the same five files; the duplicate table row is included only to show both phase-boundary curves from a common origin."]
 a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text("\n".join(lines)+"\n")
 try:
  import matplotlib.pyplot as plt
  fig,ax=plt.subplots(figsize=(5.5,4))
  for arr,mark in (("even","o-"),("bursty","s--")):
   z=[r for r in rows if r[1]==arr];x=[100*r[0] for r in z];y=[r[2][0] for r in z];lo=[v-r[2][1] for v,r in zip(y,z)];hi=[r[2][2]-v for v,r in zip(y,z)];ax.errorbar(x,y,yerr=[lo,hi],fmt=mark,capsize=2,label=arr)
  ax.axhline(0,color=".6",lw=1);ax.set_xlabel("Real-text exposure slots (%)");ax.set_ylabel("All Writes - Mask NLL (nats)");ax.legend(frameon=False);fig.tight_layout();fig.savefig(a.out.with_suffix(".pdf"));plt.close(fig)
 except ImportError:pass
 print(a.out.read_text())
if __name__=="__main__":main()
