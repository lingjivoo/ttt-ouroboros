#!/usr/bin/env python3
"""Real-text exposure-density and burst-length boundary for 125M TTT-E2E."""
from __future__ import annotations
import argparse,hashlib,json,os,sys,time,traceback
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.horizon import CS,WARMUP,distinct2,drift,find_books
from ttt_pt.config import PRESETS
from ttt_pt.model import TTTModel
from ttt_pt.stream import StreamState

BOOK_INDICES=(0,1,27,28,29,43,44,46)
POLICIES=("all","mask","no_writes")
FRACTION_COUNTS={0.0:0,0.05:6,0.10:12,0.15:18,0.20:24,0.31:37}
FRACTION_COUNTS_105={0.0:0,0.05:5,0.10:11,0.20:21,0.31:33}

def sha(path):
 h=hashlib.sha256()
 with open(path,"rb") as f:
  for b in iter(lambda:f.read(8<<20),b""):h.update(b)
 return h.hexdigest()
def save(path,value):
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+".tmp");t.write_text(json.dumps(value,indent=1)+"\n");os.replace(t,p)
def repeated4(ids):
 x=ids.tolist();seen=set();n=0
 for i in range(max(0,len(x)-3)):
  g=tuple(x[i:i+4]);n+=g in seen;seen.add(g)
 return n/max(1,len(x)-3)
def split_even(total,parts):
 q,r=divmod(total,parts);return [q+(i<r) for i in range(parts)]
def schedule(n,fraction,arrangement):
 counts=FRACTION_COUNTS_105 if n==105 else FRACTION_COUNTS
 key=min(counts,key=lambda x:abs(x-fraction))
 if abs(key-fraction)>1e-8:raise ValueError("unregistered exposure fraction")
 if n not in (105,120):raise ValueError("frozen schedule requires 105 or 120 live exposures")
 nr=counts[key];ng=n-nr
 if nr==0:return ["generated"]*n
 if arrangement=="even":
  gaps=split_even(ng,nr+1);seq=[]
  for i,g in enumerate(gaps):
   seq += ["generated"]*g
   if i<nr:seq.append("real")
 else:
  sizes=[8]*(nr//8)+([nr%8] if nr%8 else []);gaps=split_even(ng,len(sizes)+1);seq=[]
  for i,g in enumerate(gaps):
   seq += ["generated"]*g
   if i<len(sizes):seq += ["real"]*sizes[i]
 assert len(seq)==n and seq.count("real")==nr
 return seq
def max_generated_burst(seq):
 best=cur=0
 for x in seq:
  cur=cur+1 if x=="generated" else 0;best=max(best,cur)
 return best
def probe(st,x,y,cfg):
 s=st.snapshot();v=st.process_real_chunk(x,y,0.0,1.0,cfg).float().cpu().tolist();st.restore(s);return v

def run_policy(model,cfg,real,seq,policy,seed,args):
 st=StreamState(model,args.n_seqs,"cuda");real_pos=0;last=None;probes=[];generation=[]
 for _ in range(WARMUP):
  x=real[:,real_pos:real_pos+CS].cuda();y=real[:,real_pos+1:real_pos+CS+1].cuda();real_pos+=CS
  st.process_real_chunk(x,y,1.0,1.0,cfg);last=y[:,-1:]
 # Independent baseline after the common prefill.
 x=real[:,real_pos:real_pos+CS].cuda();y=real[:,real_pos+1:real_pos+CS+1].cuda();real_pos+=CS
 probes.append({"position":WARMUP,"exposure_index":0,"nll_book":probe(st,x,y,cfg),"drift":drift(st)})
 if args.canonical_105:
  live_i=0
  for scheduled_pos in range(1,121):
   if scheduled_pos%8==0:
    x=real[:,real_pos:real_pos+CS].cuda();y=real[:,real_pos+1:real_pos+CS+1].cuda();real_pos+=CS
    probes.append({"position":WARMUP+scheduled_pos,"scheduled_post_prefix_position":scheduled_pos,
                   "exposure_index":live_i,"nll_book":probe(st,x,y,cfg),"drift":drift(st)})
    continue
   kind=seq[live_i];live_i+=1
   if kind=="real":
    x=real[:,real_pos:real_pos+CS].cuda();y=real[:,real_pos+1:real_pos+CS+1].cuda();real_pos+=CS
    w=0.0 if policy=="no_writes" else 1.0;st.process_real_chunk(x,y,w,1.0,cfg);last=y[:,-1:]
   else:
    w=1.0 if policy=="all" else 0.0
    gen=st.generate_chunk(last,w,1.0,cfg,temperature=args.temperature,top_p=args.top_p,
      seed=seed*100000+scheduled_pos,sampling_device="cuda" if args.gpu_sampling else "cpu")
    last=gen[:,-1:]
    generation.append({"exposure_index":live_i,"scheduled_post_prefix_position":scheduled_pos,
                       "distinct2_book":[distinct2(z.cpu()) for z in gen],
                       "repeated4_book":[repeated4(z.cpu()) for z in gen]})
  assert live_i==105
  return {"probes":probes,"generation":generation,"final_drift":drift(st),"real_chunks_consumed":real_pos//CS}
 for i,kind in enumerate(seq,1):
  if kind=="real":
   x=real[:,real_pos:real_pos+CS].cuda();y=real[:,real_pos+1:real_pos+CS+1].cuda();real_pos+=CS
   w=0.0 if policy=="no_writes" else 1.0;st.process_real_chunk(x,y,w,1.0,cfg);last=y[:,-1:]
  else:
   w=1.0 if policy=="all" else 0.0
   gen=st.generate_chunk(last,w,1.0,cfg,temperature=args.temperature,top_p=args.top_p,
     seed=seed*100000+i,sampling_device="cuda" if args.gpu_sampling else "cpu")
   last=gen[:,-1:]
   generation.append({"exposure_index":i,"distinct2_book":[distinct2(z.cpu()) for z in gen],
                      "repeated4_book":[repeated4(z.cpu()) for z in gen]})
  if i%8==0:
   x=real[:,real_pos:real_pos+CS].cuda();y=real[:,real_pos+1:real_pos+CS+1].cuda();real_pos+=CS
   probes.append({"position":WARMUP+i,"exposure_index":i,"nll_book":probe(st,x,y,cfg),"drift":drift(st)})
 return {"probes":probes,"generation":generation,"final_drift":drift(st),"real_chunks_consumed":real_pos//CS}

def main():
 p=argparse.ArgumentParser();p.add_argument("--ckpt",required=True);p.add_argument("--val",required=True);p.add_argument("--out",required=True)
 p.add_argument("--real-fraction",type=float,required=True);p.add_argument("--arrangement",choices=("even","bursty"),required=True);p.add_argument("--seed",type=int,required=True)
 p.add_argument("--n-exposure",type=int,default=120);p.add_argument("--n-seqs",type=int,default=8);p.add_argument("--temperature",type=float,default=1.0);p.add_argument("--top-p",type=float,default=.95);p.add_argument("--gpu-sampling",action="store_true")
 p.add_argument("--canonical-105",action="store_true",help="105 live exposure slots plus 15 branch-only probes in the 128-position schedule")
 a=p.parse_args();start=time.monotonic();result={"status":"running","policies":{}}
 try:
  cfg=PRESETS["125m-e2e-ext32k"]();model=TTTModel(cfg.model,max_seq_len=(WARMUP+a.n_exposure+3)*CS).cuda().eval()
  raw=torch.load(a.ckpt,map_location="cpu",weights_only=False);model.load_state_dict(raw.get("model",raw),strict=True);del raw
  live_exposure=105 if a.canonical_105 else a.n_exposure
  seq=schedule(live_exposure,a.real_fraction,a.arrangement);arr=np.load(a.val,mmap_mode="r")
  # 8 prefill + real slots + one baseline + 15 probes + a small guard.
  n_probes=15 if a.canonical_105 else a.n_exposure//8
  need=(WARMUP+seq.count("real")+1+n_probes+2)*CS+1
  headline=find_books(arr,(WARMUP+(128-WARMUP)//8+2)*CS+1);books=[headline[i] for i in BOOK_INDICES]
  if any(e-s<need for s,e in books):raise ValueError("audited books lack required tail")
  real=torch.from_numpy(np.stack([np.asarray(arr[s:s+need]).astype(np.int64) for s,_ in books]))
  manifest={"protocol":"real-exposure-phase-boundary-v1","seed":a.seed,"real_fraction":a.real_fraction,
   "real_slot_count":seq.count("real"),"generated_slot_count":seq.count("generated"),"arrangement":a.arrangement,
   "max_generated_burst":max_generated_burst(seq),"schedule":seq,"n_exposure":live_exposure,
   "canonical_105":bool(a.canonical_105),"scheduled_post_prefix_positions":120 if a.canonical_105 else a.n_exposure,
   "branch_probe_positions":list(range(8,121,8)) if a.canonical_105 else list(range(8,a.n_exposure+1,8)),
   "total_live_chunks":WARMUP+live_exposure,
   "n_seqs":a.n_seqs,"book_indices":list(BOOK_INDICES),"book_bounds":books,"temperature":a.temperature,"top_p":a.top_p,
   "probe_rule":"branch-only before exposure and after each 8 exposure chunks; probes do not enter cache or weights",
   "policy_rule":{"all":"write real and generated exposure chunks","mask":"write real, read generated","no_writes":"read real and generated"},
   "prefill_rule":"all policies share eight unit-strength real writes","adaptation_benefit":"NLL(no_writes)-NLL(mask)",
   "damage":"NLL(all)-NLL(mask)","checkpoint_sha256":sha(a.ckpt),"val_sha256":sha(a.val),"code_sha256":sha(__file__),"audited_config":asdict(cfg)}
  torch.cuda.reset_peak_memory_stats()
  for policy in POLICIES:
   result["policies"][policy]=run_policy(model,cfg,real,seq,policy,a.seed,a);save(a.out,dict(result,manifest=manifest));print("finished",policy,flush=True)
  result.update(status="passed",manifest=manifest,seconds=time.monotonic()-start,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30);save(a.out,result);print(json.dumps({"status":"passed","seconds":result["seconds"]}))
 except Exception:
  result.update(status="failed",error=traceback.format_exc(),seconds=time.monotonic()-start);save(a.out,result);raise
if __name__=="__main__":main()
