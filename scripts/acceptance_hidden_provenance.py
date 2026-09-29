"""P0 Stage-A hidden-provenance source construction and fixed-input policies."""
import argparse,hashlib,json,os,time,traceback
from pathlib import Path
import numpy as np
import torch
from horizon import CS,WARMUP,find_books,drift
from ttt_pt.config import PRESETS
from ttt_pt.model import TTTModel
from ttt_pt.stream import StreamState

LABELS=("trusted_real","clean_external","external_model","degraded_external","own_output")
# Powered reviewer protocol: 13/128 = 10.16% real slots.  ``clean_external``
# is intentionally absent: the deployable policy cannot distinguish the two
# model-generated external sources from human text.
COUNTS=(13,0,39,38,38)
def sha(p):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for b in iter(lambda:f.read(8<<20),b""):h.update(b)
 return h.hexdigest()
def save_json(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(".tmp");t.write_text(json.dumps(x,indent=1)+"\n");os.replace(t,p)
def load(a):
 cfg=PRESETS["125m-e2e-ext32k"]();m=TTTModel(cfg.model,max_seq_len=140*CS).cuda().eval()
 raw=torch.load(a.ckpt,map_location="cpu",weights_only=False);m.load_state_dict(raw.get("model",raw),strict=True);return cfg,m
def tokhash(x):return hashlib.sha256(x.contiguous().numpy().tobytes()).hexdigest()
def chunks_from_real(arr,spans,n,start=0):
 rows=[]
 for k in range(n):
  x=[];y=[]
  for s,_ in spans:
   p=s+(start+k)*CS;x.append(np.asarray(arr[p:p+CS]));y.append(np.asarray(arr[p+1:p+CS+1]))
  rows.append((torch.tensor(np.stack(x),dtype=torch.long),torch.tensor(np.stack(y),dtype=torch.long)))
 return rows
def validation_windows(arr,n,window_chunks=129):
 out=[]
 for s,e in find_books(arr,(16+window_chunks)*CS+1):
  pos=s+16*CS
  while pos+window_chunks*CS+1<=e and len(out)<n:
   out.append((pos,pos+window_chunks*CS+1));pos+=window_chunks*CS
  if len(out)>=n:break
 if len(out)<n:raise RuntimeError(f"only {len(out)} disjoint validation windows")
 return out
def generated(model,cfg,arr,spans,seed,total,write,take_from=1):
 real=torch.tensor(np.stack([np.asarray(arr[s:s+(WARMUP+1)*CS+1]) for s,_ in spans]),dtype=torch.long,device="cuda")
 st=StreamState(model,len(spans),"cuda")
 for c in range(WARMUP):st.process_real_chunk(real[:,c*CS:(c+1)*CS],real[:,c*CS+1:(c+1)*CS+1],1 if write else 0,1,cfg)
 first=real[:,WARMUP*CS:WARMUP*CS+1];out=[]
 for ordinal in range(1,total+1):
  seeds=[(seed*1000003+(s%100000)*7919+ordinal*65537)%(2**63-1) for s,_ in spans]
  gen=st.generate_chunk(first,write,1,cfg,temperature=1,top_p=.95,row_seeds=seeds,sampling_device="cuda")
  if ordinal>=take_from:out.append((torch.cat([first.cpu(),gen[:,:-1].cpu()],1),gen.cpu()))
  first=gen[:,-1:]
 return out
def record(a,cfg,model,arr):
 B=8 if a.group==0 else 4;start=a.group*8
 eligible=find_books(arr,(WARMUP+30)*CS+1);assert len(eligible)>=36
 receiver_all=eligible[-12:];receivers=receiver_all[start:start+B]
 validation_all=validation_windows(arr,12);validation_spans=validation_all[start:start+B]
 validation_book_starts={s for s,_ in find_books(arr,(16+129)*CS+1)}
 short=find_books(arr,(WARMUP+30)*CS+1);pool=[x for x in short if x not in receiver_all[:12] and x[0] not in validation_book_starts];assert len(pool)>=3*B
 trusted=chunks_from_real(arr,pool[:B],COUNTS[0]);clean=chunks_from_real(arr,pool[B:2*B],COUNTS[1])
 external=generated(model,cfg,arr,pool[2*B:3*B],a.seed,COUNTS[2],0)
 own=generated(model,cfg,arr,pool[:B],a.seed+100003,COUNTS[4],1)
 degraded=generated(model,cfg,arr,pool[B:2*B],a.seed+200003,121,1,97)[:COUNTS[3]]
 by={LABELS[0]:trusted,LABELS[1]:clean,LABELS[2]:external,LABELS[3]:degraded,LABELS[4]:own}
 # Freeze generated source identities, then place a trusted-real interruption
 # after every 16 generated candidates. Remaining real slots are placed at the
 # boundaries; hence the maximum generated burst is exactly 16.
 rng=np.random.default_rng(a.seed+20260921)
 generated_labels=sum(([lab]*n for lab,n in zip(LABELS[1:],COUNTS[1:])),[]);rng.shuffle(generated_labels)
 labels=[LABELS[0]]*3
 while generated_labels:
  labels.extend(generated_labels[:16]);del generated_labels[:16]
  if labels.count(LABELS[0])<COUNTS[0]:labels.append(LABELS[0])
 labels.extend([LABELS[0]]*(COUNTS[0]-labels.count(LABELS[0])))
 assert len(labels)==128 and max(len(list(g)) for k,g in __import__('itertools').groupby(labels) if k!=LABELS[0])==16
 at={x:0 for x in LABELS};candidates=[]
 for lab in labels:
  x,y=by[lab][at[lab]];at[lab]+=1;candidates.append((lab,x,y))
 validation=chunks_from_real(arr,validation_spans,129)
 prefix=chunks_from_real(arr,receivers,WARMUP)
 payload={"manifest":{"protocol":"hidden-provenance-stage-a-bundle-v1","seed":a.seed,"group":a.group,"batch":B,
  "receiver_spans":receivers,"validation_spans":validation_spans,"labels":LABELS,"counts":dict(zip(LABELS,COUNTS)),"candidate_order":labels,
  "checkpoint_sha256":sha(a.ckpt),"dataset_sha256":sha(a.val),"code_sha256":sha(__file__)},
  "prefix":prefix,"candidates":candidates,"validation":validation[:-1],"endpoint":validation[-1]}
 p=Path(a.out);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(".tmp");torch.save(payload,tmp);os.replace(tmp,p)
 save_json(str(p)+".json",{"status":"passed","manifest":payload["manifest"],"candidate_token_sha256":[tokhash(torch.cat([x,y[:,-1:]],1)) for _,x,y in candidates]})
def readonly(st,x,y,cfg):
 snap=st.snapshot();st.process_real_chunk(x.cuda(),y.cuda(),0,1,cfg);v=st.last_token_nll.mean(-1).cpu().tolist();st.restore(snap);return v
def run_policy(a,cfg,model):
 b=torch.load(a.bundle,map_location="cpu",weights_only=False);B=b["manifest"]["batch"];st=StreamState(model,B,"cuda")
 for x,y in b["prefix"]:st.process_real_chunk(x.cuda(),y.cuda(),1,1,cfg)
 random_accept=None
 if a.policy=="rate_random":
  ref=json.loads(Path(a.settlement_reference).read_text());rng=np.random.default_rng(a.seed+99173)
  random_accept=set()
  for lab in LABELS:
   ids=[i for i,x in enumerate(b["manifest"]["candidate_order"]) if x==lab];k=sum(ref["decisions"][i]["accepted"] for i in ids)
   random_accept.update(rng.choice(ids,k,replace=False).tolist() if k else [])
 decisions=[]
 for i,((lab,x,y),(qx,qy)) in enumerate(zip(b["candidates"],b["validation"])):
  x=x.cuda();y=y.cuda();pre=st.snapshot()
  if a.online and lab=="own_output":
   first=x[:,:1];rowseeds=[(a.seed*1000003+(a.group*8+row)*7919+i*65537)%(2**63-1) for row in range(B)]
   gen=st.generate_chunk(first,1,1,cfg,temperature=1,top_p=.95,row_seeds=rowseeds,sampling_device="cuda")
   x=torch.cat([first,gen[:,:-1]],1);y=gen
  else:st.process_real_chunk(x,y,1,1,cfg)
  proposed=[z.clone() for z in st.fast];update=st.last_update_stats
  st.restore(pre);st.process_real_chunk(x,y,0,1,cfg);base=trial=None
  if a.policy=="settlement":
   base=float(np.mean(readonly(st,qx,qy,cfg)));snap=st.snapshot();st.fast=[z.clone() for z in proposed];trial=float(np.mean(readonly(st,qx,qy,cfg)));st.restore(snap);accept=trial<=base-a.margin
  elif a.policy=="writes_off":accept=False
  elif a.policy=="all_writes":accept=True
  elif a.policy=="own_mask":accept=lab!="own_output"
  elif a.policy=="oracle":accept=lab=="trusted_real"
  else:accept=i in random_accept
  if accept:st.fast=[z.detach() for z in proposed]
  decisions.append({"index":i,"hidden_source":lab,"visible_own_output":lab=="own_output","accepted":bool(accept),"online_own_output":bool(a.online and lab=="own_output"),"consumed_token_sha256":tokhash(torch.cat([x.cpu(),y[:,-1:].cpu()],1)),"validation_base":base,"validation_trial":trial,"update":update})
 qx,qy=b["endpoint"];endpoint=readonly(st,qx,qy,cfg)
 save_json(a.out,{"status":"passed","manifest":{"protocol":"hidden-provenance-online-v1" if a.online else "hidden-provenance-stage-a-policy-v1","policy":a.policy,"seed":a.seed,"group":a.group,"online":a.online,
  "bundle":a.bundle,"bundle_sha256":sha(a.bundle),"margin":a.margin,"settlement_reference":a.settlement_reference or None,
  "checkpoint_sha256":sha(a.ckpt),"code_sha256":sha(__file__)},"decisions":decisions,"endpoint_nll_book":endpoint,
  "accepted":sum(x["accepted"] for x in decisions),"final_drift":drift(st),"peak_reserved_gib":torch.cuda.max_memory_reserved()/2**30})
def main():
 p=argparse.ArgumentParser();p.add_argument("--task",choices=("record","run"),required=True)
 for x in ("ckpt","val","out"):p.add_argument("--"+x,required=True)
 p.add_argument("--bundle");p.add_argument("--policy",choices=("writes_off","all_writes","own_mask","rate_random","settlement","oracle"));p.add_argument("--settlement-reference",default="")
 p.add_argument("--seed",type=int,required=True);p.add_argument("--group",type=int,choices=(0,1),required=True);p.add_argument("--margin",type=float,default=0)
 p.add_argument("--online",action="store_true")
 a=p.parse_args();save_json(str(a.out)+(".state" if a.task=="record" else ""),{"status":"running","started":time.time()})
 try:
  cfg,model=load(a);arr=np.load(a.val,mmap_mode="r")
  if a.task=="record":record(a,cfg,model,arr)
  else:assert a.bundle and a.policy;run_policy(a,cfg,model)
 except Exception:
  save_json(str(a.out)+".failed.json",{"status":"failed","error":traceback.format_exc()});raise
if __name__=="__main__":main()
