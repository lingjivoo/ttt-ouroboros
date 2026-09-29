"""Independent closed-loop source trajectories for heavy-tail replay analysis."""
import argparse,hashlib,json,os,time,traceback
from pathlib import Path
import numpy as np
import torch
from horizon import BOS,CS,WARMUP,find_books,drift
from ttt_pt.config import PRESETS
from ttt_pt.model import TTTModel
from ttt_pt.stream import StreamState

def sha(p):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for b in iter(lambda:f.read(8<<20),b""):h.update(b)
 return h.hexdigest()
def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(".tmp");t.write_text(json.dumps(x,indent=1)+"\n");os.replace(t,p)
def token_hash(chunks):
 h=hashlib.sha256()
 for first,gen in chunks:h.update(first.numpy().tobytes());h.update(gen.numpy().tobytes())
 return h.hexdigest()
def probe(st,x,y,cfg):
 snap=st.snapshot();st.process_real_chunk(x,y,0,1,cfg);v=st.last_token_nll.mean(-1).cpu().tolist();st.restore(snap);return v
def fixed_probe(model,eval_snap,fast,x,y,cfg):
 """Evaluate one fast-weight state against an invariant receiver/Q context."""
 ev=StreamState(model,len(fast[0]),"cuda");ev.restore(eval_snap)
 ev.fast=[z.detach().clone() for z in fast]
 ev.process_real_chunk(x,y,0,1,cfg)
 return ev.last_token_nll.mean(-1).cpu().tolist()

def main():
 p=argparse.ArgumentParser();p.add_argument("--ckpt",required=True);p.add_argument("--val",required=True);p.add_argument("--out",required=True)
 p.add_argument("--source-book",type=int,required=True);p.add_argument("--seed",type=int,required=True)
 p.add_argument("--stages",default="1,33,65,97");p.add_argument("--passage-chunks",type=int,default=8)
 p.add_argument("--telescoping-all",action="store_true")
 a=p.parse_args();started=time.time();save(a.out,{"status":"running"})
 try:
  stages=tuple(int(x) for x in a.stages.split(","));last=max(stages)+a.passage_chunks-1
  cfg=PRESETS["125m-e2e-ext32k"]();model=TTTModel(cfg.model,max_seq_len=(WARMUP+last+4)*CS).cuda().eval()
  raw=torch.load(a.ckpt,map_location="cpu",weights_only=False);model.load_state_dict(raw.get("model",raw),strict=True);del raw
  data=np.load(a.val,mmap_mode="r");books=find_books(data,(WARMUP+3)*CS+1)
  source_span=books[a.source_book];receiver_ids=[8+a.source_book,16+a.source_book,24+a.source_book]
  assert max(receiver_ids)<len(books);receiver_spans=[books[i] for i in receiver_ids]
  source=torch.from_numpy(np.asarray(data[source_span[0]:source_span[0]+(WARMUP+2)*CS+1]).astype(np.int64)).view(1,-1)
  src=StreamState(model,1,"cuda")
  for c in range(WARMUP):
   x=source[:,c*CS:(c+1)*CS].cuda();y=source[:,c*CS+1:(c+1)*CS+1].cuda();src.process_real_chunk(x,y,1,1,cfg)
  previous=None;record=[];row_seed=(a.source_book*1000003+a.seed*7919)%(2**63-1)
  for ordinal in range(1,last+1):
   first=source[:,WARMUP*CS:WARMUP*CS+1].cuda() if previous is None else previous
   gen=src.generate_chunk(first,1,1,cfg,temperature=1,top_p=.95,row_seeds=[row_seed+ordinal*65537],sampling_device="cuda")
   record.append((first.cpu(),gen.cpu()));previous=gen[:,-1:]
   if ordinal%8==0:print(a.source_book,a.seed,ordinal,flush=True)
  need=(WARMUP+2)*CS+1
  real=torch.from_numpy(np.stack([np.asarray(data[s:s+need]).astype(np.int64) for s,_ in receiver_spans]))
  base=StreamState(model,3,"cuda")
  for c in range(WARMUP):base.process_real_chunk(real[:,c*CS:(c+1)*CS].cuda(),real[:,c*CS+1:(c+1)*CS+1].cuda(),1,1,cfg)
  base_snap=base.snapshot();qx=real[:,WARMUP*CS:(WARMUP+1)*CS].cuda();qy=real[:,WARMUP*CS+1:(WARMUP+1)*CS+1].cuda()
  rows=[]
  for stage in stages:
   chunks=record[stage-1:stage-1+a.passage_chunks];entry={"stage":stage,"passage_sha256":token_hash(chunks),"conditions":{}}
   for condition,write in (("read_only",0),("read_write",1)):
    st=StreamState(model,3,"cuda");st.restore(base_snap);updates=[]
    fixed_nll=[fixed_probe(model,base_snap,st.fast,qx,qy,cfg)]
    for first,gen in chunks:
     fi=first.expand(3,-1).cuda();ge=gen.expand(3,-1).cuda();st.process_real_chunk(torch.cat([fi,ge[:,:-1]],1),ge,write,1,cfg)
     updates.append(st.last_update_stats)
     fixed_nll.append(fixed_probe(model,base_snap,st.fast,qx,qy,cfg))
    fixed_arr=np.asarray(fixed_nll,dtype=np.float64)
    deltas=np.diff(fixed_arr,axis=0)
    entry["conditions"][condition]={"probe_nll_book":probe(st,qx,qy,cfg),"fixed_context_nll_book":fixed_nll,
     "fixed_context_delta_book":deltas.tolist(),"fixed_context_cumulative_book":(fixed_arr-fixed_arr[0]).tolist(),
     "telescoping_residual_book":(deltas.sum(0)-(fixed_arr[-1]-fixed_arr[0])).tolist(),
     "final_drift":drift(st),"updates":updates}
   r=np.asarray(entry["conditions"]["read_only"]["probe_nll_book"]);w=np.asarray(entry["conditions"]["read_write"]["probe_nll_book"])
   entry["write_damage_book"]=(w-r).tolist();rows.append(entry)
  full_telescoping=None
  if a.telescoping_all:
   st=StreamState(model,3,"cuda");st.restore(base_snap)
   nll=[fixed_probe(model,base_snap,st.fast,qx,qy,cfg)];updates=[]
   for ordinal,(first,gen) in enumerate(record,1):
    fi=first.expand(3,-1).cuda();ge=gen.expand(3,-1).cuda()
    st.process_real_chunk(torch.cat([fi,ge[:,:-1]],1),ge,1,1,cfg)
    updates.append(st.last_update_stats);nll.append(fixed_probe(model,base_snap,st.fast,qx,qy,cfg))
    if ordinal%8==0:print("telescoping",a.source_book,a.seed,ordinal,flush=True)
   arr=np.asarray(nll,dtype=np.float64);delta=np.diff(arr,axis=0);total=arr[-1]-arr[0]
   order=np.argsort(-delta,axis=0)
   shares={}
   for frac in (.01,.05,.10):
    k=max(1,int(np.ceil(len(delta)*frac)))
    vals=np.take_along_axis(delta,order[:k],axis=0).sum(0)
    shares[str(frac)]=(vals/np.where(np.abs(total)>1e-12,total,np.nan)).tolist()
   full_telescoping={"fixed_context_nll_book":nll,"delta_book":delta.tolist(),
    "cumulative_book":(arr-arr[0]).tolist(),"total_book":total.tolist(),
    "telescoping_residual_book":(delta.sum(0)-total).tolist(),"top_positive_damage_fraction":shares,
    "ranked_write_indices_book":order.T.tolist(),"updates":updates}
  out={"status":"passed","manifest":{"protocol":"acceptance-heavy-tail-v1","source_book":a.source_book,"source_span":source_span,
   "receiver_book_ids":receiver_ids,"receiver_spans":receiver_spans,"seed":a.seed,"stages":stages,"passage_chunks":a.passage_chunks,
   "independent_unit":"source_book_x_seed","primary_receiver_index":0,"robustness_receiver_indices":[1,2],"temperature":1,"top_p":.95,
   "checkpoint_sha256":sha(a.ckpt),"dataset_sha256":sha(a.val),"code_sha256":sha(__file__)},"stages":rows,
   "source_final_drift":drift(src),"full_trajectory_telescoping":full_telescoping,
   "seconds":time.time()-started,"peak_reserved_gib":torch.cuda.max_memory_reserved()/2**30}
  save(a.out,out);print("passed",a.out,flush=True)
 except Exception:
  save(a.out,{"status":"failed","error":traceback.format_exc(),"seconds":time.time()-started});raise
if __name__=="__main__":main()
