"""P3 controlled 3B long-stream retrieval, two documents per GPU cell."""
import argparse,hashlib,json,os,time,traceback
from pathlib import Path
import numpy as np
import torch
from horizon import BOS,CS,find_books,drift
from ttt_pt.config import PRESETS
from ttt_pt.model import TTTModel
from ttt_pt.stream import StreamState
WARMUP=8;HORIZONS=(32,64,128)
def sha(p):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for b in iter(lambda:f.read(8<<20),b""):h.update(b)
 return h.hexdigest()
def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(".tmp");t.write_text(json.dumps(x,indent=1)+"\n");os.replace(t,p)
def documents(arr,n=64):
 books=find_books(arr,(WARMUP+1)*CS+1)
 if len(books)<n:raise RuntimeError(f"only {len(books)} eligible documents")
 return [(i,s,s+(WARMUP+1)*CS+1) for i,(s,_) in enumerate(books[:n])]
def validation_spans(arr,group):
 books=find_books(arr,132*CS+1);assert books
 return [books[(2*group+b)%len(books)] for b in range(2)]
def frames(tok,name,code):
 stmt=tok("Memorize this record: the registry code for ",add_special_tokens=False)["input_ids"]+[name]+tok(" is ",add_special_tokens=False)["input_ids"]+[code]+tok(".",add_special_tokens=False)["input_ids"]
 query=tok("Retrieve the code assigned in the registry to ",add_special_tokens=False)["input_ids"]+[name]+tok(":",add_special_tokens=False)["input_ids"]
 return stmt,query
def readonly(st,x,y,cfg):
 snap=st.snapshot();st.process_real_chunk(x,y,0,1,cfg);v=st.last_token_nll.mean(-1).cpu().tolist();st.restore(snap);return v
@torch.no_grad()
def score_queries(st,queries,codes,candidates):
 snap=st.snapshot();results=[]
 for qi in range(4):
  q=torch.tensor([queries[b][qi] for b in range(st.B)],device="cuda")
  h=st.prefix_chunk(q);st.global_pos+=q.shape[1];logits=None
  for t in range(q.shape[1]):logits=st.suffix_token(h[:,t:t+1])
  lp=torch.log_softmax(logits.float(),-1)
  for b in range(st.B):
   cand=candidates[b];true=codes[b][qi];vals=-lp[b,torch.tensor(cand,device="cuda")]
   ti=cand.index(true);results.append({"book_row":b,"question":qi,"true_token":true,"true_nll":float(vals[ti]),"rank":int((vals<vals[ti]).sum())+1,"correct":bool(vals[ti]==vals.min())})
  st.restore(snap);snap=st.snapshot()
 st.restore(snap);return results
def main():
 p=argparse.ArgumentParser();p.add_argument("--ckpt",required=True);p.add_argument("--val",required=True);p.add_argument("--out",required=True);p.add_argument("--group",type=int,required=True);p.add_argument("--seed",type=int,required=True)
 p.add_argument("--policy",choices=("writes_off","closed","fixed_w0","settlement"),required=True);p.add_argument("--tokenizer",default="NousResearch/Meta-Llama-3-8B");p.add_argument("--margin",type=float,default=0)
 p.add_argument("--n-chunks",type=int,default=128)
 a=p.parse_args();save(a.out,{"status":"running","started":time.time()})
 try:
  from transformers import AutoTokenizer
  tok=AutoTokenizer.from_pretrained(a.tokenizer);cfg=PRESETS["official-3b-ext128k"]();model=TTTModel(cfg.model,max_seq_len=132*CS).cuda().eval()
  raw=torch.load(a.ckpt,map_location="cpu",weights_only=False);model.load_state_dict(raw.get("model",raw),strict=True);del raw
  arr=np.load(a.val,mmap_mode="r");wins=documents(arr);chosen=wins[2*a.group:2*a.group+2];assert len(chosen)==2
  real=torch.tensor(np.stack([np.asarray(arr[s:e]) for _,s,e in chosen]),dtype=torch.long)
  vspans=validation_spans(arr,a.group);validation=torch.tensor(np.stack([np.asarray(arr[s:s+130*CS+1]) for s,_ in vspans]),dtype=torch.long)
  freq=np.bincount(np.asarray(arr[:min(len(arr),5_000_000)]),minlength=128256);band=np.where((freq>5)&(freq<60))[0];rng=np.random.default_rng(20260921+a.group)
  picks=rng.choice(band,2*(4+32),replace=False).reshape(2,-1);names=picks[:,:4].tolist();codes=picks[:,4:8].tolist();candidates=[codes[b]+picks[b,8:36].tolist() for b in range(2)]
  statements=[];queries=[]
  for b in range(2):
   fs=[frames(tok,int(names[b][q]),int(codes[b][q])) for q in range(4)];statements.append([z[0] for z in fs]);queries.append([z[1] for z in fs])
  for b in range(2):
   for q,stmt in enumerate(statements[b]):
    pos=(q+1)*CS;real[b,pos:pos+len(stmt)]=torch.tensor(stmt)
  st=StreamState(model,2,"cuda");generator=StreamState(model,2,"cuda") if a.policy=="fixed_w0" else None
  for c in range(WARMUP):
   x=real[:,c*CS:(c+1)*CS].cuda();y=real[:,c*CS+1:(c+1)*CS+1].cuda();st.process_real_chunk(x,y,1,1,cfg)
   if generator is not None:generator.process_real_chunk(x,y,0,1,cfg)
  first=real[:,WARMUP*CS:WARMUP*CS+1].cuda();answers=[];decisions=[]
  for c in range(WARMUP,a.n_chunks):
   rowseeds=[(a.seed*1000003+(2*a.group+b)*7919+c*65537)%(2**63-1) for b in range(2)]
   if a.policy=="fixed_w0":
    gen=generator.generate_chunk(first,0,1,cfg,row_seeds=rowseeds,sampling_device="cuda");st.process_real_chunk(torch.cat([first,gen[:,:-1]],1),gen,1,1,cfg)
   elif a.policy=="settlement":
    pre=st.snapshot();gen=st.generate_chunk(first,1,1,cfg,row_seeds=rowseeds,sampling_device="cuda");proposed=[z.clone() for z in st.fast]
    st.restore(pre);st.process_real_chunk(torch.cat([first,gen[:,:-1]],1),gen,0,1,cfg)
    qpos=(c-WARMUP)*CS;qx=validation[:,qpos:qpos+CS].cuda();qy=validation[:,qpos+1:qpos+CS+1].cuda();base=float(np.mean(readonly(st,qx,qy,cfg)));snap=st.snapshot();st.fast=[z.clone() for z in proposed];trial=float(np.mean(readonly(st,qx,qy,cfg)));st.restore(snap);accept=trial<=base-a.margin
    if accept:st.fast=[z.detach() for z in proposed]
    decisions.append({"chunk":c,"accepted":bool(accept),"base":base,"trial":trial})
   else:gen=st.generate_chunk(first,1 if a.policy=="closed" else 0,1,cfg,row_seeds=rowseeds,sampling_device="cuda")
   first=gen[:,-1:]
   if c+1 in HORIZONS:
    scored=score_queries(st,queries,codes,candidates)
    for z in scored:z["horizon_chunks"]=c+1
    answers.extend(scored)
  save(a.out,{"status":"passed","manifest":{"protocol":"acceptance-retrieval-3b-v1","policy":a.policy,"seed":a.seed,"group":a.group,"document_windows":chosen,"validation_spans":vspans,"n_chunks":a.n_chunks,"horizons":[x for x in HORIZONS if x<=a.n_chunks],"questions_per_horizon":4,"checkpoint_sha256":sha(a.ckpt),"dataset_sha256":sha(a.val),"code_sha256":sha(__file__)},"questions":answers,"decisions":decisions,"final_drift":drift(st),"peak_reserved_gib":torch.cuda.max_memory_reserved()/2**30})
 except Exception:
  save(a.out,{"status":"failed","error":traceback.format_exc()});raise
if __name__=="__main__":main()
