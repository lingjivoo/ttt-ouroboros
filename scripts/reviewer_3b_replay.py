"""P0-B: generate fixed 3B source streams or replay them into disjoint receivers."""
import argparse,hashlib,json,os,sys,time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from horizon import CS,find_books,probe_branched,drift
from ttt_pt.config import PRESETS
from ttt_pt.model import TTTModel
from ttt_pt.stream import StreamState

def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def token_sha(x):return hashlib.sha256(x.cpu().contiguous().numpy().tobytes()).hexdigest()
def save_json(path,obj):
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(obj,indent=1)+'\n');t.replace(p)
def load_model(a):
 cfg=PRESETS['official-3b-ext128k']();model=TTTModel(cfg.model,max_seq_len=130*CS).cuda().eval()
 raw=torch.load(a.ckpt,map_location='cuda',weights_only=False);model.load_state_dict(raw.get('model',raw),strict=True)
 return cfg,model
def record(a,cfg,model,arr,books):
 indices=[a.source_offset+2*a.group,a.source_offset+2*a.group+1]
 spans=[books[i] for i in indices]
 real=torch.tensor(np.stack([arr[s:s+8*CS+1] for s,_ in spans]),dtype=torch.long,device='cuda')
 st=StreamState(model,2,'cuda');stream=[]
 for c in range(8):st.process_real_chunk(real[:,c*CS:(c+1)*CS],real[:,c*CS+1:(c+1)*CS+1],1,1,cfg)
 first=real[:,8*CS:8*CS+1]
 for g in range(120):
  seeds=[(a.seed*1000003+idx*7919+g*65537)%(2**63-1) for idx in indices]
  gen=st.generate_chunk(first,1,1,cfg,seed=a.seed*100000+g,row_seeds=seeds,sampling_device='cuda')
  stream.append((g,first.cpu(),gen.cpu()))
  if (g+1)%10==0:print('generated',g+1,flush=True)
 payload={'manifest':{'protocol':'reviewer-p0b-source-v1','seed':a.seed,'source_indices':indices,
  'source_spans':spans,'prefill_chunks':8,'generated_chunks':120,'chunk_tokens':CS,
  'temperature':1.0,'top_p':.95,'sampling_device':'cuda','checkpoint_sha256':sha(a.ckpt),
  'dataset_sha256':sha(a.val),'code_sha256':sha(__file__),'adaptation_state_ownership':'independent fast-weight tensors per batch row'},'stream':stream}
 out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True);tmp=out.with_suffix('.tmp');torch.save(payload,tmp);tmp.replace(out)
 save_json(str(out)+'.json',{'status':'passed','manifest':payload['manifest'],'token_sha256':[token_sha(torch.cat([x[2][b] for x in stream])) for b in range(2)]})
def replay(a,cfg,model,arr,books):
 payload=torch.load(a.record,map_location='cpu',weights_only=False);m=payload['manifest'];assert m['seed']==a.seed
 source_indices=m['source_indices'];receiver_pool=list(range(a.receiver_offset,a.receiver_offset+6))
 receiver_indices=[receiver_pool[(i-a.source_offset+a.mapping)%6] for i in source_indices]
 spans=[books[i] for i in receiver_indices]
 real=torch.tensor(np.stack([arr[s:s+16*CS+1] for s,_ in spans]),dtype=torch.long,device='cuda')
 st=StreamState(model,2,'cuda')
 for c in range(8):st.process_real_chunk(real[:,c*CS:(c+1)*CS],real[:,c*CS+1:(c+1)*CS+1],1,1,cfg)
 q=[(real[:,c*CS:(c+1)*CS],real[:,c*CS+1:(c+1)*CS+1]) for c in range(8,16)]
 def score():return np.mean([probe_branched(st,x,y,cfg,True).float().cpu().numpy() for x,y in q],axis=0).tolist()
 curve=[{'source_chunk':0,'nll_book':score(),'drift':drift(st)}];alltokens=[]
 for j,(_,first,gen) in enumerate(payload['stream']):
  inp=torch.cat([first,gen[:,:-1]],1).cuda();tgt=gen.cuda();alltokens.append(tgt.cpu())
  st.process_real_chunk(inp,tgt,1 if a.condition=='write' else 0,1,cfg)
  if (j+1)%8==0 or j==119:curve.append({'source_chunk':j+1,'nll_book':score(),'drift':drift(st)})
 result={'status':'passed','manifest':{'protocol':'reviewer-p0b-replay-v1','seed':a.seed,'group':a.group,
  'mapping':a.mapping,'condition':a.condition,'source_indices':source_indices,'source_spans':m['source_spans'],
  'receiver_indices':receiver_indices,'receiver_spans':spans,'prefill_chunks':8,'source_chunks':120,
  'evaluation_chunks':8,'source_record':a.record,'source_record_sha256':sha(a.record),
  'received_token_sha256':[token_sha(torch.cat([x[b] for x in alltokens])) for b in range(2)],
  'checkpoint_sha256':sha(a.ckpt),'dataset_sha256':sha(a.val),'code_sha256':sha(__file__),
  'adaptation_state_ownership':'independent fast-weight tensors per batch row'},'curve':curve,
  'first_nll_book':curve[0]['nll_book'],'last_nll_book':curve[-1]['nll_book'],
  'change_nll_book':(np.array(curve[-1]['nll_book'])-curve[0]['nll_book']).tolist()}
 save_json(a.out,result)
def main():
 p=argparse.ArgumentParser();p.add_argument('--task',choices=('record','replay'),required=True)
 for x in ('ckpt','val','out'):p.add_argument('--'+x,required=True)
 p.add_argument('--record');p.add_argument('--seed',type=int,required=True);p.add_argument('--group',type=int,choices=(0,1,2),required=True)
 p.add_argument('--mapping',type=int,choices=(0,1),default=0);p.add_argument('--condition',choices=('read','write'),default='read')
 p.add_argument('--source-offset',type=int,default=12);p.add_argument('--receiver-offset',type=int,default=2)
 a=p.parse_args();t=time.time();arr=np.load(a.val,mmap_mode='r');books=find_books(arr,130*CS+1);assert len(books)>=18
 assert set(range(a.source_offset,a.source_offset+6)).isdisjoint(range(a.receiver_offset,a.receiver_offset+6))
 cfg,model=load_model(a);torch.cuda.reset_peak_memory_stats()
 if a.task=='record':record(a,cfg,model,arr,books)
 else:
  assert a.record;replay(a,cfg,model,arr,books)
 print('passed',a.task,a.out,'seconds',time.time()-t,'peak_gib',torch.cuda.max_memory_reserved()/2**30,flush=True)
if __name__=='__main__':main()
