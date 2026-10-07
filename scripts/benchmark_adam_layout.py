"""Timing-only check: ordinary Adam with full versus chunked suffix layout.

No inner updates in either case. Chunked layout keeps the entire suffix KV
graph, so this comparison does not win speed by truncating gradients.
"""
import argparse
import gc
import json
from pathlib import Path
import statistics
import time
import numpy as np
import torch
import torch.utils.checkpoint
from ttt_pt.block_inner import suffix_block_forward
from ttt_pt.config import PRESETS
from ttt_pt.meta import masked_ce
from ttt_pt.model import TTTModel
from ttt_pt.shared_backward import loss_no_inner


def chunked_loss(model,x,y,mask,cfg):
    h=model.prefix_forward(x);fast=model.init_fast_weights(x.shape[0])
    kv=model.init_kv_caches(x.shape[0],x.device);losses=[];cs=cfg.model.mini_batch_size
    for c in range(0,x.shape[1]//cs,2):
        logits,kv=suffix_block_forward(model,h[:,c*cs:(c+2)*cs],fast,kv,c,double_backward=False)
        for j in range(2):
            losses.append(masked_ce(logits[:,j*cs:(j+1)*cs],y[:,(c+j)*cs:(c+j+1)*cs],mask[:,(c+j)*cs:(c+j+1)*cs])[0].mean())
    return torch.stack(losses).mean()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--ckpt',required=True);ap.add_argument('--data',required=True);ap.add_argument('--out',required=True)
    ap.add_argument('--preset',default='125m-e2e-ext32k',choices=sorted(PRESETS))
    a=ap.parse_args();torch.set_num_threads(4);torch.empty(1,device='cuda')
    for _ in range(60):
        if torch.cuda.mem_get_info()[0]>100*1024**3:break
        time.sleep(1)
    else:raise RuntimeError('GPU busy')
    cfg=PRESETS[a.preset]();tokens=np.load(a.data,mmap_mode='r')
    order=np.random.default_rng(1).permutation((len(tokens)-1)//16384)[:4]
    batches=[torch.as_tensor(np.array(tokens[int(i)*16384:int(i)*16384+16385],dtype=np.int64)[None],device='cuda') for i in order]
    output=dict(note='Timing-only kernel layout audit, same architecture/no writes/full KV gradient; 2 warmup + 5 timed full AdamW steps, accumulation 4.',rows={})
    gradients={};initial_losses={}
    for name,fn in [('full',loss_no_inner),('chunked',chunked_loss)]:
        model=TTTModel(cfg.model,max_seq_len=134144).cuda().train()
        checkpoint=torch.load(a.ckpt,map_location='cpu',weights_only=False)
        model.load_state_dict(checkpoint['model'] if 'model' in checkpoint else checkpoint);del checkpoint
        # Gradient audit on identical initial weights and first training sample.
        x,y=batches[0][:,:-1],batches[0][:,1:]
        loss=fn(model,x,y,y!=128000,cfg);loss.backward()
        initial_losses[name]=float(loss.detach())
        gradients[name]=torch.cat([p.grad.detach().float().flatten().cpu() for p in model.parameters()])
        model.zero_grad(set_to_none=True)
        opt=torch.optim.AdamW(model.parameters(),lr=4e-4,betas=(.9,.95),weight_decay=.1,eps=1e-8,fused=True)
        rows=[]
        for step in range(7):
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();t=time.monotonic();opt.zero_grad(set_to_none=True)
            for batch in batches:
                x,y=batch[:,:-1],batch[:,1:];(fn(model,x,y,y!=128000,cfg)/4).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();torch.cuda.synchronize()
            rows.append(dict(s=time.monotonic()-t,peak_gib=torch.cuda.max_memory_allocated()/1024**3))
        output['rows'][name]=dict(median_step_s=statistics.median(r['s'] for r in rows[2:]),peak_gib=max(r['peak_gib'] for r in rows),runs=rows)
        del opt,model,loss;gc.collect();torch.cuda.empty_cache()
    g,h=gradients['full'],gradients['chunked']
    output['initial_losses']=initial_losses
    # Float32 reductions over billions of coordinates can even report cosine
    # > 1. Accumulate FP64 dot products/norms in bounded CPU chunks instead.
    dot=gg=hh=dd=0.
    for i in range(0,g.numel(),4_000_000):
        u,v=g[i:i+4_000_000].double(),h[i:i+4_000_000].double()
        dot+=float((u*v).sum());gg+=float(u.square().sum());hh+=float(v.square().sum())
        dd+=float((u-v).square().sum())
    output['gradient_reduction_dtype']='float64'
    output['initial_gradient_cosine']=dot/(gg*hh)**.5
    output['initial_gradient_relative_l2']=(dd/gg)**.5
    Path(a.out).write_text(json.dumps(output,indent=2)+'\n');print(json.dumps(output),flush=True)


if __name__=='__main__':main()
