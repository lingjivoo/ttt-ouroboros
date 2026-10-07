"""Paired continuation of one TTT checkpoint under four training rules.

Adam baseline retains the exact prime architecture but makes no inner writes.
All methods see the same sampled token windows and use fresh AdamW state.
Shared backward truncates suffix KV credit, NOT the full prefix graph.
"""
import argparse
import dataclasses
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch

from ttt_pt.block_inner import BlockStreamState, loss_for_sequence_block
from ttt_pt.config import PRESETS
from ttt_pt.meta import loss_for_sequence_meta
from ttt_pt.model import TTTModel
from ttt_pt.shared_backward import backward_shared, loss_no_inner
from ttt_pt.stream import StreamState


def save(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2) + '\n')
    tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(16 << 20), b''):
            h.update(block)
    return h.hexdigest()


def training_backward(method, model, batch, cfg, accum, exact_remat=False):
    x, y = batch[:, :-1], batch[:, 1:]
    mask = y != cfg.model.bos_token_id
    if method == 'shared':
        return backward_shared(model, x, y, mask, cfg, block_chunks=2, outer_scale=1/accum)
    if method == 'adam':
        loss = loss_no_inner(model, x, y, mask, cfg)
    elif method == 'exact':
        if exact_remat:
            from ttt_pt.remat import loss_for_sequence_meta_remat
            loss, _ = loss_for_sequence_meta_remat(model, x, y, mask, 1., cfg)
        else:
            loss, _ = loss_for_sequence_meta(model, x, y, mask, 1., cfg, create_graph=True)
    else:
        loss, _ = loss_for_sequence_block(model, x, y, mask, 1., cfg, 2,
                                          create_graph=True, parallel_read=True, first_order=True)
    (loss/accum).backward()
    return loss.detach(), {}


def make_state(model, batch, k):
    if k == 1:
        return StreamState(model, batch, torch.device('cuda'))
    return BlockStreamState(model, batch, torch.device('cuda'), 2, parallel_update=True)


def probe_rows(state, inputs, targets, cfg):
    snapshot = state.snapshot()
    try:
        return state.process_real_chunk(inputs, targets, 0., 1., cfg).cpu().tolist()
    finally:
        state.restore(snapshot)


@torch.no_grad()
def evaluate(model, cfg, books, out, tag, chunks, eval_batch_size=4):
    model.eval()
    cs = cfg.model.mini_batch_size
    result = dict(status='running', chunks=chunks, books=len(books), eval_batch_size=eval_batch_size, policies={})
    for k in (0, 1, 2):
        started = time.monotonic()
        batches, endpoint = [], []
        for offset in range(0, len(books), eval_batch_size):
            group = books[offset:offset+eval_batch_size]
            seq = torch.as_tensor(np.stack([np.asarray(b[:(chunks+1)*cs+1],dtype=np.int64) for b in group]),device='cuda')
            state = make_state(model,len(group),k)
            values = []
            for c in range(chunks):
                nll = state.process_real_chunk(seq[:,c*cs:(c+1)*cs],seq[:,c*cs+1:(c+1)*cs+1],float(k>0),1.,cfg)
                values.append(nll.cpu().tolist())
            endpoint.extend(probe_rows(state,seq[:,chunks*cs:(chunks+1)*cs],seq[:,chunks*cs+1:(chunks+1)*cs+1],cfg))
            batches.append(np.asarray(values))
            del state, seq
        rows = np.concatenate(batches,axis=1)
        result['policies'][str(k)] = dict(chunk_nll=rows.tolist(), mean_nll_per_book=rows.mean(0).tolist(),
                      last32k_nll_per_book=rows[-min(32,chunks):].mean(0).tolist(),
                      endpoint_nll_per_book=endpoint, elapsed_s=time.monotonic()-started)
        save(out / (tag+'.json'), result)
        print(json.dumps(dict(event='real_eval', tag=tag, k=k, mean=float(rows.mean()), endpoint=float(np.mean(endpoint)))), flush=True)
    off = np.asarray(result['policies']['0']['last32k_nll_per_book'])
    for k in ('1','2'):
        result['policies'][k]['last32k_benefit_per_book'] = (off-np.asarray(result['policies'][k]['last32k_nll_per_book'])).tolist()
    result['status'] = 'complete'
    save(out / (tag+'.json'), result)
    model.train()
    return result


def diversity(tokens):
    rows=[]
    for ids in tokens.tolist():
        d2=len(set(zip(ids[:-1],ids[1:])))/max(1,len(ids)-1)
        fours=list(zip(ids,ids[1:],ids[2:],ids[3:]))
        r4=1-len(set(fours))/max(1,len(fours))
        rows.append(dict(distinct2=d2,repeated4=r4))
    return rows


@torch.no_grad()
def generated_eval(model, cfg, books, out, method, seed, chunks=16):
    # Same common K=2 deployment across all four checkpoints isolates the
    # trained weights; separate real-text evaluation also reports native K=1.
    model.eval()
    cs=cfg.model.mini_batch_size
    seq=torch.as_tensor(np.stack([np.asarray(b[:66*cs+1],dtype=np.int64) for b in books[:2]]),device='cuda')
    result=dict(status='running', method=method, seed=seed, K=2, prefill_chunks=4,
                generated_chunks=chunks, temperature=1., top_p=.95, policies={})
    qx,qy=seq[:,64*cs:65*cs],seq[:,64*cs+1:65*cs+1]
    for pol in ('off','closed'):
        state=make_state(model,seq.shape[0],2)
        for c in range(4):
            state.process_real_chunk(seq[:,c*cs:(c+1)*cs],seq[:,c*cs+1:(c+1)*cs+1],1.,1.,cfg)
        row=dict(initial_probe=probe_rows(state,qx,qy,cfg),probes=[])
        first=seq[:,4*cs:4*cs+1]
        generated=[]
        started=time.monotonic()
        result['policies'][pol]=row
        for c in range(chunks):
            tokens=state.generate_chunk(first,float(pol=='closed'),1.,cfg,temperature=1.,top_p=.95,
                                        seed=seed*100000+c,sampling_device='cuda')
            generated.append(tokens.cpu())
            first=tokens[:,-1:]
            if (c+1)%4==0:
                row['probes'].append(dict(chunks=c+1,nll=probe_rows(state,qx,qy,cfg)))
                save(out/'generated.json',result)
                print(json.dumps(dict(event='generated_eval',policy=pol,chunks=c+1)),flush=True)
        row.update(final_probe=probe_rows(state,qx,qy,cfg),diversity=diversity(torch.cat(generated,1)),elapsed_s=time.monotonic()-started)
        del state
    a,b=result['policies']['closed'],result['policies']['off']
    result['endpoint_excess_per_book']=(np.array(a['final_probe'])-np.array(b['final_probe'])).tolist()
    result['anchored_harm_per_book']=((np.array(a['final_probe'])-a['initial_probe'])-(np.array(b['final_probe'])-b['initial_probe'])).tolist()
    result['status']='complete'
    save(out/'generated.json',result)
    return result


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--method',required=True,choices=('adam','exact','fo','shared'))
    ap.add_argument('--preset',default='125m-e2e-ext32k',choices=sorted(PRESETS))
    ap.add_argument('--ckpt',required=True)
    ap.add_argument('--data',required=True)
    ap.add_argument('--out',required=True)
    ap.add_argument('--steps',type=int,default=120)
    ap.add_argument('--accum',type=int,default=8)
    ap.add_argument('--seq-length',type=int,default=8192)
    ap.add_argument('--seed',type=int,default=1)
    ap.add_argument('--smoke',action='store_true')
    ap.add_argument('--skip-generation',action='store_true')
    ap.add_argument('--exact-remat',action='store_true',help='Exact K=1 gradients with chunk rematerialization')
    ap.add_argument('--eval-batch-size',type=int,default=4)
    ap.add_argument('--peak-lr',type=float,default=4e-4)
    args=ap.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists():
        raise RuntimeError('completed output already exists; refusing overwrite')
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    torch.empty(1,device='cuda')
    deadline=time.monotonic()+60
    while torch.cuda.mem_get_info()[0]<110*1024**3:
        if time.monotonic()>deadline:
            raise RuntimeError('GPU did not yield 110 GiB; no process killed')
        time.sleep(1)
    cfg=PRESETS[args.preset]()
    cfg.training.seq_length=args.seq_length
    cfg.training.total_steps=args.steps
    cfg.training.global_batch_size=args.accum
    cfg.training.accum_steps=args.accum
    cfg.training.ilr_warmup_steps=0
    cfg.training.ilr_init=1.
    cfg.training.optimizer_outer.lr=args.peak_lr
    cfg.training.optimizer_outer.lr_warmup_steps=12
    cfg.training.optimizer_outer.lr_decay_steps=args.steps
    cfg.training.optimizer_outer.end_lr=1e-5
    model=TTTModel(cfg.model,max_seq_len=max(134144,cfg.model.sliding_window_size+args.seq_length)).cuda()
    payload=torch.load(args.ckpt,map_location='cpu',weights_only=False)
    model.load_state_dict(payload['model'] if 'model' in payload else payload,strict=True)
    del payload
    data=Path(args.data)
    manifest=json.loads((data/'manifest.json').read_text())
    train=np.load(data/'train.npy',mmap_mode='r')
    books=[np.load(data/('test_'+b['book_id']+'.npy'),mmap_mode='r') for b in manifest['books']['test']]
    order=np.random.default_rng(args.seed).permutation((len(train)-1)//args.seq_length)
    if args.steps*args.accum>len(order):
        raise ValueError('insufficient distinct training windows')
    order=order[:args.steps*args.accum]
    config=dict(method=args.method,preset=args.preset,seed=args.seed,steps=args.steps,seq_length=args.seq_length,
                micro_batch=1,accumulation=args.accum,tokens=args.steps*args.accum*args.seq_length,
                checkpoint_sha256=sha(args.ckpt),dataset_manifest=manifest,
                training_window_order=order.tolist(),train_config=dataclasses.asdict(cfg),
                torch=torch.__version__,gpu=torch.cuda.get_device_name(),parameters=sum(p.numel() for p in model.parameters()),
                experiment='paired short-budget continuation, not training from scratch',
                attention_backend='existing method-specific SDPA paths; exact requires double backward',
                prefix_checkpointing=os.environ.get('TTT_CKPT_PREFIX','0'),
                exact_rematerialization=args.exact_remat,eval_batch_size=args.eval_batch_size,
                comparison='equal tokens; no-inner Adam retains identical prime architecture; no Settlement gate',
                lr=dict(peak=args.peak_lr,warmup=12,end=1e-5),optimizer=dict(name='AdamW',betas=[.9,.95],weight_decay=.1,eps=1e-8,clip=1.))
    root=Path(__file__).resolve().parents[1]
    config['code_sha256']={str(p.relative_to(root)):sha(p) for p in sorted((root/'ttt_pt').glob('*.py'))}
    config['code_sha256']['scripts/inner_training_comparison.py']=sha(__file__)
    if (out/'config.json').exists() and json.loads((out/'config.json').read_text())!=config:
        raise RuntimeError('resume configuration mismatch')
    save(out/'config.json',config)
    opt=torch.optim.AdamW(model.parameters(),lr=args.peak_lr,betas=(.9,.95),weight_decay=.1,eps=1e-8,fused=True)
    start=0
    if (out/'latest.pt').exists():
        resume=torch.load(out/'latest.pt',map_location='cpu',weights_only=False)
        model.load_state_dict(resume['model']);opt.load_state_dict(resume['opt']);start=resume['completed_steps']
        del resume
    started=time.monotonic()
    if not args.smoke and not (out/'initial_32k.json').exists():
        evaluate(model,cfg,books,out,'initial_32k',32,args.eval_batch_size)
    model.train()
    for step in range(start,args.steps):
        lr=args.peak_lr*(step+1)/12 if step<12 else 1e-5+.5*(args.peak_lr-1e-5)*(1+math.cos(math.pi*(step-12)/max(1,args.steps-12-1)))
        for group in opt.param_groups:group['lr']=lr
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();t=time.monotonic()
        opt.zero_grad(set_to_none=True)
        losses=[];aux={}
        for j in range(args.accum):
            s=int(order[step*args.accum+j])*args.seq_length
            batch=torch.as_tensor(np.array(train[s:s+args.seq_length+1],dtype=np.int64,copy=True)[None],device='cuda')
            loss,aux=training_backward(args.method,model,batch,cfg,args.accum,args.exact_remat)
            losses.append(float(loss))
        gnorm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        if not np.isfinite(losses).all():raise RuntimeError('nonfinite training loss')
        opt.step();torch.cuda.synchronize()
        row=dict(step=step+1,loss=float(np.mean(losses)),gnorm=float(gnorm),lr=lr,
                 step_s=time.monotonic()-t,peak_gib=torch.cuda.max_memory_allocated()/1024**3,
                 completed_tokens=(step+1)*args.accum*args.seq_length)
        if args.method=='shared':row.update({k:v for k,v in aux.items() if k!='fast'})
        with (out/'train.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
        save(out/'status.json',dict(status='training',**row))
        if not args.smoke and ((step+1)%40==0 or step+1==args.steps):
            torch.save(dict(model=model.state_dict(),opt=opt.state_dict(),completed_steps=step+1,config=config),out/'latest.tmp.pt')
            (out/'latest.tmp.pt').replace(out/'latest.pt')
            # Keep model-only checkpoints for later equal-wall-clock comparisons.
            torch.save(dict(model=model.state_dict(),completed_steps=step+1),out/('model_%04d.pt'%(step+1)))
    del opt
    model.zero_grad(set_to_none=True);torch.cuda.empty_cache()
    save(out/'status.json',dict(status='evaluating',completed_steps=args.steps))
    if not args.smoke:
        evaluate(model,cfg,books,out,'final_32k',32,args.eval_batch_size)
        evaluate(model,cfg,books,out,'final_128k',128,args.eval_batch_size)
        if not args.skip_generation:
            generated_eval(model,cfg,books,out,args.method,args.seed)
    result=dict(status='complete',method=args.method,completed_steps=args.steps,
                tokens=args.steps*args.accum*args.seq_length,elapsed_this_process_s=time.monotonic()-started,
                smoke=args.smoke)
    save(out/'complete.json',result);save(out/'status.json',result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()
