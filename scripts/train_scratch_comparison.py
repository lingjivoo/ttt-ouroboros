"""Four paired 125M from-scratch arms using the official 8K/2.517B budget.

This is a PyTorch recipe replication with public re-tokenized DCLM, not an
identical rerun of the official JAX implementation/pre-tokenized data order.
No pretrained checkpoint argument exists. Resumes require an exact config.
"""
import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from ttt_pt.block_inner import loss_for_sequence_block, suffix_block_forward
from ttt_pt.config import PRESETS
from ttt_pt.meta import ilr_multiplier, masked_ce, loss_for_sequence_meta
from ttt_pt.model import TTTModel
from ttt_pt.remat import loss_for_sequence_meta_remat
from ttt_pt.shared_backward import backward_shared
from ttt_pt.train import lr_at
from scripts.inner_training_comparison import evaluate, generated_eval, make_state, save, sha


OFFICIAL_COMMIT = 'a4fc4788ace38e29b5067916d4f4be33da894085'


def state_digest(model):
    h = hashlib.sha256()
    for name, value in model.state_dict().items():
        h.update(name.encode()); h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def no_inner_chunked(model, x, y, mask, cfg):
    """Ordinary Adam baseline with full prefix and suffix KV gradients."""
    h = model.prefix_forward(x)
    fast = model.init_fast_weights(x.shape[0])
    kv = model.init_kv_caches(x.shape[0], x.device)
    cs = cfg.model.mini_batch_size
    losses = []
    for c in range(0, x.shape[1]//cs, 2):
        stop = min(c+2, x.shape[1]//cs)
        logits, kv = suffix_block_forward(model, h[:,c*cs:stop*cs], fast, kv, c, double_backward=False)
        for j in range(stop-c):
            losses.append(masked_ce(logits[:,j*cs:(j+1)*cs], y[:,(c+j)*cs:(c+j+1)*cs],
                                    mask[:,(c+j)*cs:(c+j+1)*cs])[0].mean())
    return torch.stack(losses).mean()


def backward(method, model, batch, cfg, step, accum, remat=False):
    x, y = batch[:,:-1], batch[:,1:]
    mask = y != cfg.model.bos_token_id
    mult = ilr_multiplier(step, cfg)
    if method == 'shared':
        return backward_shared(model,x,y,mask,cfg,block_chunks=2,outer_scale=1/accum,inner_multiplier=mult)
    if method == 'adam':
        loss = no_inner_chunked(model,x,y,mask,cfg)
    elif method == 'exact':
        fn = loss_for_sequence_meta_remat if remat else loss_for_sequence_meta
        loss, _ = fn(model,x,y,mask,mult,cfg)
    else:
        loss, _ = loss_for_sequence_block(model,x,y,mask,mult,cfg,2,create_graph=True,
                                          parallel_read=True,first_order=True)
    (loss/accum).backward()
    return loss.detach(), {}


def wait_data(data, needed, out):
    started = time.monotonic()
    while True:
        ready = data/'ready.json'
        if ready.exists():
            d = json.loads(ready.read_text())
            if d['committed_tokens'] >= needed and (data/'validation.npy').exists():
                return time.monotonic()-started
        save(out/'status.json',dict(status='waiting_data',needed_tokens=needed,
                                   available_tokens=d['committed_tokens'] if ready.exists() else 0))
        time.sleep(10)


@torch.no_grad()
def validate(model, cfg, data, out, step, batch_size):
    model.eval()
    rows = np.load(data/'validation.npy', mmap_mode='r')
    result = dict(step=step, tokens=step*cfg.training.global_batch_size*cfg.training.seq_length,
                  dataset='content-hash-held-out DCLM long documents', policies={})
    cs = cfg.model.mini_batch_size
    for k in [0,1,2]:
        losses = []
        for i in range(0, len(rows), batch_size):
            batch = torch.as_tensor(np.array(rows[i:i+batch_size],dtype=np.int64),device='cuda')
            state = make_state(model,len(batch),k)
            values = []
            for c in range(8):
                values.append(state.process_real_chunk(batch[:,c*cs:(c+1)*cs],
                              batch[:,c*cs+1:(c+1)*cs+1],float(k>0),1.,cfg).cpu().numpy())
            losses.extend(np.stack(values).mean(0).tolist()); del state
        if not np.isfinite(losses).all():raise RuntimeError('nonfinite validation')
        result['policies'][str(k)] = dict(nll=float(np.mean(losses)),per_document_nll=losses)
    with (out/'validation.jsonl').open('a') as f:f.write(json.dumps(result)+'\n')
    save(out/'validation_latest.json',result)
    print(json.dumps(dict(event='validation',step=step,nll={k:v['nll'] for k,v in result['policies'].items()})),flush=True)
    model.train()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--method',required=True,choices=['adam','exact','fo','shared'])
    ap.add_argument('--data',required=True)
    ap.add_argument('--out',required=True)
    ap.add_argument('--microbatch',type=int,default=2)
    ap.add_argument('--seed',type=int,default=0)
    ap.add_argument('--exact-remat',action='store_true')
    ap.add_argument('--smoke-steps',type=int,default=0)
    ap.add_argument('--pg19-eval-data')
    a = ap.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True);data=Path(a.data)
    if (out/'complete.json').exists():raise RuntimeError('already complete')
    cfg=PRESETS['125m-e2e']()
    t=cfg.training;assert t.total_steps==4800 and t.seq_length==8192 and t.global_batch_size==64
    if t.global_batch_size % a.microbatch:raise ValueError('microbatch must divide 64')
    t.accum_steps=t.global_batch_size//a.microbatch;t.model_seed=a.seed;t.data_seed=0
    wait_data(data,t.global_batch_size*t.seq_length+1,out)
    torch.set_num_threads(4);torch.manual_seed(a.seed)
    # Construct on CPU so CUDA context/GPU assignment cannot alter initialization.
    model=TTTModel(cfg.model,max_seq_len=134144)
    initial_hash=state_digest(model)
    plan=json.loads((data/'plan.json').read_text())
    assert plan['target_tokens'] >= t.total_steps*t.global_batch_size*t.seq_length+1
    root=Path(__file__).resolve().parents[1]
    config=dict(method=a.method,initialization='random Gaussian from model seed; no pretrained weights',
        initialization_sha256=initial_hash,seed=a.seed,official_commit=OFFICIAL_COMMIT,
        preset='125m-e2e',cfg=dataclasses.asdict(cfg),planned_tokens=t.total_steps*t.global_batch_size*t.seq_length,
        microbatch=a.microbatch,accumulation=t.accum_steps,steps=t.total_steps,
        data_plan_sha256=sha(data/'plan.json'),data_order='contiguous 8K windows through seed-shuffled source shards; identical four arms',
        prefix_checkpointing=os.environ.get('TTT_CKPT_PREFIX','0'),exact_remat=a.exact_remat,
        parameters=sum(p.numel() for p in model.parameters()),torch=torch.__version__,smoke_steps=a.smoke_steps,
        differences_from_official=['PyTorch port rather than JAX','public re-tokenized long-document DCLM with independent content-hash holdout',
         'source-shard shuffle rather than Grain full window shuffle','ordinary Adam shares prime architecture, unlike official FA architecture'],
        code_sha256={str(p.relative_to(root)):sha(p) for p in sorted((root/'ttt_pt').glob('*.py'))})
    for p in [Path(__file__),root/'scripts/inner_training_comparison.py']:
        config['code_sha256'][str(p.relative_to(root))]=sha(p)
    if (out/'config.json').exists() and json.loads((out/'config.json').read_text())!=config:
        raise RuntimeError('config or code changed; unsafe resume')
    save(out/'config.json',config)
    torch.empty(1,device='cuda')
    for _ in range(90):
        if torch.cuda.mem_get_info()[0]>110*1024**3:break
        time.sleep(1)
    else:raise RuntimeError('GPU busy; no process killed')
    model=model.cuda().train();o=t.optimizer_outer
    opt=torch.optim.AdamW(model.parameters(),lr=o.lr,betas=(o.b1,o.b2),weight_decay=o.weight_decay,eps=1e-8,fused=True)
    start=0
    if (out/'latest.pt').exists():
        ck=torch.load(out/'latest.pt',map_location='cpu',weights_only=False)
        assert ck['config']==config
        model.load_state_dict(ck['model']);opt.load_state_dict(ck['opt']);start=ck['completed_steps']
        torch.set_rng_state(ck['cpu_rng']);torch.cuda.set_rng_state(ck['cuda_rng']);del ck
    if not a.smoke_steps and start==0 and not (out/'validation_latest.json').exists():
        validate(model,cfg,data,out,0,a.microbatch)
    started=time.monotonic();steps=a.smoke_steps or t.total_steps
    for step in range(start,steps):
        need=(step+1)*t.global_batch_size*t.seq_length+1
        waiting=wait_data(data,need,out)
        stream=np.memmap(data/'train.bin',dtype='<u4',mode='r',shape=(need,))
        for g in opt.param_groups:g['lr']=lr_at(step,o)
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();clock=time.monotonic()
        opt.zero_grad(set_to_none=True);losses=[];norms=[];clips=[]
        for j in range(t.accum_steps):
            idx=step*t.global_batch_size+j*a.microbatch
            batch=np.stack([stream[(idx+b)*t.seq_length:(idx+b+1)*t.seq_length+1] for b in range(a.microbatch)])
            batch=torch.as_tensor(batch.astype(np.int64),device='cuda')
            loss,aux=backward(a.method,model,batch,cfg,step,t.accum_steps,a.exact_remat)
            losses.append(float(loss))
            if aux:norms.append(aux['mean_update_norm']);clips.append(aux['clipping_frequency'])
            del aux,loss,batch
        gnorm=torch.nn.utils.clip_grad_norm_(model.parameters(),o.clip_gradient,error_if_nonfinite=True)
        if not np.isfinite(losses).all():raise RuntimeError('nonfinite training loss')
        opt.step();torch.cuda.synchronize();del stream
        row=dict(step=step+1,loss=float(np.mean(losses)),gnorm=float(gnorm),lr=lr_at(step,o),
                 inner_multiplier=ilr_multiplier(step,cfg),step_s=time.monotonic()-clock,
                 peak_gib=torch.cuda.max_memory_allocated()/1024**3,waiting_data_s=waiting,
                 completed_tokens=(step+1)*t.global_batch_size*t.seq_length)
        if norms:row.update(mean_update_norm=float(np.mean(norms)),clipping_frequency=float(np.mean(clips)))
        with (out/'train.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True);save(out/'status.json',dict(status='training',**row))
        if not a.smoke_steps and ((step+1)%100==0 or step+1==steps):
            torch.save(dict(model=model.state_dict(),opt=opt.state_dict(),completed_steps=step+1,
                            config=config,cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state()),out/'latest.tmp.pt')
            (out/'latest.tmp.pt').replace(out/'latest.pt')
            if step+1 in [400,1200,2400,4800]:
                torch.save(dict(model=model.state_dict(),completed_steps=step+1,config=config),out/f'model_{step+1:05d}.pt')
        if not a.smoke_steps and (step+1)%200==0:validate(model,cfg,data,out,step+1,a.microbatch)
    del opt;model.zero_grad(set_to_none=True);torch.cuda.empty_cache()
    if not a.smoke_steps and a.pg19_eval_data:
        save(out/'status.json',dict(status='evaluating',completed_steps=steps))
        base=Path(a.pg19_eval_data);manifest=json.loads((base/'manifest.json').read_text())
        books=[np.load(base/('test_'+b['book_id']+'.npy'),mmap_mode='r') for b in manifest['books']['test']]
        save(out/'supplementary_pg19_provenance.json',manifest)
        evaluate(model,cfg,books,out,'final_32k',32,a.microbatch)
        evaluate(model,cfg,books,out,'final_128k',128,a.microbatch)
        generated_eval(model,cfg,books,out,a.method,a.seed)
    if not a.smoke_steps:
        while not (data/'manifest.json').exists():time.sleep(10)
        save(out/'dataset_manifest.json',json.loads((data/'manifest.json').read_text()))
    result=dict(status='smoke_complete' if a.smoke_steps else 'complete',method=a.method,
                completed_steps=steps,tokens=steps*t.global_batch_size*t.seq_length,
                elapsed_this_process_s=time.monotonic()-started,initialization_sha256=initial_hash)
    save(out/'complete.json',result);save(out/'status.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
