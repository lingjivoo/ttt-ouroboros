"""32K mechanism pilot with explicit generator state and boundary policy."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np
import torch

from horizon import CS, WARMUP, find_books, drift
from ttt_pt.config import PRESETS
from ttt_pt.model import COMPUTE_DTYPE, TTTModel
from ttt_pt.stream import StreamState


def sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(8<<20),b""): h.update(b)
    return h.hexdigest()


def save(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+".tmp");tmp.write_text(json.dumps(obj,indent=1)+"\n");os.replace(tmp,path)


def diversity(row):
    x=row.tolist(); n=len(x)
    d2=len(set(zip(x[:-1],x[1:])))/max(1,n-1)
    d3=len(set(zip(x[:-2],x[1:-1],x[2:])))/max(1,n-2)
    seen=set(); repeats=0
    for i in range(max(0,n-3)):
        g=tuple(x[i:i+4]); repeats += g in seen; seen.add(g)
    longest=1; run=1
    for a,b in zip(x,x[1:]):
        run=run+1 if a==b else 1; longest=max(longest,run)
    return {"distinct2":d2,"distinct3":d3,
            "repeated4_fraction":repeats/max(1,n-3),
            "longest_identical_token_run":longest}


def tensors(obj):
    if torch.is_tensor(obj):
        yield obj
    elif isinstance(obj, dict):
        for value in obj.values(): yield from tensors(value)
    elif isinstance(obj, (list, tuple)):
        for value in obj: yield from tensors(value)


def state_digest(snap):
    h=hashlib.sha256(); nbytes=0
    for tensor in tensors(snap):
        x=tensor.detach().contiguous().view(torch.uint8).cpu().numpy()
        h.update(x.tobytes());nbytes+=x.nbytes
    for name in ("chunk_id","global_pos"):
        h.update(f"{name}={snap[name]}".encode())
    return h.hexdigest(),nbytes


def probe(st,inp,tgt,cfg):
    t0=time.monotonic();snap=st.snapshot();before,nbytes=state_digest(snap);holder=[]
    st.process_real_chunk(inp,tgt,0,1,cfg,logit_hook=lambda z,y:holder.append(z.detach()))
    nll=st.last_token_nll.detach(); z=holder[0]; p=z.softmax(-1)
    ent=-(p*p.clamp_min(1e-30).log()).sum(-1).mean(-1)
    st.restore(snap);after,_=state_digest(st.snapshot())
    text=hashlib.sha256(torch.cat([inp,tgt[:,-1:]],1).cpu().numpy().tobytes()).hexdigest()
    result={"nll_book":nll.mean(-1).cpu().tolist(),"entropy_book":ent.cpu().tolist(),
            "real_text_sha256":text,"state_hash_before":before,"state_hash_after":after,
            "state_restored":before==after,"snapshot_bytes":nbytes,
            "probe_seconds":time.monotonic()-t0}
    return result


def displacement(st):
    absolute=[];relative=[]
    for current,initial in zip(st.fast,st.fast_init):
        delta=(current-initial).float().flatten(1)
        base=initial.float().flatten(1)
        dn=delta.pow(2).sum(-1).sqrt();bn=base.pow(2).sum(-1).sqrt()
        absolute.append(dn.cpu().tolist())
        relative.append((dn/bn.clamp_min(1e-30)).cpu().tolist())
    return {"absolute_per_layer_book":absolute,
            "relative_per_layer_book":relative,
            "relative_global":drift(st)}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--ckpt",required=True);ap.add_argument("--val",required=True)
    ap.add_argument("--preset",default="125m-e2e-ext32k")
    ap.add_argument("--out",required=True);ap.add_argument("--mode",required=True,
        choices=["closed","masked","fixed_shared","fixed_w0","fixed_post",
                 "replay_record","replay","replay_off","real_on","real_off"])
    ap.add_argument("--seed",type=int,required=True);ap.add_argument("--book-offset",type=int,default=16)
    ap.add_argument("--n-seqs",type=int,default=2);ap.add_argument("--n-chunks",type=int,default=32)
    ap.add_argument("--boundary",choices=["historical","continuation"],default="historical")
    ap.add_argument("--record-to",default="");ap.add_argument("--replay-from",default="")
    ap.add_argument("--temperature",type=float,default=1.0)
    ap.add_argument("--top-p",type=float,default=.95)
    ap.add_argument("--snapshot-dir",default="")
    ap.add_argument("--book-indices",default="")
    ap.add_argument("--write-multiplier",type=float,default=1.0)
    ap.add_argument("--repetitive-prefix-period",type=int,default=0)
    args=ap.parse_args();t0=time.monotonic()
    result={"status":"running","chunks":[],"probes":[]}
    try:
        cfg=PRESETS[args.preset](); model=TTTModel(cfg.model,max_seq_len=(args.n_chunks+1)*CS).cuda().eval()
        payload=torch.load(args.ckpt,map_location="cpu",weights_only=False)
        model.load_state_dict(payload.get("model",payload),strict=True);del payload
        data=np.load(args.val,mmap_mode="r")
        n_real=(args.n_chunks+2 if args.mode.startswith("real") else
                WARMUP+(args.n_chunks-WARMUP)//8+2);need=n_real*CS+1
        if args.book_indices:
            book_indices=[int(x) for x in args.book_indices.split(",")]
            assert len(book_indices)==args.n_seqs
            common=find_books(data,(args.n_chunks+2)*CS+1)
            books=[common[i] for i in book_indices]
        else:
            book_indices=list(range(args.book_offset,args.book_offset+args.n_seqs))
            books=find_books(data,need)[args.book_offset:args.book_offset+args.n_seqs]
        assert len(books)==args.n_seqs
        real=torch.from_numpy(np.stack([np.asarray(data[s:s+need]).astype(np.int64) for s,_ in books]))
        if args.repetitive_prefix_period:
            period=args.repetitive_prefix_period;assert 1<=period<=CS
            pattern=real[:,:period].clone()
            real[:,:WARMUP*CS]=pattern.repeat(1,(WARMUP*CS+period-1)//period)[:,:WARMUP*CS]
        recv=StreamState(model,args.n_seqs,"cuda"); generator=None
        if args.mode=="fixed_w0": generator=StreamState(model,args.n_seqs,"cuda")
        real_pos=0; previous=None; recording=[]; replay=None
        if args.mode in ("replay","replay_off"):
            replay=torch.load(args.replay_from,weights_only=False)
            assert replay["manifest"]["n_seqs"]==args.n_seqs and replay["manifest"]["boundary"]==args.boundary
        manifest={"config_id":f"pilot-{args.mode}-{args.boundary}-s{args.seed}",
            "mode":args.mode,"seed":args.seed,"receiver_book_indices":book_indices,
            "receiver_book_bounds":books,"source_stream":args.replay_from or None,
            "source_book_indices":(replay["manifest"].get("book_indices") if replay else None),
            "source_book_bounds":(replay["manifest"].get("books") if replay else None),
            "batch_width":args.n_seqs,"parameter_count":sum(p.numel() for p in model.parameters()),
            "parameter_dtype":str(next(model.parameters()).dtype),"compute_dtype":str(COMPUTE_DTYPE),
            "tokenizer":"checkpoint token ids; BOS=128000",
            "n_chunks":args.n_chunks,"chunk_size":CS,"temperature":args.temperature,
            "top_p":args.top_p,"penalties":None,
            "learning_rate":cfg.training.optimizer_inner.lr,"clip":cfg.training.optimizer_inner.clip_gradient,
            "write_multiplier":args.write_multiplier,
            "preset":args.preset,"repetitive_prefix_period":args.repetitive_prefix_period,
            "loss_normalization":"valid token mean per row, then summed gradient","retention":1.0,
            "generator_cache_policy":args.mode,"boundary_token_policy":args.boundary,"probe_policy":"branched snapshot/restore",
            "checkpoint_sha256":sha(args.ckpt),"val_sha256":sha(args.val),
            "source_sha256":{"pilot.py":sha(__file__),"stream.py":sha(Path(__file__).parents[1]/"ttt_pt/stream.py")}}
        row_base=[(s*1000003+args.seed*7919)%(2**63-1) for s,_ in books]
        generation_seconds=0.0;generated_tokens=0
        for c in range(args.n_chunks):
            is_probe=c>=WARMUP and (c-WARMUP)%8==7
            use_real=args.mode.startswith("real") or c<WARMUP or is_probe
            if use_real:
                inp=real[:,real_pos:real_pos+CS].cuda();tgt=real[:,real_pos+1:real_pos+CS+1].cuda()
                if is_probe:
                    q=probe(recv,inp,tgt,cfg);assert q["state_restored"]
                    q.update(schedule_index=c,generated_index=len(result["chunks"]),
                             live_token_count=recv.global_pos,probe_role="trajectory")
                    q.update(displacement=displacement(recv),update=recv.last_update_stats)
                    result["probes"].append(q)
                else:
                    real_write=0 if args.mode=="real_off" else args.write_multiplier
                    recv.process_real_chunk(inp,tgt,real_write,1,cfg)
                    if generator is not None: generator.process_real_chunk(inp,tgt,0 if args.mode=="fixed_w0" else 1,1,cfg)
                real_pos+=CS
                if c==WARMUP-1:
                    pin=real[:,real_pos:real_pos+CS].cuda();ptgt=real[:,real_pos+1:real_pos+CS+1].cuda()
                    q=probe(recv,pin,ptgt,cfg);assert q["state_restored"]
                    q.update(schedule_index=c,generated_index=0,live_token_count=recv.global_pos,
                             probe_role="post_prefill_baseline")
                    q.update(displacement=displacement(recv),update=recv.last_update_stats)
                    result["probes"].append(q)
                continue
            if args.mode=="fixed_post" and generator is None:
                generator=StreamState(model,args.n_seqs,"cuda");generator.restore(recv.snapshot())
            historical=real[:,real_pos:real_pos+1].cuda()
            first=historical if args.boundary=="historical" or previous is None else previous
            generated_ordinal=len(result["chunks"])+1; stats={"raw":[],"sampled":[]}
            if args.snapshot_dir and generated_ordinal in (1,33,65,97):
                folder=Path(args.snapshot_dir);folder.mkdir(parents=True,exist_ok=True)
                target=folder/f"g{generated_ordinal:03d}.pt";tmp=target.with_suffix(".pt.tmp")
                torch.save({"manifest":manifest,"generated_ordinal":generated_ordinal,
                            "schedule_index":c,"real_pos":real_pos,"first_input":first.cpu(),
                            "previous":None if previous is None else previous.cpu(),
                            "receiver":recv.snapshot(),
                            "generator":None if generator is None else generator.snapshot()},tmp)
                os.replace(tmp,target)
            def observer(i,z,p,keep,sp,si,nxt):
                stats["raw"].append((-(p*p.clamp_min(1e-30).log()).sum(-1)).cpu())
                stats["sampled"].append((-(sp*sp.clamp_min(1e-30).log()).sum(-1)).cpu())
            row_seeds=[x+c*65537 for x in row_base]
            gen_t0=time.monotonic()
            if args.mode in ("closed","masked","fixed_shared","replay_record"):
                pv=0 if args.mode=="masked" else args.write_multiplier
                gf=recv.fast_init if args.mode=="fixed_shared" else None
                gen=recv.generate_chunk(first,pv,1,cfg,temperature=args.temperature,
                                        top_p=args.top_p,seed=args.seed*100000+c,
                                        gen_fast=gf,row_seeds=row_seeds,observer=observer)
            elif args.mode in ("fixed_w0","fixed_post"):
                gen=generator.generate_chunk(first,0,1,cfg,temperature=args.temperature,
                                             top_p=args.top_p,seed=args.seed*100000+c,
                                             row_seeds=row_seeds,observer=observer)
                recv.process_real_chunk(torch.cat([first,gen[:,:-1]],1),gen,args.write_multiplier,1,cfg)
            else:
                rc,rfirst,rgen=replay["stream"][generated_ordinal-1];assert rc==c
                first=rfirst.cuda();gen=rgen.cuda()
                replay_write=0 if args.mode=="replay_off" else 1
                recv.process_real_chunk(torch.cat([first,gen[:,:-1]],1),gen,replay_write,1,cfg)
            gen_seconds=time.monotonic()-gen_t0;generation_seconds+=gen_seconds
            generated_tokens+=gen.numel()
            previous=gen[:,-1:]
            row=[]
            for b in range(args.n_seqs):
                v=diversity(gen[b].cpu())
                if stats["raw"]:
                    v.update(entropy_raw=float(torch.stack(stats["raw"])[:,b].mean()),
                             entropy_sampled=float(torch.stack(stats["sampled"])[:,b].mean()))
                else:v.update(entropy_raw=None,entropy_sampled=None)
                row.append(v)
            chunk={"generated_index":generated_ordinal,"schedule_index":c,"live_token_count":recv.global_pos,
                   "first_input":first.cpu().flatten().tolist(),"tokens":gen.cpu().tolist(),
                   "metrics_book":row,"inner_loss":recv.last_nll,"update":recv.last_update_stats,
                   "drift":drift(recv),"seconds":gen_seconds,
                   "displacement":displacement(recv),
                   "tokens_per_second":gen.numel()/max(gen_seconds,1e-9)}
            result["chunks"].append(chunk)
            if args.mode=="replay_record": recording.append((c,first.cpu(),gen.cpu()))
            save(args.out,dict(result,manifest=manifest))
        if args.record_to:
            Path(args.record_to).parent.mkdir(parents=True,exist_ok=True)
            torch.save({"manifest":{"seed":args.seed,"n_seqs":args.n_seqs,"book_offset":args.book_offset,
                        "book_indices":book_indices,
                        "books":books,"boundary":args.boundary,"checkpoint_sha256":manifest["checkpoint_sha256"]},
                        "stream":recording},args.record_to)
        result.update(status="passed",manifest=manifest,seconds=time.monotonic()-t0,
                      generation_seconds=generation_seconds,
                      generated_tokens_per_second=generated_tokens/max(generation_seconds,1e-9),
                      peak_reserved_gb=torch.cuda.max_memory_reserved()/2**30)
        save(args.out,result)
    except Exception:
        result.update(status="failed",error=traceback.format_exc(),seconds=time.monotonic()-t0);save(args.out,result);raise


if __name__=="__main__":main()
