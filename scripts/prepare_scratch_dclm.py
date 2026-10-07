"""Prepare a pinned, long-document DCLM stream with resumable atomic commits.

Public DCLM is not the official TTT-E2E pretokenized bucket. Randomized source
shards, >=8192-token documents, Llama-3 BOS/EOS, content-hash validation split.
Training can consume only the prefix advertised in ready.json; never pad/repeat.
"""
import argparse
import concurrent.futures as cf
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import requests


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def atomic(path, obj):
    path = Path(path)
    t = path.with_suffix('.tmp')
    t.write_text(json.dumps(obj, indent=2) + '\n')
    t.replace(path)


def worker(job):
    plan, index, directory = job
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    out = Path(directory) / 'parts' / f'{index:05d}'
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'complete.json').exists():
        return str(out)
    os.environ['RAYON_NUM_THREADS'] = str(plan['tokenizer_threads'])
    tok = AutoTokenizer.from_pretrained(plan['tokenizer'], revision=plan['tokenizer_revision'])
    assert tok.bos_token_id == 128000 and tok.eos_token_id == 128001
    source = plan['files'][index]
    path = hf_hub_download(plan['dataset'], source, repo_type='dataset', revision=plan['dataset_revision'])
    n = count = scanned = 0
    start = time.time()
    with (out / 'tokens.bin').open('wb') as binary, (out / 'documents.jsonl').open('w') as ledger:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=64, columns=['text']):
            texts = batch.column(0).to_pylist()
            enc = tok(texts, add_special_tokens=False)['input_ids']
            for text, ids in zip(texts, enc):
                scanned += 1
                if len(ids) < plan['min_document_tokens']:
                    continue
                key = hashlib.sha256(text.encode('utf-8')).hexdigest()
                arr = np.asarray([128000] + ids + [128001], dtype='<u4')
                binary.write(arr.tobytes())
                ledger.write(json.dumps(dict(sha256=key, offset=n, tokens=len(arr),
                                             validation=int(key[:16], 16) % 100 == 0)) + '\n')
                n += len(arr)
                count += 1
    atomic(out / 'complete.json', dict(source=source, index=index, documents=count,
           scanned=scanned, tokens=n, seconds=time.time()-start, source_sha256=digest(path)))
    return str(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--tokenizer-threads', type=int, default=8)
    ap.add_argument('--tokens', type=int, default=4800*64*8192+1)
    ap.add_argument('--tokenizer-revision', required=True)
    a = ap.parse_args()
    root = Path(a.out); root.mkdir(parents=True, exist_ok=True)
    if (root / 'manifest.json').exists():
        print('DATA_ALREADY_COMPLETE', flush=True); return
    if (root / 'plan.json').exists():
        plan = json.loads((root / 'plan.json').read_text())
        assert plan['target_tokens'] == a.tokens and plan['tokenizer_revision'] == a.tokenizer_revision
    else:
        repo = 'mlfoundations/dclm-baseline-1.0-parquet'
        response = requests.get('https://huggingface.co/api/datasets/' + repo, timeout=60)
        response.raise_for_status(); info = response.json()
        files = sorted(x['rfilename'] for x in info['siblings'] if x['rfilename'].endswith('.parquet'))
        np.random.default_rng(0).shuffle(files)
        plan = dict(dataset=repo, dataset_revision=info['sha'], files=files,
                    tokenizer='NousResearch/Meta-Llama-3-8B', tokenizer_revision=a.tokenizer_revision,
                    target_tokens=a.tokens, min_document_tokens=8192, source_shuffle_seed=0,
                    tokenizer_threads=a.tokenizer_threads, validation_documents=64,
                    order='seed-0 shuffled parquet shards; original document order within each shard',
                    split='SHA256(text) first 64 bits mod 100 == 0 is held out; dedup exact text',
                    difference_from_official='Public raw DCLM re-tokenized; not identical official bucket or Grain shuffle')
        atomic(root / 'plan.json', plan)
    state = dict(committed_tokens=0, completed_shards=0, ledger_bytes=0, validation=[])
    if (root / 'ready.json').exists():
        state = json.loads((root / 'ready.json').read_text())
    for name, size in [('train.bin', state['committed_tokens']*4), ('accepted.jsonl', state['ledger_bytes'])]:
        path = root / name
        if not path.exists():path.touch()
        with path.open('r+b') as f:f.truncate(size)
    seen = {json.loads(line)['sha256'] for line in (root / 'accepted.jsonl').read_text().splitlines()}
    val = [np.load(root / f'validation_{i:03d}.npy') for i in range(len(state['validation']))]
    jobs = iter(range(state['completed_shards'], len(plan['files'])))
    pool = cf.ProcessPoolExecutor(max_workers=a.workers)
    futures = {}
    try:
        for _ in range(a.workers):
            i = next(jobs); futures[i] = pool.submit(worker, (plan, i, str(root)))
        while futures:
            i = min(futures)
            part = Path(futures.pop(i).result())
            meta = json.loads((part / 'complete.json').read_text())
            with (part / 'tokens.bin').open('rb') as src, (root / 'train.bin').open('ab') as dst, (root / 'accepted.jsonl').open('a') as ledger:
                for line in (part / 'documents.jsonl').read_text().splitlines():
                    d = json.loads(line)
                    if d['sha256'] in seen:continue
                    seen.add(d['sha256']); ledger.write(line+'\n')
                    src.seek(d['offset']*4)
                    if d['validation']:
                        if len(val) < plan['validation_documents']:
                            arr = np.frombuffer(src.read(8193*4), dtype='<u4').copy()
                            assert len(arr) == 8193
                            np.save(root/f'validation_{len(val):03d}.npy', arr)
                            val.append(arr); state['validation'].append(dict(sha256=d['sha256'], source=meta['source']))
                    elif state['committed_tokens'] < a.tokens:
                        take = min(d['tokens'], a.tokens-state['committed_tokens'])
                        raw = src.read(take*4); assert len(raw) == take*4
                        dst.write(raw); state['committed_tokens'] += take
                dst.flush(); os.fsync(dst.fileno()); ledger.flush(); os.fsync(ledger.fileno())
            state.update(completed_shards=i+1, ledger_bytes=(root/'accepted.jsonl').stat().st_size,
                         updated_at=time.time(), last_shard=meta,
                         status='ready' if len(val)==plan['validation_documents'] else 'preparing_validation')
            if len(val) == plan['validation_documents'] and not (root/'validation.npy').exists():
                np.save(root/'validation.tmp.npy', np.stack(val)); (root/'validation.tmp.npy').replace(root/'validation.npy')
            atomic(root/'ready.json', state); print(json.dumps(state | {'validation':len(val)}), flush=True)
            if state['committed_tokens'] == a.tokens and len(val) == plan['validation_documents']:
                atomic(root/'manifest.json', dict(plan_sha256=digest(root/'plan.json'),
                    tokens=state['committed_tokens'], train_sha256=digest(root/'train.bin'),
                    validation_sha256=digest(root/'validation.npy'), accepted_sha256=digest(root/'accepted.jsonl'),
                    source_shards=state['completed_shards'], validation=state['validation'], status='complete'))
                print('DATA_READY_COMPLETE', flush=True); break
            j = next(jobs, None)
            if j is not None:futures[j] = pool.submit(worker, (plan,j,str(root)))
    finally:
        for f in futures.values():f.cancel()
        pool.shutdown(wait=True, cancel_futures=True)


if __name__ == '__main__':main()
