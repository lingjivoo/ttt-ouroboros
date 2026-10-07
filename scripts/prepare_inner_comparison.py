"""Small official PG-19 train/test split with explicit book provenance."""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path

import numpy as np
import requests
from transformers import AutoTokenizer


def fetch(url):
    r = requests.get(url, timeout=90)
    r.raise_for_status()
    return r.content


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--train-tokens', type=int, default=10_000_000)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    repo = 'NousResearch/Meta-Llama-3-8B'
    revision = json.loads(fetch('https://huggingface.co/api/models/' + repo))['sha']
    tok = AutoTokenizer.from_pretrained(repo, revision=revision)
    assert tok.bos_token_id == 128000 and tok.eos_token_id == 128001
    manifests = {}
    for split in ('train', 'test'):
        listing = fetch('https://huggingface.co/datasets/deepmind/pg19/raw/main/data/' + split + '_files.txt')
        files = sorted(listing.decode().splitlines())
        np.random.default_rng(20261007).shuffle(files)
        records, arrays, count = [], [], 0
        for start in range(0, len(files), 16):
            names = files[start:start + 16]
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
                texts = list(ex.map(lambda x: fetch('https://storage.googleapis.com/deepmind-gutenberg/' + x), names))
            for name, raw in zip(names, texts):
                ids = np.asarray([128000] + tok.encode(raw.decode('utf-8'), add_special_tokens=False) + [128001], dtype=np.uint32)
                if split == 'test' and len(ids) < 133_121:
                    continue
                record = dict(book_id=Path(name).stem, source=name, tokens=len(ids),
                              text_sha256=hashlib.sha256(raw).hexdigest(),
                              tokens_sha256=hashlib.sha256(ids.tobytes()).hexdigest(), start=count)
                records.append(record)
                if split == 'train':
                    arrays.append(ids)
                else:
                    np.save(out / ('test_' + record['book_id'] + '.npy'), ids)
                count += len(ids)
                print(split, len(records), count, flush=True)
                if (split == 'train' and count >= args.train_tokens) or (split == 'test' and len(records) == 4):
                    break
            if (split == 'train' and count >= args.train_tokens) or (split == 'test' and len(records) == 4):
                break
        if split == 'train':
            np.save(out / 'train.npy', np.concatenate(arrays))
        else:
            assert len(records) == 4
        manifests[split] = records
    assert not ({r['book_id'] for r in manifests['train']} & {r['book_id'] for r in manifests['test']})
    manifest = dict(tokenizer=repo, tokenizer_revision=revision, shuffle_seed=20261007,
                    train_file_sha256=hashlib.sha256((out / 'train.npy').read_bytes()).hexdigest(), books=manifests,
                    note='Official split separation for this continuation; prior checkpoint pretraining overlap not audited.')
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print('DATA_READY', flush=True)


if __name__ == '__main__':
    main()
