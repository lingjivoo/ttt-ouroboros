"""Create immutable Qwen-tokenized PG19 streams from the released Llama-token array."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--val", required=True)
    p.add_argument("--source-tokenizer", required=True)
    p.add_argument("--qwen-tokenizer", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--first-book", type=int, default=2)
    p.add_argument("--n-books", type=int, default=16)
    p.add_argument("--qwen-tokens", type=int, default=143361)
    a = p.parse_args()
    from transformers import AutoTokenizer

    src = AutoTokenizer.from_pretrained(a.source_tokenizer, local_files_only=True)
    qtok = AutoTokenizer.from_pretrained(a.qwen_tokenizer, local_files_only=True)
    arr = np.load(a.val, mmap_mode="r")
    starts = np.flatnonzero(arr == 128000)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for book in range(a.first_book, a.first_book + a.n_books):
        start = int(starts[book])
        take = min(len(arr) - start, max(a.qwen_tokens + 32768, 180000))
        while True:
            source = np.asarray(arr[start : start + take], dtype=np.int64)
            text = src.decode(source.tolist(), skip_special_tokens=False)
            ids = np.asarray(
                qtok(text, add_special_tokens=False)["input_ids"], dtype=np.int32
            )
            if len(ids) >= a.qwen_tokens or start + take >= len(arr):
                break
            take = min(len(arr) - start, take + 65536)
        if len(ids) < a.qwen_tokens:
            raise RuntimeError(f"book {book}: only {len(ids)} Qwen tokens")
        ids = ids[: a.qwen_tokens]
        target = out / f"book_{book:03d}.npy"
        tmp = target.with_suffix(".tmp")
        with tmp.open("wb") as f:
            np.save(f, ids)
        os.replace(tmp, target)
        rows.append(
            {
                "book_id": book,
                "source_start": start,
                "source_tokens_decoded": take,
                "qwen_tokens_saved": len(ids),
                "source_token_sha256": hashlib.sha256(source.tobytes()).hexdigest(),
                "qwen_token_sha256": hashlib.sha256(ids.tobytes()).hexdigest(),
                "file": target.name,
            }
        )
        print("prepared", book, len(ids), flush=True)
    manifest = {
        "status": "passed",
        "protocol": "qwen-pg19-retokenize-v1",
        "val": a.val,
        "val_sha256": sha(a.val),
        "source_tokenizer": a.source_tokenizer,
        "qwen_tokenizer": a.qwen_tokenizer,
        "first_book": a.first_book,
        "n_books": a.n_books,
        "qwen_tokens_per_stream": a.qwen_tokens,
        "decode_skip_special_tokens": False,
        "books": rows,
    }
    t = out / "manifest.json.tmp"
    t.write_text(json.dumps(manifest, indent=1) + "\n")
    os.replace(t, out / "manifest.json")


if __name__ == "__main__":
    main()
