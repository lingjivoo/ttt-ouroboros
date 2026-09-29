# Data and checkpoints

The repository does not redistribute corpora or weights. Expected layout:

```text
$TTT_DATA/
  pg19/val.npy
  records/self.pt
  records/degraded_replay.pt
$TTT_CKPT/
  125m-ext32k.pt
  760m-ext32k.pt
  3b_128k_pt.pt
```

`val.npy` is a one-dimensional integer token array. Book boundaries are encoded
by the protocol's BOS token and are selected by `find_books`; do not replace
book identities with row numbers alone. Record the byte range or stable book
ID, tokenization version, selection threshold, and SHA256 in every public
artifact manifest.

Checkpoints must be loaded with the matching preset in `ttt_pt/config.py`.
Never infer a preset from parameter count. The 3B 8K and 128K checkpoints are
distinct experimental objects.

Place an optional `<artifact>.sha256` sidecar beside large data and checkpoint
files when hashing them at every launch is too expensive. Its first whitespace
separated field must be the lowercase SHA256 digest. Otherwise the runners hash
the artifact directly and store the digest in `_config`.

The canonical 125M protocol loads physical books 0--7 at logical width eight
and reports books 2--7. This distinction is encoded in the manifests; a result
that ran only six physical rows is not canonical even if it labels them 2--7.

For Qwen3-4B experiments, retokenize immutable PG-19 streams and retain the
resulting `manifest.json`:

```bash
python scripts/qwen_prepare_pg19.py \
  --val "$TTT_DATA/pg19/val.npy" \
  --source-tokenizer "$TTT_SOURCE_TOKENIZER" \
  --qwen-tokenizer "$TTT_QWEN_MODEL" \
  --out "$TTT_DATA/qwen_pg19"
```

WebShop expects the upstream repository in `WEBSHOP_DIR` and the deterministic
catalogue made by `scripts/ws_build_catalogue.py` in `WEBSHOP_DATA`.
