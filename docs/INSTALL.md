# Installation

Use Python 3.11–3.12 and a CUDA-matched PyTorch build. The reported language
experiments used bf16 CUDA kernels; CPU execution is for unit tests and analysis.

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -e '.[language,dev]'
```

For WebShop, install its upstream repository separately, set `WEBSHOP_DIR` and
`WEBSHOP_DATA`, then install this repository's agent extras:

```bash
pip install -e '.[agents]'
```

Run `python scripts/selfcheck.py --full` on a GPU host after exporting
`TTT_DATA`, `TTT_CKPT`, and `TTT_OUT`.
