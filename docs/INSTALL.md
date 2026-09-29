# Installation

The repository supports Conda, pip, and Docker. Conda is the simplest route on
a managed GPU host because it pins Python and the CUDA runtime together.

## Conda

```bash
conda env create -f environment.yml
conda activate ttt-feedback
cp env.sh.example env.sh
# Edit the three TTT_* paths, then:
source env.sh
python scripts/selfcheck.py --profile language
```

## Pip

Use Python 3.11–3.12 and a CUDA-matched PyTorch 2.x build. The reported language
experiments used bf16 CUDA kernels; CPU execution is for unit tests and analysis.

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -e '.[language,dev]'
```

Copy `.env.example` or `env.sh.example` and set `TTT_DATA`, `TTT_CKPT`, and
`TTT_OUT`. Validate a real checkpoint before launching a long run:

```bash
python scripts/selfcheck.py --profile language --full
python scripts/run_paper_suite.py configs/suites/main_125m.yaml --smoke
```

For WebShop, install its upstream repository separately, set `WEBSHOP_DIR` and
`WEBSHOP_DATA`, then install this repository's agent extras:

```bash
pip install -e '.[agents]'
```

Run `python scripts/selfcheck.py --full` on a GPU host after exporting
`TTT_DATA`, `TTT_CKPT`, and `TTT_OUT`.

## Docker

Build arguments let the image select a CUDA-compatible PyTorch wheel:

```bash
docker build -t ttt-feedback .
docker run --rm --gpus all \
  -v "$TTT_DATA:/data:ro" -v "$TTT_CKPT:/checkpoints:ro" -v "$TTT_OUT:/outputs" \
  -e TTT_DATA=/data -e TTT_CKPT=/checkpoints -e TTT_OUT=/outputs \
  ttt-feedback --suite configs/suites/main_125m.yaml --smoke
```
