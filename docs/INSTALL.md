# Installation

Use Python 3.11–3.12 and a CUDA-matched PyTorch build. The reported language
experiments used bf16 CUDA kernels; CPU execution is intended for unit tests and
analysis only. `requirements.txt` covers language experiments and
`requirements-agents.txt` adds ALFWorld.

Run `python scripts/selfcheck.py --full` on a GPU host after exporting
`TTT_DATA`, `TTT_CKPT`, and `TTT_OUT`. The script verifies imports, readable
artifacts, CUDA, checkpoint loading, and a real forward pass.
