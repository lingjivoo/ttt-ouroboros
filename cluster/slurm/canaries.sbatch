#!/bin/bash
#SBATCH --job-name=ttt-canaries
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00

set -euo pipefail
: "${TTT_REPO:?set TTT_REPO to the checkout}"
: "${TTT_CKPT:?set TTT_CKPT}"
: "${TTT_DATA:?set TTT_DATA}"
cd "$TTT_REPO"
mkdir -p logs "${TTT_OUT:-results}/canaries"
python scripts/gate_canaries.py --ckpt "$TTT_CKPT/125m-ext32k.pt" \
  --val "$TTT_DATA/pg19/val.npy" --keep "${TTT_OUT:-results}/canaries/gate"
python scripts/state_probe_audit.py --ckpt "$TTT_CKPT/125m-ext32k.pt" \
  --val "$TTT_DATA/pg19/val.npy"
python scripts/stream_parity.py --ckpt "$TTT_CKPT/125m-ext32k.pt" \
  --val "$TTT_DATA/pg19/val.npy"
