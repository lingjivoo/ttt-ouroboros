#!/bin/bash
#SBATCH --job-name=ttt
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00

set -euo pipefail
: "${TTT_REPO:?set TTT_REPO to the checkout}"
: "${TTT_MANIFEST:?set TTT_MANIFEST to a YAML manifest}"
cd "$TTT_REPO"
mkdir -p logs "${TTT_OUT:-results}"
python scripts/run_config.py "$TTT_MANIFEST"
