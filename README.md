<p align="center">
  <img src="assets/ouroboros_icon_handdrawn.png" width="180" alt="Hand-drawn Ouroboros icon">
</p>

<h1 align="center">TTT Ouroboros</h1>

<p align="center"><b>Self-Generated Feedback Destabilizes Test-Time Training:<br>A Causal Decomposition of Long-Horizon Adaptation</b></p>
<p align="center">Cheng Luo · Bing Li · Bernard Ghanem</p>
<p align="center">
  <a href="https://arxiv.org/abs/2610.05076">Paper</a> ·
  <a href="https://arxiv.org/pdf/2610.05076">PDF</a> ·
  <a href="docs/INSTALL.md">Installation</a> ·
  <a href="docs/PAPER_REPRODUCTION.md">Reproduce the paper</a> ·
  <a href="docs/ARTIFACTS.md">Artifacts</a>
</p>

What happens when a model repeatedly learns from its own output during inference?
**Ouroboros** studies this feedback loop in persistent test-time training:
the adapted state changes future text, which becomes the next training signal.

Our experiments separate the effects of generated content, attention context,
and persistent weight updates. We also study **Settlement**: propose an update,
validate its transfer to independent evidence, and only then commit it.
The repository includes the PyTorch TTT runtime, experiment configurations,
analysis scripts, and figure sources.

## Quick start

Use a CUDA GPU for model experiments. CPU-only environments can run the
lightweight tests and analyses; they do not reproduce GPU experiment results.

```bash
git clone https://github.com/lingjivoo/ttt-ouroboros.git
cd ttt-ouroboros
conda env create -f environment.yml
conda activate ttt-ouroboros
pip install -e .

cp env.sh.example env.sh
# Edit TTT_DATA, TTT_CKPT and TTT_OUT in env.sh.
source env.sh
```

Prepare the corpus and checkpoint using [Data](docs/DATA.md), then check the
installation and run a small integration test:

```bash
make check
python scripts/selfcheck.py --profile language --full
python scripts/run_paper_suite.py --suite configs/suites/main_125m.yaml --smoke
```

The smoke run checks execution, not the paper's reported result.
See [Installation](docs/INSTALL.md) for pip, Docker, and agent dependencies.

## Models and data

| Model | Expected checkpoint path | Download |
| --- | --- | --- |
| TTT-E2E 125M, extended to 32K | `$TTT_CKPT/125m-ext32k.pt` | Release pending |
| TTT-E2E 760M, extended to 32K | `$TTT_CKPT/760m-ext32k.pt` | Release pending |

Checkpoints and tokenized corpora are distributed separately from source code.
Download links will be added after checkpoint identity and SHA256 verification.
Do not substitute a books8k checkpoint for an extended-context checkpoint.
[Artifact availability](docs/ARTIFACTS.md) distinguishes bundled data from
external or unavailable raw results.

## Reproduce the experiments

Start with the paired language controls:

```bash
python scripts/run_paper_suite.py --suite configs/suites/main_125m.yaml --dry-run
python scripts/run_paper_suite.py --suite configs/suites/main_125m.yaml
```

| Experiment | Where to start |
| --- | --- |
| Closed Loop / Writes Off / Fixed Generation | [`configs/suites/`](configs/suites/) |
| Exposure density, decoding, and update strength | [Paper reproduction guide](docs/PAPER_REPRODUCTION.md) |
| Language Settlement | [`scripts/preq_obs.py`](scripts/preq_obs.py) |
| WebShop causal controls and Settlement | [`scripts/ws_arms.py`](scripts/ws_arms.py) |
| Replay and gradient correlation | [Artifact and analysis guide](docs/ARTIFACTS.md) |
| Tables and figures | [`analysis/`](analysis/) and [`figures/`](figures/) |

Read the [protocol definitions](docs/EXPERIMENTS.md) before comparing suites.
Canonical harm and Settlement's within-suite endpoint gap use different
protocols and must not be pooled. The [reproduction guide](docs/PAPER_REPRODUCTION.md)
provides commands and configuration details; [validation notes](docs/VALIDATION.md)
record what has actually been checked.

## Acknowledgments

We thank the authors of [End-to-End Test-Time Training](https://github.com/test-time-training/e2e)
for making their research and official JAX implementation available.
Our native TTT experiments build on TTT-E2E; this repository provides a PyTorch
runtime and the feedback interventions studied in our paper.
Third-party code and data retain their respective licenses; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Citation

```bibtex
@article{luo2026selfgenerated,
  title={Self-Generated Feedback Destabilizes Test-Time Training: A Causal Decomposition of Long-Horizon Adaptation},
  author={Luo, Cheng and Li, Bing and Ghanem, Bernard},
  journal={arXiv preprint arXiv:2610.05076},
  year={2026},
  doi={10.48550/arXiv.2610.05076}
}
```

Code is released under the [MIT License](LICENSE).
