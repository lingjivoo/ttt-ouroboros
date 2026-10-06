<p align="center">
  <img src="assets/ouroboros_icon_handdrawn.png" width="180" alt="Hand-drawn Ouroboros icon">
</p>

<h1 align="center">TTT Ouroboros</h1>

<p align="center"><b>Self-Generated Feedback Destabilizes Test-Time Training:<br>A Causal Decomposition of Long-Horizon Adaptation</b></p>
<p align="center">Cheng Luo · Bing Li · Bernard Ghanem</p>
<p align="center">
  <a href="https://arxiv.org/abs/2610.05076">Paper</a> ·
  <a href="https://arxiv.org/pdf/2610.05076">PDF</a> ·
  <a href="REPRODUCE.md">Reproduction guide</a> ·
  <a href="README.zh-CN.md">中文</a>
</p>

What happens when a model repeatedly learns from its own output during inference?
**Ouroboros** studies this feedback loop in persistent test-time training:
the adapted state changes future text, which becomes the next training signal.

Our experiments separate the effects of generated content, attention context,
and persistent weight updates. We also study **Settlement**: propose an update,
validate its transfer to independent evidence, and only then commit it.
The repository includes the PyTorch TTT runtime, experiment configurations,
analysis scripts, and figure sources.

## Main findings

The paper compares three policies after a shared real-text prefix. **Closed
Loop** retains updates from generated text. **Writes Off** reads generated
text but discards its updates. **Fixed Generation** has a frozen copy generate
the text while a separate learner retains updates. Independent human-written
passages are scored without changing the continuing stream.

| TTT-E2E model | Extra clean-text NLL under Closed Loop vs. Writes Off |
| --- | ---: |
| 125M | +3.01 nats |
| 760M | +6.00 nats |
| 3B | +0.40 nats |

These are first-to-last differences from the canonical six-book, five-seed,
128K-token comparison. Lower NLL is better. The effect has the same direction
at all three scales, but its size is not monotonic in model size.

- **Generation feedback matters.** Fixed Generation removes more than 98% of
  the measured harm at 125M and 760M even though the learner still updates.
- **Reading and writing have distinct costs.** Recorded Replay holds the text
  fixed and measures the extra cost of retaining its update.
- **Fitting the source does not establish transfer.** An update can improve
  prediction of the text that produced it while worsening prediction on new
  real text. Its cost depends on the receiving state.
- **External text changes the exposure.** Frequent real-text passages can
  interrupt the loop; the exposure-density suite uses a separate book set.
- **Settlement checks before commitment.** It tests the proposed state on
  independent evidence before retaining it. Its reported endpoint gaps of
  +0.07 nats at 125M and −0.02 nats at 760M come from separate validation
  suites, so they must not be subtracted from the canonical values above.

The paper also studies Adam updates to Qwen3-4B and an agent setting. The
[reproduction guide](REPRODUCE.md) maps each result to its protocol and code.

## Repository map

| Path | Purpose |
| --- | --- |
| [`ttt_pt/`](ttt_pt/) | PyTorch TTT runtime and stream state |
| [`configs/`](configs/) | Canonical experiment configurations |
| [`scripts/`](scripts/) | Causal controls, replay, and Settlement runners |
| [`analysis/`](analysis/) | Paired summaries and statistical analyses |
| [`figures/`](figures/) | Figure-generation code |
| [`data/unified_perbook_data.json`](data/unified_perbook_data.json) | Bundled per-book data for selected figures |
| [`REPRODUCE.md`](REPRODUCE.md) | Setup, commands, protocol map, and artifact inventory |

This compact release focuses on the main experiments. Large checkpoints,
corpora, and some raw trajectories and result JSONs are external. The
reproduction guide distinguishes figures available from bundled data from
measurements that require a fresh run or additional raw files.

## Installation and quick start

Model runs require a CUDA GPU. The reference environment uses Python 3.11
and the dependencies in `environment.yml`. CPU-only environments can run the
lightweight tests and analyses.

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

Prepare the corpus and checkpoint using the [data instructions](REPRODUCE.md#data-and-checkpoints), then check the
installation and run a small integration test:

```bash
python -m pytest
python scripts/selfcheck.py --profile language --full
python scripts/run_paper_suite.py --suite configs/suites/main_125m.yaml --smoke
```

The smoke run checks checkpoint loading, generation, updates, and clean
probes. It is an integration check, not a paper result. The
[reproduction guide](REPRODUCE.md) also covers pip and agent dependencies.

## Models and data

| Model | Expected checkpoint path | Download |
| --- | --- | --- |
| TTT-E2E 125M, extended to 32K | `$TTT_CKPT/125m-ext32k.pt` | [Dropbox](https://www.dropbox.com/scl/fi/b3hdau2dn5s95kdydycp6/ext-125m-e2e-32k-pt?rlkey=rdyy0wbn4c5zwxj69p1ywp0xi&dl=1) |
| TTT-E2E 760M, extended to 32K | `$TTT_CKPT/760m-ext32k.pt` | [Dropbox](https://www.dropbox.com/scl/fi/13qtne20u54t3x0wbm9h7/ext-760m-e2e-32k-pt?rlkey=q7qmr61uj3u7dr58s68oonr35&dl=1) |

Checkpoints and tokenized corpora are distributed separately from source code.
Save the downloads under the filenames shown above so the released
configurations can locate them. The 3B suite requires a matching 128K
checkpoint, which is not linked here.
Do not substitute a books8k checkpoint for an extended-context checkpoint.
[Artifact availability](REPRODUCE.md) distinguishes bundled data from
external or unavailable raw results.

## Reproduce the experiments

Start with the paired language controls:

```bash
python scripts/run_paper_suite.py --suite configs/suites/main_125m.yaml --dry-run
python scripts/run_paper_suite.py --suite configs/suites/main_125m.yaml
```

The canonical suite runs eight physical rows at logical width eight and
reports the screened books 2–7. It uses seeds 42, 1, 7, 2, and 3; 128 chunks
of 1,024 tokens; a shared 8K real-text prefix; temperature 1 and top-p 0.95;
and 16 read-only clean probes. Analysis averages seeds within each book and
resamples books for intervals. Keep the checkpoint, corpus, book selection,
decoder, probe schedule, and batch width matched across arms.

| Experiment | Where to start |
| --- | --- |
| Closed Loop / Writes Off / Fixed Generation | [`configs/suites/`](configs/suites/) |
| Exposure density and decoding | [Paper reproduction guide](REPRODUCE.md) |
| Single-update transfer / language Settlement | [`scripts/preq_obs.py`](scripts/preq_obs.py) / [`scripts/deferred.py`](scripts/deferred.py) |
| WebShop causal controls and Settlement | [`scripts/ws_arms.py`](scripts/ws_arms.py) |
| Replay and gradient correlation | [Artifact and analysis guide](REPRODUCE.md) |
| Tables and figures | [`analysis/`](analysis/) and [`figures/`](figures/) |

Read the [protocol definitions](REPRODUCE.md) before comparing suites.
Canonical harm and Settlement's within-suite endpoint gap use different
protocols and must not be pooled. The [reproduction guide](REPRODUCE.md)
provides commands, configuration details, and an inventory of which raw
results are available. A runnable script alone does not verify a paper value.

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
