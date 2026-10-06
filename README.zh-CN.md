<p align="center">
  <img src="assets/ouroboros_icon_handdrawn.png" width="180" alt="手绘衔尾蛇图标">
</p>

<h1 align="center">TTT Ouroboros</h1>

<p align="center"><strong>Self-Generated Feedback Destabilizes Test-Time Training:<br>A Causal Decomposition of Long-Horizon Adaptation</strong></p>
<p align="center">Cheng Luo · Bing Li · Bernard Ghanem</p>
<p align="center">
  <a href="README.md">English</a> ·
  <a href="https://arxiv.org/abs/2610.05076">论文</a> ·
  <a href="https://arxiv.org/pdf/2610.05076">PDF</a> ·
  <a href="REPRODUCE.md">复现指南（英文）</a>
</p>

这是[论文](https://arxiv.org/abs/2610.05076)的配套代码。我们研究持续进行测试时训练（TTT）的模型：它生成一段文本，从这段文本学习，再用更新后的状态生成下一段训练文本。这个循环什么时候会损害对独立真实文本的预测？损害发生在生成、注意力读取还是持久权重写入？候选更新能否先验证、再提交？

## 论文的主要发现

主实验从相同的真实文本前缀开始，比较三种策略：

| 策略 | 后续生成 | 对生成文本的更新 |
| --- | --- | --- |
| Closed Loop | 当前已适应模型 | 保留 |
| Writes Off | 当前模型 | 丢弃；仍在注意力中读取生成文本 |
| Fixed Generation | 冻结的初始模型 | 由另一个学习器保留 |

每隔一段时间，实验在独立的人类文本上计算 NLL，评估完成后恢复继续运行的状态。NLL 越低越好。在共同六本书、五个种子、128K token 的 canonical 实验中，Closed Loop 相比 Writes Off 的额外首末损害为：

| TTT-E2E 模型 | 额外真实文本 NLL |
| --- | ---: |
| 125M | +3.01 nats |
| 760M | +6.00 nats |
| 3B | +0.40 nats |

三个规模的方向一致，但损害幅度并不随模型规模单调变化。

- **生成反馈是关键路径。** Fixed Generation 在 125M 和 760M 上消除了超过 98% 的测得损害；学习器仍然写入冻结模型生成的文本。
- **读取与写入分别产生代价。** Recorded Replay 固定文本，区分注意力中读取退化文本的影响和将其更新保留到权重中的额外影响。
- **拟合当前文本不代表能迁移。** 一次更新可以改善来源文本预测，同时损害后续独立真实文本的预测；损害还取决于接收更新时的模型状态。
- **外部文本改变暴露程度。** 频繁插入真实文本可以打断循环。Exposure-density 实验使用单独的书集，应在其自身条件内比较。
- **Settlement 在持久提交前检验候选状态。** 它用独立证据比较更新前后的状态。论文报告的 125M +0.07 和 760M −0.02 nats 是另外两套验证实验的 endpoint gap，不能从上表的 canonical 数值直接相减。

论文还研究了 Qwen3-4B 上的 Adam 更新和 agent 环境。各图表对应的运行脚本、统计单位和已知证据范围见[复现指南](REPRODUCE.md)。

## 仓库内容

| 路径 | 内容 |
| --- | --- |
| [`ttt_pt/`](ttt_pt/) | PyTorch TTT 运行时与流状态 |
| [`configs/`](configs/) | 主要对照实验的配置 |
| [`scripts/`](scripts/) | 因果对照、replay 和 Settlement 实验脚本 |
| [`analysis/`](analysis/) | 配对统计和结果分析 |
| [`figures/`](figures/) | 论文绘图代码 |
| [`data/unified_perbook_data.json`](data/unified_perbook_data.json) | 部分图使用的逐书数据 |
| [`REPRODUCE.md`](REPRODUCE.md) | 安装、实验命令、协议与 artifact 清单 |

仓库集中保留论文主线实验。大型 checkpoint、语料以及部分原始轨迹和结果 JSON 单独存放。复现指南标明哪些图可以从仓库现有数据生成，哪些结果需要重新运行或取得额外原始文件。**有脚本并不等于已经核验论文中的数字。**

## 安装与快速检查

模型实验需要 CUDA GPU。参考环境使用 Python 3.11 和 `environment.yml`；没有 GPU 时仍可运行轻量测试及分析。

```bash
git clone https://github.com/lingjivoo/ttt-ouroboros.git
cd ttt-ouroboros
conda env create -f environment.yml
conda activate ttt-ouroboros
pip install -e .

cp env.sh.example env.sh
# 在 env.sh 中设置 TTT_DATA、TTT_CKPT、TTT_OUT 的绝对路径。
source env.sh
```

按[数据说明](REPRODUCE.md#data-and-checkpoints)准备 PG-19 与 checkpoint，再运行：

```bash
python -m pytest
python scripts/selfcheck.py --profile language --full
python scripts/run_paper_suite.py --suite configs/suites/main_125m.yaml --smoke
```

Smoke test 检查权重加载、生成、更新和 clean probe 的执行链路，**不代表论文正式实验结果**。pip 安装和 agent 依赖见[复现指南](REPRODUCE.md#installation)。

## 模型权重

权重不放在 Git 仓库中。下载后请按表中的文件名保存，保证配置能找到它们；Dropbox 原始下载名可能不同。

| 模型 | 保存为 | 下载 |
| --- | --- | --- |
| TTT-E2E 125M，32K 扩展 | `$TTT_CKPT/125m-ext32k.pt` | [Dropbox](https://www.dropbox.com/scl/fi/b3hdau2dn5s95kdydycp6/ext-125m-e2e-32k-pt?rlkey=rdyy0wbn4c5zwxj69p1ywp0xi&dl=1) |
| TTT-E2E 760M，32K 扩展 | `$TTT_CKPT/760m-ext32k.pt` | [Dropbox](https://www.dropbox.com/scl/fi/13qtne20u54t3x0wbm9h7/ext-760m-e2e-32k-pt?rlkey=q7qmr61uj3u7dr58s68oonr35&dl=1) |

3B 实验需要与配置相符的 128K checkpoint，此处尚未提供下载链接。不要用 books8k checkpoint 代替扩展上下文权重。125M 下载后可运行 `python scripts/selfcheck.py --profile language --full` 验证加载。

## 运行主要对照实验

先查看 125M 三个对照条件展开后的命令，再正式运行：

```bash
python scripts/run_paper_suite.py --suite configs/suites/main_125m.yaml --dry-run
python scripts/run_paper_suite.py --suite configs/suites/main_125m.yaml
```

Canonical 协议以逻辑宽度 8 运行八个物理行，报告筛选后的 books 2–7。使用 seeds 42、1、7、2、3；128 个 1,024-token chunks；共同的 8K 真实文本前缀；temperature 1、top-p 0.95；以及 16 次只读 clean probes。统计时先在每本书内平均种子，再以书为单位估计区间。各条件应使用相同 checkpoint、语料、书集选择、decoder、probe 日程和 batch width。

760M 与 3B 的对应配置位于 [`configs/suites/`](configs/suites/)。Replay、exposure-density、Settlement 和 WebShop 使用各自的协议；请从[复现指南](REPRODUCE.md)进入，不要将不同实验套件的绝对数值混合。

## 致谢与引用

感谢 [End-to-End Test-Time Training](https://github.com/test-time-training/e2e) 的作者公开研究成果和官方 JAX 实现。本仓库提供 PyTorch 运行时及论文中的反馈路径实验。第三方代码、数据与权重遵循各自的使用条款，详见 [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)。

```bibtex
@article{luo2026selfgenerated,
  title={Self-Generated Feedback Destabilizes Test-Time Training: A Causal Decomposition of Long-Horizon Adaptation},
  author={Luo, Cheng and Li, Bing and Ghanem, Bernard},
  journal={arXiv preprint arXiv:2610.05076},
  year={2026},
  doi={10.48550/arXiv.2610.05076}
}
```

仓库原创代码采用 [MIT License](LICENSE)。
