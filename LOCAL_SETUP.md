# Palu 完整模型路径：本地环境与模型分工

## 已完成

环境位于 `/data/user25101366/.conda/envs/biocache-baseline-palu`，独立于 KVTC 环境。

```bash
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate /data/user25101366/.conda/envs/biocache-baseline-palu
cd /data/user25101366/NIPS-2027-baseline/Palu
python -m pip check
```

核心版本：PyTorch 2.4.0+cu124、Transformers 4.40.2、Triton 3.0.0、datasets 2.19.2、Accelerate 0.32.1、LM Evaluation Harness 0.4.2、fast-hadamard-transform 1.0.4.post1。

相对上游 requirements 的兼容调整：Transformers 4.37.2 → 4.40.2，以支持选定 Llama-3 的分词器；使用 `fschat==0.2.36` 提供 LMSYS `fastchat.model`；固定 PEFT/evaluate/Arrow 等版本避免旧评测接口与新依赖混用；Ninja 采用1.13.0以通过本机平台检查。原始 requirements.txt 保留。

Hadamard 扩展从固定子模块源码编译，使用系统 CUDA toolkit 12.8、MAX_JOBS=2；PyTorch 自带运行库为12.4。CUDA小张量往返验证通过。该扩展是完整模型代码的导入依赖；本次没有运行 kernel/ 下的 Palu attention/kernel 性能评测。

## 仓库完整性

| 仓库 | 固定 commit |
|---|---|
| Palu | bb22666e2ef96707e8dd21d93fc00146c2e0d615 |
| lm-evaluation-harness | 4600d6bf73ba2cf7037ae7feada03315839ef185 |
| fast-hadamard-transform | d1a56eee9d502e67faacf61b7b947180d66b32a0 |

两个原来为空的子模块已完整 clone，按上述 commit 检出。三个仓库均为非浅克隆，`git fsck --full` 通过，未发现缺失的已跟踪文件。lm-eval 的本地子模块 URL 改为 HTTPS，以避免原 SSH URL 依赖个人 SSH 授权；.gitmodules 中上游配置保留。

fast-hadamard-transform 子模块中的本地 `.so` 和 `__pycache__` 是编译/验证产物，可能使 git status 显示未跟踪文件，不代表源码缺失。完整审计见 [repository-check.json](../env-locks/palu/repository-check.json)。

## 模型分工

| 本地模型 | 用途 | 权重检查 |
|---|---|---|
| `/publicdata/model/Meta-Llama-3-8B-Instruct` | 与论文同名模型的PPL、零样本任务对照 | 4/4分片，可读；原始模型FP16短前向通过 |
| `/publicdata/model/Mistral-7B-Instruct-v0.3` | 本地长上下文测试 | 3/3分片，可读；原始模型FP16短前向通过 |

共享Mistral的SOURCE.txt说明来源为ModelScope复制的unsloth/Mistral-7B-Instruct-v0.3。它与论文的v0.2不同，应标记为本地扩展实验。Llama-3-8B非Instruct目录缺少config；Qwen2.5-3B链接目标无读取权限；两者均未选用。共享根目录未发现对应Llama-2或LongChat目录。

两个选定模型均已通过离线配置、分词器及对话模板检查。Llama最大位置8192，Mistral最大位置32768。参数规划见 [configs/local_models.json](configs/local_models.json)；这是实验规范文件，旧脚本不会自动读取它。

Mistral-v0.3已加入LongBench模型名映射，最大输入31500，并使用其自身chat template。该映射和模板经离线验证。后续总输入加输出仍须在32768位置上限内。

## 论文实验与本地对应

论文主表使用Llama-2-7B、Llama-3-8B-Instruct；评测WikiText-2/C4 PPL及OpenBookQA、HellaSwag、PIQA、ARC-Easy、ARC-Challenge、WinoGrande。长上下文使用Mistral-7B-v0.2、LongChat-7B-v1.5与LongBench。来源：[论文第4节及附录H](https://arxiv.org/html/2407.21118v2)。

论文正文称LongBench 16项，附录H.2及当前默认代码列出8项；本地先按可核查的8项任务列表配置，具体名称见local_models.json，不声称已经对齐全部16项。

后续先测试16-bit latent、group size 4、whiten与fisher_uniform；param_ratio_target=0.7/0.5分别代表约保留70%/50%的rank。完整模型路径可评估分解与量化模拟对质量的影响，但会重建dense K/V，不能据此报告低秩KV缓存的实际内存或kernel加速收益。

## 验证证据

```bash
CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
python ../env-locks/palu/verify_full_model.py
```

本次通过：

- compress.py、run_ppl_eval.py、run_lm_eval.py、run_long_bench.py导入。
- Palu Llama/Mistral小型完整模型前向、保存、重新加载后logits一致。
- 两个共享原始模型的FP16短前向，A100上输出有限值。
- 既有KVTC目录下WikiText Arrow数据可由datasets 2.19.2读取。
- `pip check`无依赖冲突。

详细记录：[verification.json](../env-locks/palu/verification.json)、[models.json](../env-locks/palu/models.json)、[tokenizer-check.json](../env-locks/palu/tokenizer-check.json)。

这些检查不等于已经完成7B/8B模型的Fisher校准、白化分解或benchmark评分。本次没有生成正式Palu准确率/PPL结果。原评测脚本仍默认通过HF加载数据；正式运行前应接入本地数据目录，并核查校准标签移位、样本数硬编码等已发现的历史逻辑。

## 精确恢复

完整版本、build及恢复步骤统一记录在 [env.md](../env.md#palu-完整模型环境2026-11-06)，锁定文件在 [env-locks/palu](../env-locks/palu)。新机器还需要CUDA编译工具链，以重新构建Hadamard扩展。
