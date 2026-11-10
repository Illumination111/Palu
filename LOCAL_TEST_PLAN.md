# Palu 本地校准与测试执行计划

## 目标与固定设置

只使用完整模型实现路径，环境为 `biocache-baseline-palu`。

| 模型 | 目标 | 首轮数据 |
|---|---|---|
| `/publicdata/model/Meta-Llama-3-8B-Instruct` | 论文同名模型质量对照 | WikiText-2、C4、六项零样本任务 |
| `/publicdata/model/Mistral-7B-Instruct-v0.3` | 本地长上下文质量测试 | LongBench八项任务 |

每个模型均比较三个独立checkpoint：原始模型、Palu保留70% rank、Palu保留50% rank。`param_ratio_target=0.7/0.5`分别对应约30%/50%的rank缩减，不是权重字节或实际KV显存的缩减比例。

首轮固定 FP16、batch=1、head_group_size=4、search_method=fisher_uniform、decompose_method=whiten、lt_bits=16、seed=42。一次只使用一张空闲A100，明确CUDA_VISIBLE_DEVICES，禁止自动跨卡影响可比性。

完整模型路径重建dense K/V，本计划评价PPL、任务分数和实际运行资源；不把理论rank节省或fake quant位数当作真实低秩KV显存/内核加速结果。

## 阶段0：修复正式校准与评测入口

1. Fisher校准：修正输入/labels提前错位后又被HF内部shift的问题；明确输入与labels相同，由模型内部负责移位。保留原实现兼容选项用于必要的论文代码差异核查，两者使用不同缓存。
2. 将n_fisher_calib_samples、n_whiten_calib_samples真正接入代码；拆分fisher_seqlen与whiten_seqlen。白化输入/输出缓冲根据实际长度创建，去掉256×2048的硬编码。
3. 统一数据入口。WikiText优先复用既有本地parquet/Arrow；新增C4、零样本与LongBench资料统一放在Palu/dataset-benchmark，不混入环境目录。不静默更换数据源。
4. 校准从train split按token窗口抽样，不按字符截取后假定长度满足要求；校验样本长度、attention mask和有效标签。评测使用固定独立test/validation集合。
5. 缓存键包含模型配置/权重标识、数据revision、sample IDs、seed、样本数、长度、dtype及代码版本。Fisher、白化、rank搜索分别保存元数据；不复用原有不完整缓存键。
6. PPL增加样本上限与明确输出路径，采用目标token加权的FP32交叉熵；校准与评测代码无静默回退。保留原脚本PPL无缓存协议，与此前KVTC的条件PPL区别标注。
7. 每次Hadamard/量化配置从干净checkpoint加载，避免重复融合；主评测先不量化。
8. 验证压缩checkpoint保存、重新加载、ranks数量/范围、KV heads与group整除关系以及各层输出形状。当前主线采用whiten，普通SVD路径的精确重建断言不作为降rank验收条件。

交付：明确的CLI、回归检查、数据manifest、缓存元数据与运行配置。此阶段完成前不启动2048例Fisher大任务。

## 阶段1：小规模端到端校准与分解

两个模型依次运行，先使用保留70% rank的温和配置。

| 项目 | smoke设置 |
|---|---|
| Fisher校准 | 8个train窗口 × 512 token |
| 白化校准 | 8个train窗口 × 512 token |
| 分解 | group=4，fisher_uniform，whiten，param_ratio_target=0.7 |
| PPL | 8个test块 × 2048 token，原始与分解模型用相同块 |
| 生成检查 | 少量固定prompt，确认无NaN、异常循环与加载错误 |

验收：校准指标有限、梯度与目标模块覆盖正确；rank满足边界；保存恢复后的同输入logits一致；所有组样本ID和配置一致。记录实际耗时、校准峰值、模型文件及缓存大小，据此估算后续资源。小样本不用于宣称准确率保持。

若PPL恶化，先用近满rank与不同保留比例检查分解、标签和保存恢复，区分实现错误与压缩误差；保留失败结果。不得只因分数差而替换test样本。

## 阶段2：中等规模验证

| 项目 | 设置 |
|---|---|
| Fisher | 32 × 1024 token |
| 白化 | 32 × 2048 token |
| 压缩组 | 原始、ratio=0.7、ratio=0.5 |
| Llama PPL | WikiText-2固定32块、长度2048 |
| Llama零样本 | 六项任务各固定100例或实际可用数量 |
| Mistral长上下文 | 先qasper/triviaqa，固定样本集，2K/8K输入上限 |

三组使用同一份校准统计，不同ratio只重做rank分配和分解。先核对原始模型分数、prompt、BOS/EOS、评分器与停止条件，再扩大任务。

质量验收不预设“必须通过”的漂亮阈值；所有退化均报告。只有实现/数据/资源错误才暂停队列并修复重跑受影响组。

## 阶段3：正式校准

对两个模型分别完成并冻结：

- Fisher：2048个WikiText-2 train窗口，每窗口1024 token，与论文说明对齐。
- 白化：256个train窗口，每窗口2048 token，与当前仓库默认过程对齐，明确这不是论文单独给出的统一配额。
- 生成ratio=0.7和ratio=0.5两套ranks与完整模型checkpoint。
- 不同保留比例共享同模型的Fisher/白化统计；两个模型之间不共享统计。
- 每模型先用seed42完成主实验；资源允许时用其他校准seed重复以评估校准敏感性，另作扩展结果。

2048例Fisher包含反向传播，耗时由阶段1/2实测估算，不用仅推理速度估算。校准峰值与评测峰值分开记录。

## 阶段4：Llama-3-8B-Instruct论文质量对照

1. 原始模型先运行WikiText-2、C4的PPL及六项零样本任务，保存逐样本记录。
2. ratio=0.7/0.5按相同协议完整运行。
3. WikiText-2/C4主PPL长度2048；后续量化模拟评测使用4096并单独列出，不能跨长度直接比较PPL。
4. 零样本任务：openbookqa、hellaswag、piqa、arc_easy、arc_challenge、winogrande。使用固定版本harness的官方task定义，固定BOS、fewshot=0和batch；逐任务保留acc/acc_norm的实际key，最终表按论文指标选择，不混用两者。
5. 比较论文表中分数时，同时报告本地原始baseline与本地Palu差值，记录checkpoint来源、版本、数据revision和模板差异。Fisher修正造成的协议差异明确标注，必要时单独做原实现兼容消融。

可选：主线16-bit验证完成后，对干净分解checkpoint测试3-bit、2-bit latent fake quant以及Hadamard开关；这些是质量消融，不声称真实2/3-bit缓存存储。

## 阶段5：Mistral-v0.3长上下文

任务：qasper、qmsum、multi_news、trec、triviaqa、samsum、lcc、repobench-p。

分两种实验，避免混淆：

- 长度扫描：先在固定、可重用的样本ID上，将输入上限设为2048/8192/16384/31500。逐例记录实际输入长度和截断；上限不等于每题实际长度。三种模型对相同长度设置使用相同截断文本，截断可能删除答案的影响由原始模型对照反映。
- 标准任务评分：按任务完整test集与官方prompt/评分器，最大输入31500；输出上限按dataset2maxlen配置。总长度不超过模型32768位置上限，保留模型chat template。

先每任务固定20例验证，再扩大完整数据集；样本不足则如实报告，不重复补数。评分分别采用QA F1、摘要ROUGE、分类分数、代码相似度等任务指标，不把全部任务写成EM或回答准确率。

Mistral-v0.3是本地扩展，与论文v0.2版本不同，不以绝对分数相同作为验收条件。

## 监督与结果目录

所有步骤顺序执行，逐样本记录进度；阶段结束立即检查，长任务每1800秒检查一次进程、进度、GPU/RAM、磁盘和错误。仅对可恢复错误重跑，保留旧日志与失败配置。

建议目录：

```text
Palu/
  dataset-benchmark/             # 原始数据与revision manifest
  outputs/full_model_validation/<run_id>/
    run_config.json
    source_snapshot/
    checks.jsonl
    llama3/{calibration,checkpoints,results,logs}/
    mistral_v03/{calibration,checkpoints,results,logs}/
    report.md
```

校准/模型产物可能较大：原始权重直接用共享路径，不复制；白化矩阵和分解checkpoint按实际占用记录，先由smoke测得大小再规划完整任务空间，不把rank保留比例当作checkpoint磁盘压缩比例。

最终交付包括：参数及环境锁、数据与样本manifest、Fisher/白化统计、两种保留比例的checkpoint、逐任务原始分数、相对原始模型的差值、运行资源表和可视化。PPL越低越好；任务分数依各自指标解释。

本文件是执行计划，并非已完成的正式校准或benchmark结果。环境与接口验证证据见LOCAL_SETUP.md和../env-locks/palu/verification.json。
