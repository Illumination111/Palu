# GPU 4 顺序执行说明

运行批次：`outputs/supervised_gpu4_20261106/`。报告自动更新，`report.md` 及 PNG/SVG 保存真实结果。

物理 GPU 4 通过 UUID `GPU-93ee35e2-cf8f-ac28-3768-a5cea298f202` 固定；进程内显示 cuda:0。CUDA 编号与 nvidia-smi 序号不保证一致。

## 执行链

1. 两个模型 smoke：原始与 Palu 70% 对照，8×512 校准、8×2048 PPL，保存重载和固定问答。
2. 扩大验证：Fisher 32×1024、白化 32×2048；三个组 PPL、Llama 六项各 100 例、Mistral qasper/triviaqa 各 20 例，输入上限 2K/8K。
3. 扩大验证检查后追加正式队列：两个模型各 Fisher 2048×1024、白化 256×2048；70%/50% checkpoint 共享同模型校准统计。
4. Llama 完整 WikiText-2/C4 PPL 与六项零样本；Mistral 八项任务每项前 20 例的 2K/8K/16K/31500 扫描，然后三个组完整 test。
5. 单独 memory 作业测 dense KV 张量字节与预填充峰值。

## 日志与恢复

监督器逐作业顺序执行，阶段边界检查 GPU 和磁盘；单个长任务每 1800 秒追加审计，每分钟输出日志末尾。benchmark 保存逐题预测。

`queue.json` 为生效队列；`sealed=false` 时，完成已排任务后等待追加。正式预览在 `formal_queue_preview.json`。成功任务才有 `receipts/<id>.json`。`failure.json` 是历史最近失败，须结合 checks.jsonl 最新事件判断当前状态。

启动/恢复前确认没有同批次监督器运行：

```bash
cd /data/user25101366/NIPS-2027-baseline/Palu
/data/user25101366/.conda/envs/biocache-baseline-palu/bin/python scripts/supervise_local.py
```

恢复跳过成功 receipt。失败任务修复后重启监督器；LongBench 重跑将旧预测归档再生成，避免重复样本评分。

自动报告：

```bash
/data/user25101366/.conda/envs/biocache-baseline-palu/bin/python scripts/watch_local_report.py
```

数据存放 `dataset-benchmark/`；WikiText Arrow 复用 KVTC 文件，路径与 split fingerprints 见 input_manifest.json。下载固定 revision；LongBench zip/C4 分片保存 SHA256。

## 论文差异

Fisher 修正 HF 双移位；冻结非 K/V 权重，已验证目标梯度一致。保留 FP32 RoPE buffer。白化采集 K/V 共用输入协方差，jitter 写入 cache。秩分配按 32-rank 整数块有界分配，实际 fraction 见 ranks.json。

上述修改和 token-window 抽样均作为本地协议差异报告。Mistral-v0.3 是扩展，论文为 v0.2。本路径使用 HF/PyTorch 完整模型，重建 dense K/V；结果仅支持质量与本实现实际资源结论。
