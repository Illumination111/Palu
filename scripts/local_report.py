"""Refresh the auditable Markdown report and standalone scientific figures."""
import json,datetime
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'outputs/supervised_gpu4_20261106'
rows=[]
for file in sorted(RUN.glob('**/result.json')):
    d=json.loads(file.read_text()); j=d['job'];rows.append((j['id'],d,file))
plan=json.loads((RUN/'queue.json').read_text())
lines=['# Palu GPU 4 本地验证记录','', '[正式结果差值与结论](findings.md)', '',f'更新时间：{datetime.datetime.now().astimezone().isoformat()}。成功完成 {len(rows)} / 当前队列 {len(plan["jobs"])} 项；队列封存：{plan["sealed"]}。',
'', 'GPU：物理 4，A100-SXM4-80GB，UUID `GPU-93ee35e2-cf8f-ac28-3768-a5cea298f202`。环境：`biocache-baseline-palu`。',
'', '仅完整模型路径，FP16、batch=1、group=4、Fisher uniform、whiten、seed=42、无 latent 量化。70%/50% 是 rank 保留比例。该路径重建 dense K/V，不能据此声称低秩 KV 显存节省。',
'', '小规模 8×512；扩大 Fisher 32×1024、白化 32×2048；正式 Fisher 2048×1024、白化 256×2048。校准 train 窗口与 PPL test 块分离；同模型三组共用测试块、任务行序与 prompt。PPL 为非重叠 2048 token 块，FP32 交叉熵；每块首 token 不计分。',
'', '## 已完成的 PPL 对照','', '| 作业 | WikiText-2 PPL ↓ | C4 PPL ↓ | 秒数 | 任务峰值 GiB | 重载 logits 最大差 |','|---|---:|---:|---:|---:|---:|']
for name,d,file in rows:
    if 'ppl' in d:
        pp=d['ppl'];lines.append(f'| {name} | {pp.get("wikitext2", "—")} | {pp.get("c4", "—")} | {d["elapsed_seconds"]:.1f} | {d["peak_allocated_bytes"]/2**30:.2f} | {d.get("checkpoint_reload_max_abs", "—")} |')
lines+=['','任务峰值包含该作业的校准/分解/重载过程，不可将 compression 作业峰值与仅推理作业峰值直接作 KV 显存节省比较。memory 作业单独测量预填充与返回 KV 张量。','', '![PPL comparison](ppl_comparison.png)','', '## 已完成的 benchmark','']
for name,d,file in rows:
    if 'benchmark' in d:
        lines+=['### '+name,'','```json',json.dumps(d['benchmark'],ensure_ascii=False,indent=2),'```','']
# Comparable zero-shot task scores; raw acc and acc_norm remain in harness_results.json.
zero=[(n,d) for n,d,_ in rows if d['job']['action']=='zero_shot']
if zero:
    tasks=['openbookqa','hellaswag','piqa','arc_easy','arc_challenge','winogrande']
    metric={t:('acc,none' if t in ('piqa','winogrande') else 'acc_norm,none') for t in tasks}
    lines+=['## 零样本对照表（百分数）','','本表 PIQA/Winogrande 使用 acc，其他四项使用 acc_norm；各任务两类原始指标均保留于 harness_results.json。扩大阶段每任务前 100 题，不能当作完整 benchmark 结果。','',
            '| 作业 | '+' | '.join(tasks)+' |','|---|'+'---:|'*len(tasks)]
    for n,d in zero:
        values=[d['benchmark']['scores'][t][metric[t]]*100 for t in tasks]
        lines.append('| '+n+' | '+' | '.join(f'{v:.2f}' for v in values)+' |')
    fig,ax=plt.subplots(figsize=(11,4))
    width=.8/len(zero)
    for index,(n,d) in enumerate(zero):
        values=[d['benchmark']['scores'][t][metric[t]]*100 for t in tasks]
        ax.bar([i-.4+width/2+index*width for i in range(len(tasks))],values,width,label=n)
    ax.set_xticks(range(len(tasks)),tasks);ax.set_ylabel('Task score (%)');ax.set_ylim(0,100)
    ax.legend(fontsize=7);ax.grid(axis='y',alpha=.2);fig.tight_layout()
    fig.savefig(RUN/'zero_shot_comparison.png',dpi=180);fig.savefig(RUN/'zero_shot_comparison.svg');plt.close(fig)
    lines+=['','![Zero-shot comparison](zero_shot_comparison.png)','']
long=[(n,d) for n,d,_ in rows if d['job']['action']=='longbench']
if long:
    lines+=['## LongBench 对照表','','| 作业 | 输入上限 | 任务 | 分数 |','|---|---:|---|---:|']
    for n,d in long:
        for task,value in d['benchmark']['scores'].items():
            lines.append(f'| {n} | {d["benchmark"]["input_cap"]} | {task} | {value:.2f} |')
    fig,ax=plt.subplots(figsize=(12,4))
    keys=[(n,task,value) for n,d in long for task,value in d['benchmark']['scores'].items()]
    ax.bar(range(len(keys)),[v for _,_,v in keys]);ax.set_xticks(range(len(keys)),[n.replace('mistral_v03_','')+'/'+t for n,t,_ in keys],rotation=75,ha='right',fontsize=6)
    ax.set_ylabel('Official task score (0-100)');ax.grid(axis='y',alpha=.2);fig.tight_layout()
    fig.savefig(RUN/'longbench_comparison.png',dpi=180);plt.close(fig)
    lines+=['','![LongBench comparison](longbench_comparison.png)','']
lines+=['## 运行中断与恢复','','2026-11-07 08:41 检查确认 worker、监督器和审计进程均已退出，退出原因未记录；08:43 恢复。原始完整 LongBench 已完成，70% 组中途断开。恢复前备份已有预测，逐条校验 sample ID、原始行哈希、实际输入 token 哈希和答案，只生成缺失样本；期间审计存在空档，不能称为全程连续监督。证据见 recovery_20261107/。','','']
lines+=['## 修复与协议差异','',
'- 校准从 train token 流抽固定长度窗口，替代原字符截取；样本 ID、seed 和实际长度保存，属于本地修正协议。',
'- Fisher 由 HF 内部移位一次，保留 legacy double-shift 开关；主实验采用修正协议，不能声称逐行复刻原始有误实现。',
'- Fisher 只保留 K/V 权重梯度；小模型上与全部权重求导的 K/V 梯度逐元素一致。',
'- 保留 FP32 RoPE 频率缓冲；原来的 model.half() 会改变频率并造成重载 logits 差异。',
'- 秩分配采用确定性、有上限的整数 32-rank block 分配，避免原浮点预算循环不退出；报告 actual_ratio。这是原分配代码的稳健修正，未宣称 rank 列表与作者完全相同。',
'- 白化只采集 K/V 共用的输入协方差；对称化、Cholesky 与相对 jitter 均记录。三角求解代替显式逆；满秩重建回归通过。',
'- C4 保持指定 validation 分片前 1100 行的原脚本取样；Llama-3 实得 252 个完整 2048-token 块，不人为补齐 256。',
'- 原始序号映射造成首次任务自动 CPU 卸载，随后使用物理 UUID 修正；历史失败日志保留。',
'- Mistral-v0.3 为本地长上下文扩展，论文使用 v0.2；共享权重来源记录于模型 SOURCE.txt。',
'- Mistral chat template 已包含 BOS，编码时不再重复添加。LongBench 保留仓库官方任务模板、评分器与任务输出上限；最终输入加输出受位置上限检查。初步阶段取各任务前 20 行，正式阶段完整 test。',
'', '## 可复查文件','',
'- `queue.json`：参数、顺序与输出路径；`jobs/`、`receipts/`：逐作业配置与成功凭据。',
'- `checks.jsonl`：阶段边界及每 1800 秒资源检查；`logs/`：完整 stdout/stderr。',
'- 模型目录下 `ranks.json`、`checkpoint/`、`result.json`；LongBench 每任务 JSONL 保存 row/input hashes、回答、长度、耗时和显存。',
'- `../../dataset-benchmark/download-manifest.json`：固定数据 revision、大小及关键文件 SHA256。',
'- `source_snapshot/`、`source_manifest.json`：执行源代码快照；`regression.json`：数学回归。',
'', '正式实验尚未全部完成时，上述表格仅表示已落盘结果。']
(RUN/'report.md').write_text('\n'.join(lines)+'\n')
fig,axes=plt.subplots(1,2,figsize=(12,4))
for ax,slug,title in zip(axes,['llama3','mistral_v03'],['Llama-3-8B-Instruct','Mistral-7B-Instruct-v0.3']):
    selected=[(n,d) for n,d,_ in rows if slug in n and 'ppl' in d and 'wikitext2' in d['ppl']]
    if selected:
        ax.bar(range(len(selected)),[d['ppl']['wikitext2'] for _,d in selected],color=['#4c78a8' if 'original' in n else '#f58518' if 'ratio07' in n else '#54a24b' for n,_ in selected])
        ax.set_xticks(range(len(selected)),[n.replace(slug+'_','') for n,_ in selected],rotation=45,ha='right',fontsize=8)
        ax.set_ylabel('WikiText-2 PPL (lower is better)')
    ax.set_title(title);ax.grid(axis='y',alpha=.2)
fig.tight_layout();fig.savefig(RUN/'ppl_comparison.png',dpi=180);fig.savefig(RUN/'ppl_comparison.svg');plt.close(fig)
print('Report refreshed',len(rows),flush=True)

from local_findings import build_findings
build_findings()
