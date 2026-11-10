"""Write formal comparisons from completed artifacts, preserving missing results."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'outputs/supervised_gpu4_20261106'
def read(path):return json.loads(path.read_text()) if path.exists() else None
def build_findings():
    lines=['# 正式实验结果解读','','以下仅使用 formal 目录结果；未完成项标为待完成。rank 保留比例不是实际 KV 字节缩减比例。','','## PPL（越低越好）','','| 模型/数据 | 原始 | 70% rank | 变化 | 50% rank | 变化 |','|---|---:|---:|---:|---:|---:|']
    for slug,title in [('llama3','Llama-3-8B-Instruct'),('mistral_v03','Mistral-v0.3')]:
        stage=RUN/slug/'formal';base=read(stage/'original/result.json')
        if base:
            for task,b in base['ppl'].items():
                cells=[]
                for label in ('ratio07','ratio05'):
                    d=read(stage/label/'result.json');v=d['ppl'][task] if d else None
                    cells.extend([f'{v:.3f}',f'{(v/b-1)*100:+.2f}%'] if v is not None else ['待完成','—'])
                lines.append('| '+' | '.join([title+'/'+task,f'{b:.3f}']+cells)+' |')
    lines+=['','## Llama-3 完整零样本评测','','PIQA/Winogrande 展示 acc，其余展示 acc_norm；保留全部原始指标，不将这个混合指标表声称为论文统一平均分。','','| 任务 | 原始 % | 70% rank % | 差值 pp | 50% rank % | 差值 pp |','|---|---:|---:|---:|---:|---:|']
    group={label:read(RUN/'llama3/formal'/label/'zero_shot/result.json') for label in ('original','ratio07','ratio05')}
    if group['original']:
        for task in ('openbookqa','hellaswag','piqa','arc_easy','arc_challenge','winogrande'):
            metric='acc,none' if task in ('piqa','winogrande') else 'acc_norm,none'
            b=group['original']['benchmark']['scores'][task][metric]*100;cells=[]
            for label in ('ratio07','ratio05'):
                d=group[label];v=d['benchmark']['scores'][task][metric]*100 if d else None
                cells.extend([f'{v:.2f}',f'{v-b:+.2f}'] if v is not None else ['待完成','—'])
            lines.append('| '+' | '.join([task,f'{b:.2f}']+cells)+' |')
    lines+=['','## Mistral 完整 LongBench','','以下为完整 test，不能与每任务前20例的长度扫描混用。不同任务使用官方各自评分器。','','| 任务 | 原始 | 70% rank | 差值 pp | 50% rank | 差值 pp |','|---|---:|---:|---:|---:|---:|']
    group={label:read(RUN/'mistral_v03/formal'/label/'long_full/result.json') for label in ('original','ratio07','ratio05')}
    if group['original']:
        for task,b in group['original']['benchmark']['scores'].items():
            cells=[]
            for label in ('ratio07','ratio05'):
                d=group[label];v=d['benchmark']['scores'][task] if d else None
                cells.extend([f'{v:.2f}',f'{v-b:+.2f}'] if v is not None else ['待完成','—'])
            lines.append('| '+' | '.join([task,f'{b:.2f}']+cells)+' |')
    lines+=['','## 显存结论','','对两个模型分别比较原始、70%、50% 三组，同长度返回的 KV 张量形状和字节均相同。此完整模型实现重建 dense K/V；权重体积或峰值显存的少量变化不代表 KV 压缩。Mistral 31500-token 的 KV 张量为 4,128,768,000 字节，约 3.845 GiB。预填充时长为单次测量，不作稳定加速比结论。','','## 适用范围','','本地修正了标签双移位、RoPE buffer、整数秩预算与校准数据取样等问题；这些协议差异见 report.md。Llama-3 使用论文同名模型，本地 Mistral-v0.3 是相对论文 v0.2 的扩展。运行期间发生过进程退出与监督空档，已有预测经逐题身份及 token 哈希校验后恢复；不隐去这一限制。']
    (RUN/'findings.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':build_findings()
