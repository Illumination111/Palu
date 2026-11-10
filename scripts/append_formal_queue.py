import argparse,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--apply',action='store_true');args=p.parse_args()
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'outputs/supervised_gpu4_20261106'
plan=json.loads((RUN/'queue.json').read_text());jobs=plan['jobs'];known={j['id'] for j in jobs}
models={'llama3':'/publicdata/model/Meta-Llama-3-8B-Instruct','mistral_v03':'/publicdata/model/Mistral-7B-Instruct-v0.3'}
all_tasks=['qasper','qmsum','multi_news','trec','triviaqa','samsum','lcc','repobench-p']
def add(id,model,action,out,**kw):
    if id not in known:jobs.append(dict(id=id,model=str(model),action=action,output=str(out),**kw));known.add(id)
# Formal calibration and PPL are completed for both models before long task evaluation.
for slug,model in models.items():
    stage=RUN/slug/'formal'
    datasets='wikitext2,c4' if slug=='llama3' else 'wikitext2'
    add(f'formal_{slug}_original_ppl',model,'ppl',stage/'original',datasets=datasets,ppl_blocks=None)
    for ratio,label in [(.7,'ratio07'),(.5,'ratio05')]:
        add(f'formal_{slug}_{label}',model,'compress',stage/label,fisher_n=2048,fisher_len=1024,
            whiten_n=256,whiten_len=2048,ratio=ratio,ppl_blocks=None,datasets=datasets)
for slug,model in models.items():
    stage=RUN/slug/'formal'
    for label in ['original','ratio07','ratio05']:
        ckpt=model if label=='original' else stage/label/'checkpoint'
        add(f'formal_{slug}_{label}_memory',ckpt,'memory',stage/label/'memory',
            lengths=[2048,8192] if slug=='llama3' else [2048,8192,16384,31500])
        if slug=='llama3':
            add(f'formal_{slug}_{label}_zero',ckpt,'zero_shot',stage/label/'zero_shot',limit=None)
# Each cap uses exactly the same first 20 task rows for all three variants.
for cap in (2048,8192,16384,31500):
    for label in ['original','ratio07','ratio05']:
        stage=RUN/'mistral_v03/formal';ckpt=models['mistral_v03'] if label=='original' else stage/label/'checkpoint'
        add(f'formal_mistral_v03_{label}_scan_{cap}',ckpt,'longbench',stage/label/f'scan_{cap}',limit=20,max_input=cap,tasks=all_tasks)
for label in ['original','ratio07','ratio05']:
    stage=RUN/'mistral_v03/formal';ckpt=models['mistral_v03'] if label=='original' else stage/label/'checkpoint'
    add(f'formal_mistral_v03_{label}_long_full',ckpt,'longbench',stage/label/'long_full',limit=None,max_input=31500,tasks=all_tasks)
plan.update(jobs=jobs,sealed=True)
file=RUN/('queue.json.tmp' if args.apply else 'formal_queue_preview.json')
file.write_text(json.dumps(plan,indent=2))
if args.apply:file.replace(RUN/'queue.json')
print('Formal plan total jobs',len(jobs),'applied',args.apply)
