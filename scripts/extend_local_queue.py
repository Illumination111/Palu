import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'outputs/supervised_gpu4_20261106'
plan=json.loads((RUN/'queue.json').read_text());jobs=plan['jobs'];known={j['id'] for j in jobs}
models={'llama3':'/publicdata/model/Meta-Llama-3-8B-Instruct','mistral_v03':'/publicdata/model/Mistral-7B-Instruct-v0.3'}
def add(id,model,action,output,**kw):
    if id not in known:jobs.append(dict(id=id,model=str(model),action=action,output=str(output),**kw));known.add(id)
for slug,model in models.items():
    stage=RUN/slug/'expanded'
    add(f'expanded_{slug}_original',model,'ppl',stage/'original',ppl_blocks=32)
    for ratio,label in [(.7,'ratio07'),(.5,'ratio05')]:
        add(f'expanded_{slug}_{label}',model,'compress',stage/label,fisher_n=32,fisher_len=1024,whiten_n=32,whiten_len=2048,ratio=ratio,ppl_blocks=32)
    for label in ['original','ratio07','ratio05']:
        checkpoint=model if label=='original' else stage/label/'checkpoint'
        if slug=='llama3':
            add(f'expanded_{slug}_{label}_zero',checkpoint,'zero_shot',stage/label/'zero_shot',limit=100)
        else:
            for cap in (2048,8192):
                add(f'expanded_{slug}_{label}_long_{cap}',checkpoint,'longbench',stage/label/f'long_{cap}',limit=20,max_input=cap,tasks=['qasper','triviaqa'])
# The formal queue is appended only after expanded checks are reviewed.
plan['jobs']=jobs
(RUN/'queue.json.tmp').write_text(json.dumps(plan,indent=2));(RUN/'queue.json.tmp').replace(RUN/'queue.json')
print('Queued jobs',len(jobs))
