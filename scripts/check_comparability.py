import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'outputs/supervised_gpu4_20261106'
checks=[]
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True).encode()).hexdigest()
for stage in ('expanded','formal'):
    zero=[]
    for label in ('original','ratio07','ratio05'):
        file=RUN/'llama3'/stage/label/'zero_shot/harness_results.json'
        if file.exists():
            d=json.loads(file.read_text())
            values={task:[(row['doc_id'],row['doc'],row['target'],row['arguments']) for row in rows] for task,rows in d['samples'].items()}
            zero.append((label,digest(values),{task:len(rows) for task,rows in d['samples'].items()}))
    if len(zero)>1:
        assert len({v[1] for v in zero})==1,(stage,zero)
        checks.append({'kind':'zero_shot','stage':stage,'variants':zero,'identical':True})
    variants={}
    for label in ('original','ratio07','ratio05'):
        folder=RUN/'mistral_v03'/stage/label
        for prediction in folder.glob('*/*.jsonl'):
            if '.previous-' in prediction.name:continue
            rows=[json.loads(x) for x in prediction.read_text().splitlines() if x.strip()]
            if rows and 'input_ids_sha256' in rows[0]:
                key=prediction.parent.name+'/'+prediction.stem
                variants.setdefault(key,[]).append((label,digest([(r['sample_id'],r['row_sha256'],r['input_ids_sha256'],r['actual_input_tokens']) for r in rows]),len(rows)))
    for key,values in variants.items():
        if len(values)==3:
            assert len({v[1] for v in values})==1,(stage,key,values)
            checks.append({'kind':'longbench','stage':stage,'set':key,'variants':values,'identical':True})
(RUN/'comparability.json').write_text(json.dumps(checks,indent=2));print('COMPARABILITY_PASSED',len(checks),flush=True)
