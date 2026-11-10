"""Audit every queued deliverable against actual result and prediction files."""
import hashlib,json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'outputs/supervised_gpu4_20261106'

def audit_coverage():
    plan=json.loads((RUN/'queue.json').read_text());checks=[]
    for job in plan['jobs']:
        if job['action']=='audit':continue
        receipt=json.loads((RUN/'receipts'/f"{job['id']}.json").read_text())
        assert receipt['returncode']==0 and receipt['job']==job
        out=Path(job['output']);result=json.loads((out/'result.json').read_text())
        assert result['job']==job
        if 'ppl' in result:assert all(math.isfinite(v) and v>0 for v in result['ppl'].values())
        if job['action']=='compress':
            assert (out/'checkpoint/config.json').exists()
            index=json.loads((out/'checkpoint/model.safetensors.index.json').read_text())
            assert all((out/'checkpoint'/name).exists() for name in set(index['weight_map'].values()))
            assert result['checkpoint_reload_max_abs']==0
        if job['action']=='longbench':
            counts={}
            for task in job['tasks']:
                source=ROOT/'dataset-benchmark/longbench/data'/f'{task}.jsonl'
                expected=[json.loads(line) for line in source.read_text().splitlines() if line.strip()]
                if job.get('limit') is not None:expected=expected[:job['limit']]
                rows=[json.loads(line) for line in (out/f'{task}.jsonl').read_text().splitlines() if line.strip()]
                assert len(rows)==len(expected),(job['id'],task,len(rows),len(expected))
                for i,(prediction,original) in enumerate(zip(rows,expected)):
                    assert prediction['sample_id']==f'{task}:{i}'
                    assert prediction['row_sha256']==hashlib.sha256(json.dumps(original,sort_keys=True).encode()).hexdigest()
                    assert prediction['answers']==original['answers']
                    assert 0<prediction['actual_input_tokens']<=job['max_input']
                counts[task]=len(rows)
            assert set(result['benchmark']['scores'])==set(job['tasks'])
            checks.append({'job':job['id'],'complete_rows':counts})
        elif job['action']=='zero_shot':
            harness=json.loads((out/'harness_results.json').read_text())
            counts={task:len(samples) for task,samples in harness['samples'].items()}
            assert len(counts)==6 and all(n>0 for n in counts.values())
            checks.append({'job':job['id'],'scored_rows':counts})
        else:checks.append({'job':job['id'],'result_and_receipt_valid':True})
    (RUN/'coverage_audit.json').write_text(json.dumps({'passed':True,'checks':checks},indent=2))
    return checks
if __name__=='__main__':audit_coverage();print('COVERAGE_AUDIT_PASSED')
