import json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'outputs/supervised_gpu4_20261106'
subprocess.run([sys.executable,str(ROOT/'scripts/check_comparability.py')],check=True)
from audit_job_coverage import audit_coverage
audit_coverage()
checks=[]
for slug in ('llama3','mistral_v03'):
    stage=RUN/slug/'formal'
    baseline=json.loads((stage/'original/result.json').read_text())
    for label in ('ratio07','ratio05'):
        d=json.loads((stage/label/'result.json').read_text())
        assert d['checkpoint_reload_max_abs']==0
        assert d['ppl_statistics']==baseline['ppl_statistics'] or all(
            {k:v for k,v in stat.items() if k!='nll_sum'}=={k:v for k,v in baseline['ppl_statistics'][task].items() if k!='nll_sum'}
            for task,stat in d['ppl_statistics'].items())
        ranks=d['ranks']['selected']
        assert len(ranks)==64 and all(len(v)==2 and all(32<=r<=512 and r%32==0 for r in v) for v in ranks.values())
        checks.append({'model':slug,'variant':label,'ppl_inputs_equal':True,'checkpoint_reload_exact':True,'ranks_valid':True})
    sizes=[]
    for label in ('original','ratio07','ratio05'):
        d=json.loads((stage/label/'memory/result.json').read_text())
        sizes.append([(m['input_tokens'],m['kv_tensor_bytes'],m['first_layer_shapes']) for m in d['benchmark']['measurements']])
    assert sizes[0]==sizes[1]==sizes[2]
    checks.append({'model':slug,'dense_kv_sizes_equal':True,'sizes':sizes[0]})
(RUN/'final_audit.json').write_text(json.dumps({'passed':True,'checks':checks},indent=2))
subprocess.run([sys.executable,str(ROOT/'scripts/local_report.py')],check=True)
print('FINAL_AUDIT_PASSED',flush=True)
