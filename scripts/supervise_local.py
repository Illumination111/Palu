"""Sequential subprocess queue, resumable receipts, 30-minute resource audit."""
import datetime, json, os, shutil, subprocess, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'outputs/supervised_gpu4_20261106';RUN.mkdir(parents=True,exist_ok=True)
os.chdir(ROOT)
os.environ.update(CUDA_VISIBLE_DEVICES='GPU-93ee35e2-cf8f-ac28-3768-a5cea298f202',PYTHONUNBUFFERED='1',TOKENIZERS_PARALLELISM='false',HF_HOME=str(ROOT/'dataset-benchmark/hf-cache'),HF_DATASETS_CACHE=str(ROOT/'dataset-benchmark/datasets-cache'),OMP_NUM_THREADS='8',MKL_NUM_THREADS='8')

def audit(event,job=None,process=None):
    gpu=subprocess.run(['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu','--format=csv,noheader','-i','4'],capture_output=True,text=True).stdout.strip()
    line={'time':datetime.datetime.now().astimezone().isoformat(),'event':event,'job':job,'pid':None if process is None else process.pid,'gpu':gpu,'disk_free_bytes':shutil.disk_usage(ROOT).free}
    with (RUN/'checks.jsonl').open('a') as f:f.write(json.dumps(line)+'\n')
    print(json.dumps(line),flush=True)
while True:
    plan=json.loads((RUN/'queue.json').read_text())
    pending=[j for j in plan['jobs'] if not (RUN/'receipts'/f"{j['id']}.json").exists()]
    if not pending:
        if plan.get('sealed'):
            audit('queue_complete');break
        time.sleep(20);continue
    job=pending[0];jobfile=RUN/'jobs'/f"{job['id']}.json";jobfile.parent.mkdir(exist_ok=True);jobfile.write_text(json.dumps(job,indent=2))
    logpath=RUN/'logs'/f"{job['id']}.log";logpath.parent.mkdir(exist_ok=True)
    command=[sys.executable,'scripts/local_worker.py','--job',str(jobfile)]
    audit('start',job['id'])
    with logpath.open('a') as log:
        proc=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
        last=time.monotonic();last_status=last
        while proc.poll() is None:
            time.sleep(10)
            if time.monotonic()-last>=1800:
                audit('half_hour_check',job['id'],proc);last=time.monotonic()
            if time.monotonic()-last_status>=60:
                tail=logpath.read_text(errors='replace')[-1200:]
                print(json.dumps({'job':job['id'],'log_tail':tail}),flush=True);last_status=time.monotonic()
    audit('finished' if proc.returncode==0 else 'failed',job['id'],proc)
    receipt={'job':job,'returncode':proc.returncode,'log':str(logpath),'finished':time.time()}
    if proc.returncode:
        (RUN/'failure.json').write_text(json.dumps(receipt,indent=2));sys.exit(proc.returncode)
    folder=RUN/'receipts';folder.mkdir(exist_ok=True);(folder/f"{job['id']}.json").write_text(json.dumps(receipt,indent=2))
