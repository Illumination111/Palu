"""Independent half-hour audit, including checks while queue waits between stages."""
import datetime,json,os,shutil,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'outputs/supervised_gpu4_20261106'

def audit():
    plan=json.loads((RUN/'queue.json').read_text())
    checks=(RUN/'checks.jsonl').read_text().splitlines()
    latest=json.loads(checks[-1]) if checks else {}
    gpu=subprocess.run(['nvidia-smi','-i','4','--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout.strip()
    processes=subprocess.run(['ps','-eo','pid,ppid,stat,etime,%cpu,rss,args'],capture_output=True,text=True,check=True).stdout
    active=[line.strip() for line in processes.splitlines() if ('scripts/local_worker.py' in line or 'scripts/supervise_local.py' in line) and 'ps -eo' not in line]
    logs=sorted((RUN/'logs').glob('*.log'),key=lambda p:p.stat().st_mtime)
    lastlog=logs[-1] if logs else None
    tail='' if lastlog is None else lastlog.read_text(errors='replace')[-1200:]
    data={'time':datetime.datetime.now().astimezone().isoformat(),'event':'independent_half_hour_audit',
          'completed':len(list((RUN/'receipts').glob('*.json'))),'planned':len(plan['jobs']),
          'latest_event':latest,'gpu':gpu,'processes':active,'memory':Path('/proc/meminfo').read_text(),
          'disk_free_bytes':shutil.disk_usage(ROOT).free,'last_log':str(lastlog),'log_tail':tail}
    with (RUN/'half_hour_audits.jsonl').open('a') as f:f.write(json.dumps(data)+'\n')
    print(json.dumps({k:v for k,v in data.items() if k not in ('memory','log_tail')}),flush=True)
    return plan['sealed'] and data['completed']==data['planned']
while True:
    if audit():break
    # Short sleeps keep the process interruptible; the audit interval stays exactly 1800 seconds.
    deadline=time.monotonic()+1800
    while time.monotonic()<deadline:time.sleep(min(30,deadline-time.monotonic()))
