import json,subprocess,sys,time
from pathlib import Path
root=Path(__file__).resolve().parents[1];run=root/'outputs/supervised_gpu4_20261106'
last=None
while True:
    files=list(run.glob('**/result.json'));signature=tuple(sorted((str(p),p.stat().st_mtime_ns) for p in files))
    if signature!=last:
        subprocess.run([sys.executable,str(root/'scripts/local_report.py')],cwd=root,check=True);last=signature
    plan=json.loads((run/'queue.json').read_text())
    if plan['sealed'] and len(list((run/'receipts').glob('*.json')))==len(plan['jobs']):break
    time.sleep(30)
