"""Pinned benchmark preparation; all artifacts stay below dataset-benchmark."""
import hashlib,json,os,zipfile,subprocess
from pathlib import Path
from huggingface_hub import HfApi,hf_hub_download,snapshot_download
ROOT=Path(__file__).resolve().parents[1];D=ROOT/'dataset-benchmark'
os.environ.setdefault('HF_HOME',str(D/'hf-cache'))
api=HfApi();manifest={}
def download(repo,rev,file,folder):
    path=folder/file
    if not path.exists():
        path.parent.mkdir(parents=True,exist_ok=True)
        subprocess.run(['curl','--fail','--location','--retry','3','--max-time','900',
            f'https://huggingface.co/datasets/{repo}/resolve/{rev}/{file}', '-o', str(path)+'.part'],check=True)
        Path(str(path)+'.part').replace(path)
    p=Path(path);manifest[str(p.relative_to(D))]={'repo':repo,'revision':rev,'file':file,'bytes':p.stat().st_size,'sha256':hashlib.file_digest(p.open('rb'),'sha256').hexdigest() if hasattr(hashlib,'file_digest') else hashlib.sha256(p.read_bytes()).hexdigest()}
    (D/'download-manifest.json').write_text(json.dumps(manifest,indent=2));print('downloaded',file,flush=True)
    return p
rev='5e628be450b7e67fb7ae6e201bd6d8f7056f7672'
z=download('THUDM/LongBench',rev,'data.zip',D/'longbench')
with zipfile.ZipFile(z) as archive:
    for name in archive.namelist():
        if name.endswith('.jsonl') and Path(name).stem in ['qasper','qmsum','multi_news','trec','triviaqa','samsum','lcc','repobench-p']:
            archive.extract(name,D/'longbench')
rev='607bd4c8450a42878aa9ddc051a65a055450ef87'
p=download('allenai/c4',rev,'en/c4-validation.00000-of-00008.json.gz',D/'c4')
link=D/'c4/c4-validation.00000-of-00008.json.gz'
if not link.exists():link.symlink_to(Path('en')/p.name)
# Prepare data-only local copies for harness task overrides.
for repo,rev,pattern in [
('allenai/ai2_arc','210d026faf9955653af8916fad021475a3f00453',['ARC-Easy/*.parquet','ARC-Challenge/*.parquet']),
('allenai/openbookqa','388097ea7776314e93a529163e0fea805b8a6454',['main/*.parquet']),
('Rowan/hellaswag','218ec52e09a7e7462a5400043bb9a69a41d06b76',['data/*.parquet']),
('allenai/winogrande','01e74176c63542e6b0bcb004dcdea22d94fb67b5',['winogrande_xl/*.parquet'])]:
    folder=D/'zero-shot'/repo.split('/')[-1]
    snapshot_download(repo,repo_type='dataset',revision=rev,allow_patterns=pattern,local_dir=str(folder),max_workers=2)
    manifest[repo]={'revision':rev,'local_path':str(folder),'patterns':pattern}
    (D/'download-manifest.json').write_text(json.dumps(manifest,indent=2))
# PIQA upstream repository is script based. Cache its official split files locally.
from datasets import load_dataset
piqa=load_dataset('ybisk/piqa',revision='2e8ac2dffd59bac8c3c6714948f4c551a0848bb0',trust_remote_code=True,cache_dir=str(D/'datasets-cache'))
piqa.save_to_disk(str(D/'zero-shot/piqa-arrow'))
manifest['ybisk/piqa']={'revision':'2e8ac2dffd59bac8c3c6714948f4c551a0848bb0','fingerprints':{k:v._fingerprint for k,v in piqa.items()}}
(D/'download-manifest.json').write_text(json.dumps(manifest,indent=2))
print('ALL_DATA_READY',flush=True)
