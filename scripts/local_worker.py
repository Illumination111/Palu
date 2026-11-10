"""One GPU, one model, one reproducible validation job."""
import argparse, gc, hashlib, json, os, sys, time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from utils import load_model_and_tokenizer, dump_to_huggingface_repos, set_seed
from palu.rank_search import rank_search
from palu.decomposition import compress_model
from palu.model import HeadwiseLowRankModule
from run_ppl_eval import eval_ppl

p=argparse.ArgumentParser();p.add_argument('--job',required=True); a=p.parse_args()
j=json.loads(Path(a.job).read_text()); out=Path(j['output']);out.mkdir(parents=True,exist_ok=True)
set_seed(42); started=time.time(); torch.cuda.reset_peak_memory_stats()
root=Path(__file__).resolve().parents[1]
source_paths=[root/'utils.py',root/'run_ppl_eval.py']+list((root/'palu').rglob('*.py'))+list((root/'scripts').glob('*.py'))
source_hashes={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
(out/'source_hashes.json').write_text(json.dumps(source_hashes,indent=2))
if j['action']=='audit':
    import subprocess
    subprocess.run([sys.executable,str(root/'scripts/final_local_audit.py')],check=True)
    (out/'result.json').write_text(json.dumps({'job':j,'audit_passed':True,'elapsed_seconds':time.time()-started,'peak_allocated_bytes':0},indent=2))
    sys.exit(0)
model,tok=load_model_and_tokenizer(j['model'])
result={'job':j,'started':started,'physical_gpu':os.environ['CUDA_VISIBLE_DEVICES']}
if j['action']=='compress':
    args=argparse.Namespace(model_id=j['model'],seed=42,device='cuda',use_cache=True,
        calib_dataset='wikitext2',n_fisher_calib_samples=j['fisher_n'],calib_seqlen=j['fisher_len'],
        n_whiten_calib_samples=j['whiten_n'],whiten_seqlen=j['whiten_len'],head_group_size=4,
        search_method='fisher_uniform',param_ratio_target=j['ratio'],decompose_method='whiten',legacy_double_shift=False)
    selected,total,full=rank_search(model,tok,args)
    result['ranks']={'selected':selected,'sum':total,'full':full,'actual_ratio':total/full}
    (out/'ranks.json').write_text(json.dumps(result['ranks'],indent=2))
    compress_model(model,tok,args,'cuda',selected)
    model.eval()
    probe=tok('The capital of France is',return_tensors='pt').input_ids.to('cuda')
    with torch.no_grad(): before=model(probe,use_cache=False).logits.cpu()
    checkpoint=out/'checkpoint'
    dump_to_huggingface_repos(model,tok,str(checkpoint),args)
    del model;gc.collect();torch.cuda.empty_cache()
    model,tok=load_model_and_tokenizer(str(checkpoint))
    with torch.no_grad(): after=model(probe,use_cache=False).logits.cpu()
    delta=(before-after).abs().max().item()
    if not torch.allclose(before,after,atol=1e-3,rtol=1e-3):
        raise RuntimeError(f'Checkpoint logits mismatch {delta}')
    assert len([m for m in model.modules() if isinstance(m,HeadwiseLowRankModule)])==64
    result['checkpoint_reload_max_abs']=delta
    result['checkpoint_bytes']=sum(f.stat().st_size for f in checkpoint.rglob('*') if f.is_file())
if j['action'] in ('compress','ppl'):
    result['ppl']=eval_ppl(model,tok,j['model'],j.get('datasets','wikitext2'),2048,'cuda',j.get('ppl_blocks'))
    result['ppl_statistics']=eval_ppl.last_stats
    if j.get('generate',False):
        result['generation']=[]
        for prompt in ('What is the capital of France?','Calculate 12 + 7.'):
            ids=tok.apply_chat_template([{'role':'user','content':prompt}],add_generation_prompt=True,return_tensors='pt').to('cuda')
            with torch.no_grad(): answer=model.generate(ids,max_new_tokens=48,do_sample=False,pad_token_id=tok.eos_token_id)
            result['generation'].append({'prompt':prompt,'answer':tok.decode(answer[0,ids.shape[1]:],skip_special_tokens=True)})
if j['action'] in ('zero_shot','longbench','memory'):
    from local_benchmarks import zero_shot,longbench,memory_profile
    result['benchmark']={'zero_shot':zero_shot,'longbench':longbench,'memory':memory_profile}[j['action']](model,tok,j,out)
result.update(elapsed_seconds=time.time()-started,peak_allocated_bytes=max(torch.cuda.max_memory_allocated(),result.get('benchmark',{}).get('peak_allocated_bytes',0)),peak_reserved_bytes=torch.cuda.max_memory_reserved())
(out/'result.json').write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2),flush=True)
