import hashlib,json,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'dataset-benchmark'

def zero_shot(model,tokenizer,job,out):
    import lm_eval
    from lm_eval.models.huggingface import HFLM
    from lm_eval.utils import load_yaml_config
    from lm_eval.tasks import TaskManager
    base=ROOT/'3rdparty/lm-evaluation-harness/lm_eval/tasks'
    paths={'openbookqa':'openbookqa/openbookqa.yaml','hellaswag':'hellaswag/hellaswag.yaml',
           'piqa':'piqa/piqa.yaml','arc_easy':'arc/arc_easy.yaml','arc_challenge':'arc/arc_challenge.yaml',
           'winogrande':'winogrande/default.yaml'}
    folders={'openbookqa':'openbookqa/main','hellaswag':'hellaswag/data',
             'arc_easy':'ai2_arc/ARC-Easy','arc_challenge':'ai2_arc/ARC-Challenge',
             'winogrande':'winogrande/winogrande_xl'}
    configs=[]
    for task in job.get('tasks',list(paths)):
        config=load_yaml_config(str(base/paths[task]))
        if task=='piqa':
            from datasets import load_from_disk
            folder=DATA/'zero-shot/piqa-parquet';folder.mkdir(exist_ok=True)
            for split,ds in load_from_disk(str(DATA/'zero-shot/piqa-arrow')).items():
                file=folder/f'{split}.parquet'
                if not file.exists():ds.to_parquet(str(file))
        else:folder=DATA/'zero-shot'/folders[task]
        files={split:sorted(str(p) for p in folder.glob(f'{split}*.parquet')) for split in ('train','validation','test')}
        files={k:v for k,v in files.items() if v}
        config.update(dataset_path='parquet',dataset_name=None,dataset_kwargs={'data_files':files})
        configs.append(config)
    # Preserve the repository's official task templates and metrics, only replace data transport.
    wrapper=HFLM(pretrained=model,tokenizer=tokenizer,batch_size=1,max_length=4096,add_bos_token=False)
    res=lm_eval.simple_evaluate(model=wrapper,tasks=configs,task_manager=TaskManager(),num_fewshot=0,
          limit=job.get('limit'),log_samples=True,bootstrap_iters=1000,random_seed=42,numpy_random_seed=42,torch_random_seed=42)
    (out/'harness_results.json').write_text(json.dumps(res,indent=2,default=str))
    return {'scores':res['results'],'sample_counts':res.get('n-samples'),'protocol':'official_harness_4600d6b_local_data_zero_shot_no_bos'}

@torch.no_grad()
def longbench(model,tokenizer,job,out):
    from longbench_utils import DATASET2PROMPT,DATASET2MAXLEN,scorer
    from run_long_bench import build_chat
    scores={}; overall_peak=0
    for task in job.get('tasks',['qasper','triviaqa']):
        file=DATA/'longbench/data'/f'{task}.jsonl'
        data=[json.loads(x) for x in file.read_text().splitlines() if x.strip()]
        if job.get('limit') is not None:data=data[:job['limit']]
        predictions=[];predfile=out/f'{task}.jsonl'
        saved=[]
        if predfile.exists():
            saved=[json.loads(line) for line in predfile.read_text().splitlines() if line.strip()]
            if len(saved)>len(data): raise ValueError('Resume has excess samples')
            print(f'Resume {task}: validating {len(saved)} existing predictions',flush=True)
        for i,row in enumerate(data):
            started=time.time();torch.cuda.reset_peak_memory_stats()
            prompt=DATASET2PROMPT[task].format(**row)
            ids=tokenizer(prompt,return_tensors='pt').input_ids[0]
            maxlen=job['max_input'];raw_length=len(ids)
            if len(ids)>maxlen:
                half=maxlen//2
                prompt=tokenizer.decode(ids[:half],skip_special_tokens=True)+tokenizer.decode(ids[-half:],skip_special_tokens=True)
            templated = task not in ['trec','triviaqa','samsum','lsht','lcc','repobench-p']
            if templated:
                prompt=build_chat(tokenizer,prompt,'Mistral-7B-Instruct-v0.3')
            inputs=tokenizer(prompt,return_tensors='pt',add_special_tokens=not templated).to('cuda')
            # Apply the final cap after chat-template insertion; retain both ends.
            if inputs.input_ids.shape[1]>maxlen:
                for key,value in inputs.items():
                    inputs[key]=torch.cat([value[:,:maxlen//2],value[:,-(maxlen-maxlen//2):]],dim=1)
            if i<len(saved):
                record=saved[i]
                assert record['sample_id']==f'{task}:{i}'
                assert record['row_sha256']==hashlib.sha256(json.dumps(row,sort_keys=True).encode()).hexdigest()
                assert record['input_ids_sha256']==hashlib.sha256(inputs.input_ids.cpu().numpy().tobytes()).hexdigest()
                assert record['answers']==row['answers']
                predictions.append(record)
                overall_peak=max(overall_peak,record['peak_allocated_bytes'])
                continue
            length=inputs.input_ids.shape[1];maxnew=DATASET2MAXLEN[task]
            if length+maxnew>model.config.max_position_embeddings:
                raise ValueError('Input+output exceeds model position capacity')
            kwargs=dict(max_new_tokens=maxnew,num_beams=1,do_sample=False,pad_token_id=tokenizer.eos_token_id,use_cache=True)
            if task=='samsum':
                kwargs.update(min_length=length+1,eos_token_id=[tokenizer.eos_token_id,tokenizer.encode('\n',add_special_tokens=False)[-1]])
            output=model.generate(**inputs,**kwargs)[0]
            pred=tokenizer.decode(output[length:],skip_special_tokens=True)
            record={'sample_id':f'{task}:{i}','row_sha256':hashlib.sha256(json.dumps(row,sort_keys=True).encode()).hexdigest(),
                'input_ids_sha256':hashlib.sha256(inputs.input_ids.cpu().numpy().tobytes()).hexdigest(),
                'raw_prompt_tokens':raw_length,'actual_input_tokens':length,'generated_tokens':len(output)-length,
                'pred':pred,'answers':row['answers'],'all_classes':row['all_classes'],'length':row['length'],
                'seconds':time.time()-started,'peak_allocated_bytes':torch.cuda.max_memory_allocated()}
            overall_peak=max(overall_peak,record['peak_allocated_bytes'])
            predictions.append(record)
            with predfile.open('a') as f:f.write(json.dumps(record)+'\n')
            print(f'LongBench {task} {i+1}/{len(data)} cap={maxlen} input={length} elapsed={record["seconds"]:.2f}s',flush=True)
        scores[task]=scorer(task,[r['pred'] for r in predictions],[r['answers'] for r in predictions],predictions[0]['all_classes'])
        (out/'scores.json').write_text(json.dumps(scores,indent=2))
    return {'scores':scores,'input_cap':job['max_input'],'sample_limit':job.get('limit'),'peak_allocated_bytes':overall_peak,'protocol':'LongBench_v1_repo_templates_metrics_greedy'}

@torch.no_grad()
def memory_profile(model,tokenizer,job,out):
    import gc
    from palu.data_utils import wiki_split
    ids=tokenizer('\n\n'.join(wiki_split('test')['text']),return_tensors='pt').input_ids
    result=[]
    for length in job['lengths']:
        gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        baseline=torch.cuda.memory_allocated();start=time.perf_counter()
        cache=model.model(ids[:,:length].to('cuda'),use_cache=True,return_dict=True).past_key_values
        torch.cuda.synchronize();seconds=time.perf_counter()-start
        legacy=cache.to_legacy_cache() if hasattr(cache,'to_legacy_cache') else cache
        size=sum(t.numel()*t.element_size() for layer in legacy for t in layer)
        row={'input_tokens':length,'baseline_allocated_bytes':baseline,'kv_tensor_bytes':size,
             'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'post_prefill_allocated_bytes':torch.cuda.memory_allocated(),
             'prefill_seconds':seconds,'first_layer_shapes':[list(t.shape) for t in legacy[0]]}
        result.append(row);print('Memory',json.dumps(row),flush=True)
        del cache,legacy
    return {'measurements':result,'protocol':'decoder_body_prefill_returned_dense_KV_no_logits_batch1_fp16'}
