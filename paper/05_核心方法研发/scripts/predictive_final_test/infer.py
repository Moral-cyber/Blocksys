"""Freeze TRAIN-fitted predictors, then infer once on official TEST; no fitting."""
from pathlib import Path
import argparse
from datetime import datetime
import importlib.util
import json
import sys
import time

import numpy as np
import torch

import data as d
import scorer

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
MODEL_SOURCE=ROOT/'work/tinyproto_diag/tools/fedavg_update_diagnostic.py'

def model_class():
    spec=importlib.util.spec_from_file_location('_final_test_model',MODEL_SOURCE)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SmallCNN

def paths(task,seed):
    partition=d.TASKS[task]['partition_seed']
    tag=f'{task}_p{partition}_s{seed}'
    return (ROOT/f'work/predictive_pfl_confirm/development_{tag}',
            HERE/f'train_states_{tag}')

def inputs_for_task(task):
    inputs={str(path):d.file_sha(path) for path in
        (Path(__file__).resolve(),MODEL_SOURCE,HERE/'data.py',HERE/'scorer.py',
         HERE/'FINAL_TEST_PLAN.md',HERE/'export_train_states.py')}
    for seed in d.TASKS[task]['seeds']:
        model_root,state_root=paths(task,seed)
        exported=json.loads((state_root/'result.json').read_text())
        if (exported.get('status')!='DONE' or exported.get('task')!=task
                or exported.get('training_seed')!=seed
                or exported.get('partition_seed')!=d.TASKS[task]['partition_seed']
                or exported.get('rounds')!=20 or exported.get('methods')!=list(scorer.METHODS)):
            raise ValueError('All TRAIN state exports must complete before TEST: '+str(state_root))
        manifest=json.loads((state_root/'manifest.json').read_text())
        counts=d.validate_train_counts(np.load(state_root/'train_counts.npy',allow_pickle=False))
        if not np.array_equal(counts,np.asarray(manifest['client_label_counts'],dtype=np.int64)):
            raise ValueError('Exported TRAIN counts differ from exact manifest')
        if (exported['train_counts_file']['sha256']!=d.file_sha(state_root/'train_counts.npy')
                or exported['train_manifest_file']['sha256']!=d.file_sha(state_root/'manifest.json')
                or exported.get('forbidden_official_test_open_attempts')!=[]
                or exported.get('new_calibration_fits')!=0):
            raise ValueError('TRAIN exporter provenance or output hashes differ')
        for item in exported['exports']:
            if d.file_sha(state_root/item['path'])!=item['sha256']:
                raise ValueError('TRAIN state bytes differ from export report')
        for path in (state_root/'result.json',state_root/'manifest.json',state_root/'train_counts.npy'):
            inputs[str(path)]=d.file_sha(path)
        for method in scorer.METHODS:
            config=json.loads((model_root/method/'config.json').read_text())
            if (config['data_manifest']!=manifest
                    or exported['results'][method]['checkpoint_sha256']!=d.file_sha(model_root/method/'checkpoint.pt')
                    or exported['results'][method]['config_signature']!=config['signature']):
                raise ValueError('State export does not bind exact method/checkpoint/TRAIN partition')
            for name in ('checkpoint.pt','config.json','result.json'):
                path=model_root/method/name
                inputs[str(path)]=d.file_sha(path)
            for client in range(20):
                path=state_root/f'{method}_client{client}.npz'
                inputs[str(path)]=d.file_sha(path)
    return inputs

def freeze(task,lock_path):
    lock_path=Path(lock_path).resolve()
    if lock_path.exists():
        raise ValueError('Never overwrite a final TEST execution lock')
    snapshot=inputs_for_task(task)
    lock={'status':'ROOT_LOCKED_FINAL_TEST','tasks':[task],
        'created_at':datetime.now().astimezone().isoformat(),
        'plan_sha256':d.file_sha(HERE/'FINAL_TEST_PLAN.md'),
        'source_sha256':{name:d.file_sha(HERE/name) for name in ('data.py','scorer.py')},
        'frozen_input_snapshot':snapshot,
        'method_selection':'predeclared main method and all controls; no TEST-based retuning',
        'scope':'all three fixed final20-round models/TRAIN-fitted states for this task'}
    lock_path.write_text(json.dumps(lock,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'status':'ROOT_LOCKED_FINAL_TEST','task':task,'inputs':len(snapshot),'lock':str(lock_path)}),flush=True)

def forward(model,images,task,batch_size=256):
    model.eval()
    raw=[];features=[]
    with torch.inference_mode():
        for begin in range(0,len(images),batch_size):
            x=torch.from_numpy(d.normalize_batch(images[begin:begin+batch_size],task))
            h=model.classifier[:-1](model.features(x))
            z=model.classifier[-1](h)
            if h.shape[1]!=64 or z.shape[1]!=10:
                raise ValueError('Frozen SmallCNN feature/class layout differs')
            raw.append(z.numpy().copy());features.append(h.numpy().copy())
    return np.concatenate(raw),np.concatenate(features)

def run(task,lock_path,out):
    out=Path(out).resolve()
    if out.exists():
        raise ValueError('Fresh final task output required')
    authorization=d.validate_root_lock(lock_path,True,task)
    locked=json.loads(Path(lock_path).read_text())
    snapshot=inputs_for_task(task)
    if snapshot!=locked.get('frozen_input_snapshot'):
        raise ValueError('Frozen TRAIN/model/source snapshot differs before TEST')
    torch.set_num_threads(1)
    SmallCNN=model_class()
    # TEST source loading happens only after every state/model/source is frozen.
    began=time.perf_counter()
    if task=='cifar':
        raw=ROOT/'work/tinyproto_diag/dataset/Cifar10/rawdata'
        images,labels,provenance=d.load_cifar_test(raw,root_lock=lock_path,execute=True)
    else:
        raw=ROOT/'work/method_followup/datasets/FashionMNIST/raw'
        images,labels,provenance=d.load_fashion_test(raw,root_lock=lock_path,execute=True)
    load_seconds=time.perf_counter()-began
    scoring_snapshot={**snapshot,**{item['path']:item['sha256'] for item in provenance['files']}}
    _,first_states=paths(task,d.TASKS[task]['seeds'][0])
    counts=np.load(first_states/'train_counts.npy',allow_pickle=False)
    groups,assignment=d.assign_clients(labels,counts,task=task,
        train_manifest_sha256=d.file_sha(first_states/'manifest.json'))
    out.mkdir(parents=True)
    (out/'assignment.json').write_text(json.dumps(assignment,indent=2)+'\n')
    reports={method:[] for method in scorer.METHODS}
    for seed in d.TASKS[task]['seeds']:
        model_root,state_root=paths(task,seed)
        if not np.array_equal(counts,np.load(state_root/'train_counts.npy',allow_pickle=False)):
            raise ValueError('All seeds must share exact TRAIN counts')
        for method in scorer.METHODS:
            config=json.loads((model_root/method/'config.json').read_text())
            checkpoint=torch.load(model_root/method/'checkpoint.pt',map_location='cpu',weights_only=False)
            if (config['seed']!=seed or config['partition_seed']!=d.TASKS[task]['partition_seed']
                    or config['rounds']!=20 or config['method']!=method
                    or checkpoint['config']!=config or len(checkpoint['rows'])!=20
                    or checkpoint['signature']!=config['signature']):
                raise ValueError('Only complete fixed final checkpoints allowed')
            model=SmallCNN();model.load_state_dict(checkpoint['model']);model.eval()
            started=time.perf_counter()
            logits,features=forward(model,images,task)
            states=[];head_logits=[]
            for client,ids in enumerate(groups):
                with np.load(state_root/f'{method}_client{client}.npz',allow_pickle=False) as saved:
                    if not np.array_equal(saved['train_counts'],counts[client]):
                        raise ValueError('Exported client state TRAIN counts differ')
                    states.append({name:saved['state_'+name].copy() for name in scorer.STATE_DIMENSIONS})
                    weight=torch.from_numpy(saved['head_weight'].copy())
                    bias=torch.from_numpy(saved['head_bias'].copy())
                    if (weight.dtype!=torch.float32 or bias.dtype!=torch.float32
                            or tuple(weight.shape)!=(10,64) or tuple(bias.shape)!=(10,)):
                        raise ValueError('Exact finite FP32 last-Linear head layout required')
                with torch.inference_mode():
                    prediction=torch.nn.functional.linear(torch.from_numpy(features[ids]),weight,bias)
                    head_logits.append(prediction.numpy().copy())
            inference_seconds=time.perf_counter()-started
            report=scorer.score_export(out/f'{method}_s{seed}',task=task,seed=seed,method=method,
                row_ids=np.arange(10000),raw_logits=logits,head_logits_by_client=head_logits,
                head_row_ids_by_client=groups,labels=labels,assignment=assignment,states=states,
                test_provenance=provenance,input_snapshot=scoring_snapshot,root_lock=lock_path,execute=True,
                inference_cost={'cnn_forward_samples':10000,'raw_forward_batches':40,
                    'head_forward_samples':10000,'cached_features_bytes':features.nbytes,
                    'load_seconds':load_seconds,'inference_seconds':inference_seconds,
                    'backend':'CPU','torch_version':torch.__version__,
                    'class_order':list(range(10)),
                    'scope':'shared frozen features reused by fixed client heads; no training/fit/backward'})
            reports[method].append(report)
            print(json.dumps({'status':'DONE','task':task,'seed':seed,'method':method,
                'candidate':report['summaries']['prioranchored_vector']}),flush=True)
    summaries={method:scorer.summarize_three_seeds(items,task,method) for method,items in reports.items()}
    if any(d.file_sha(path)!=expected for path,expected in scoring_snapshot.items()):
        raise ValueError('Frozen input changed during final TEST')
    result={'status':'DONE','scope':'LOCKED_OFFICIAL_FINAL_TEST','task':task,
        'fixed_three_seeds':d.TASKS[task]['seeds'],'training_methods':list(scorer.METHODS),
        'assignment_hash':assignment['assignment_hash'],'summaries':summaries,
        'authorization':authorization,'input_snapshot':scoring_snapshot,
        'selection':'no TEST-driven method, seed, checkpoint or parameter selection'}
    (out/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'status':'FINAL_TASK_COMPLETE','task':task}),flush=True)

def main():
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument('--task',choices=d.TASKS,required=True)
    parser.add_argument('--root-lock',type=Path,required=True)
    parser.add_argument('--freeze',action='store_true')
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--out',type=Path)
    args=parser.parse_args()
    if args.freeze==args.execute:
        parser.error('Exactly one of --freeze or --execute is required')
    if args.freeze:
        freeze(args.task,args.root_lock)
    elif args.out is None:
        parser.error('--out required for final inference')
    else:
        run(args.task,args.root_lock,args.out)

if __name__=='__main__':
    main()
