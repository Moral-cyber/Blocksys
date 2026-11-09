"""Fixed-state final TEST score/export contract; no optimizer or model import."""
from pathlib import Path
import json
import time

import numpy as np
import data as d

STATE_DIMENSIONS={'analytic_prior':10,'prioranchor_bias':10,'zero_anchor_bias':10,
    'temperature_scaling':1,'bcts':11,'zero_anchor_vector':20,
    'prioranchored_vector':20,'prioranchored_bcts':11}
VARIANTS=('raw','prior_only','analytic_prior','prioranchor_bias','zero_anchor_bias',
    'temperature_scaling','bcts','zero_anchor_vector','prioranchored_vector',
    'learned_head','learned_head_plus_prior','prioranchored_bcts')
METHODS=('ce_uniform3','fedlc_uniform3','fedlc_dense')
METRICS=('accuracy','nll','brier','ece15')


def metrics(logits,labels):
    z=np.asarray(logits,dtype=np.float64)
    y=np.asarray(labels)
    if (z.ndim!=2 or z.shape[1]!=10 or not len(z) or y.shape!=(len(z),)
            or y.dtype.kind not in 'iu' or np.any(y<0) or np.any(y>=10)
            or not np.isfinite(z).all()):
        raise ValueError('finite nonempty ten-class logits and integer GT required')
    maximum=z.max(axis=1)
    exponentials=np.exp(z-maximum[:,None])
    probabilities=exponentials/exponentials.sum(axis=1,keepdims=True)
    nll=maximum+np.log(exponentials.sum(axis=1))-z[np.arange(len(y)),y]
    correct=probabilities.argmax(axis=1)==y
    confidence=probabilities.max(axis=1)
    residual=probabilities.copy()
    residual[np.arange(len(y)),y]-=1.
    buckets=np.minimum((confidence*15).astype(np.int64),14)
    counts=np.bincount(buckets,minlength=15)
    confidence_sum=np.bincount(buckets,weights=confidence,minlength=15)
    correct_sum=np.bincount(buckets,weights=correct.astype(np.float64),minlength=15)
    return {'accuracy':float(correct.mean()),'nll':float(nll.mean()),'loss':float(nll.mean()),
        'brier':float(np.mean(np.sum(residual**2,axis=1))),
        'ece15':float(np.abs(correct_sum-confidence_sum).sum()/len(y)),
        'samples':len(y),'ece_bins':{'counts':counts.tolist(),
            'confidence_sum':confidence_sum.tolist(),'correct_sum':correct_sum.tolist()}}


def validate_states(states,train_counts):
    if not isinstance(states,(list,tuple)) or len(states)!=20:
        raise ValueError('all20 client TRAIN-fitted deployment-state dictionaries required')
    counts=d.validate_train_counts(train_counts)
    hashes=[]
    for client,state in enumerate(states):
        if set(state)!=set(STATE_DIMENSIONS):
            raise ValueError('complete fixed state families required, no added or missing family')
        for name,dimension in STATE_DIMENSIONS.items():
            array=np.asarray(state[name])
            if array.dtype!=np.float32 or array.shape!=(dimension,) or not np.isfinite(array).all():
                raise ValueError('exact finite FP32 deployed state required: '+name)
            if name in ('temperature_scaling','bcts','prioranchored_bcts') and array[0]<np.float32(.001):
                raise ValueError('positive scalar control below fixed0.001 bound')
        expected=np.log((counts[client]+1.)/(counts[client].sum()+10.)).astype(np.float32)
        if not np.array_equal(expected,state['analytic_prior']):
            raise ValueError('fixed analytic prior does not match local TRAIN counts')
        hashes.append({name:d.array_sha(value,'<f4') for name,value in state.items()})
    return hashes


def apply_states(raw,learned,states):
    z=np.asarray(raw,dtype=np.float64)
    h=np.asarray(learned,dtype=np.float64)
    if z.ndim!=2 or z.shape[1]!=10 or h.shape!=z.shape or not np.isfinite(z).all() or not np.isfinite(h).all():
        raise ValueError('finite paired raw and fixed-head logits required')
    state={name:np.asarray(value,dtype=np.float64) for name,value in states.items()}
    scores={'raw':z,'prior_only':np.broadcast_to(state['analytic_prior'],z.shape),
        'analytic_prior':z+state['analytic_prior'],'prioranchor_bias':z+state['prioranchor_bias'],
        'zero_anchor_bias':z+state['zero_anchor_bias'],
        'temperature_scaling':z*state['temperature_scaling'][0],
        'bcts':z*state['bcts'][0]+state['bcts'][1:],
        'zero_anchor_vector':z*state['zero_anchor_vector'][:10]+state['zero_anchor_vector'][10:],
        'prioranchored_vector':z*state['prioranchored_vector'][:10]+state['prioranchored_vector'][10:],
        'learned_head':h,'learned_head_plus_prior':h+state['analytic_prior'],
        'prioranchored_bcts':z*state['prioranchored_bcts'][0]+state['prioranchored_bcts'][1:]}
    if any(not np.isfinite(value).all() for value in scores.values()):
        raise ValueError('nonfinite deployed predictor scores')
    return scores


def summarize(clients):
    if len(clients)!=20 or [row['client'] for row in clients]!=list(range(20)):
        raise ValueError('all20 ordered clients required for macro score')
    weights=np.array([row['test_rows'] for row in clients],dtype=np.float64)
    if np.any(weights<=0) or weights.sum()!=10000:
        raise ValueError('all20 nonempty clients and exactly10000 TEST rows required')
    weights/=weights.sum()
    result={}
    for variant in VARIANTS:
        value={}
        for name in METRICS:
            numbers=np.array([row['metrics'][variant][name] for row in clients])
            value['macro_'+name]=float(numbers.mean())
            value['weighted_'+name]=float(numbers@weights)
        value['macro_loss']=value['macro_nll'];value['weighted_loss']=value['weighted_nll']
        value['client_accuracy_q10']=float(np.quantile([row['metrics'][variant]['accuracy'] for row in clients],.1))
        confidence=np.sum([row['metrics'][variant]['ece_bins']['confidence_sum'] for row in clients],axis=0)
        correct=np.sum([row['metrics'][variant]['ece_bins']['correct_sum'] for row in clients],axis=0)
        value['pooled_ece15']=float(np.abs(correct-confidence).sum()/10000)
        result[variant]=value
    return result


def _validate_assignment(labels,assignment):
    unsigned={key:value for key,value in assignment.items() if key!='manifest_hash'}
    if assignment.get('manifest_hash')!=d.canonical_sha(unsigned):
        raise ValueError('assignment manifest changed')
    if assignment['labels_int64le_sha256']!=d.array_sha(labels,'<i8'):
        raise ValueError('exact original TEST labels differ from assignment')
    groups=[np.asarray(ids,dtype=np.uint32) for ids in assignment['client_ids']]
    if d.assignment_digest(groups)!=assignment['assignment_hash']:
        raise ValueError('assignment ordered-ID hash differs')
    replay,recreated=d.assign_clients(labels,np.asarray(assignment['train_counts'],dtype=np.int64),
        task=assignment['task'],train_manifest_sha256=assignment['train_manifest_sha256'])
    if recreated['assignment_hash']!=assignment['assignment_hash']:
        raise ValueError('assignment does not follow fixed deterministic TRAIN-count rule')
    return groups


def score_export(out,*,task,seed,method,row_ids,raw_logits,head_logits_by_client,
        labels,assignment,states,test_provenance,input_snapshot,
        head_row_ids_by_client=None,root_lock=None,execute=False,synthetic=False,inference_cost=None):
    """Inference driver supplies raw rows0..9999 and each assigned-client head.

    states must be exported from TRAIN before any TEST access. input_snapshot
    binds checkpoint, fitted-state files, TRAIN manifest and protocol sources.
    This function never fits, imports a model, or chooses a predictor.
    """
    out=Path(out).resolve()
    if out.exists():
        raise ValueError('fresh final export directory required')
    if task not in d.TASKS or seed not in d.TASKS[task]['seeds'] or method not in METHODS:
        raise ValueError('fixed task/three seeds/training method required')
    authorization=None if synthetic else d.validate_root_lock(root_lock,execute,task)
    if synthetic and test_provenance.get('scope')!='SYNTHETIC_CPU_FIXTURE':
        raise ValueError('synthetic scope must be explicit; never relabel official final evaluation')
    if not synthetic and (test_provenance.get('task')!=task or test_provenance.get('official_split')!='official TEST only'):
        raise ValueError('strict official loader provenance required')
    if not input_snapshot or any(d.file_sha(path)!=expected for path,expected in input_snapshot.items()):
        raise ValueError('prelocked final-model/TRAIN-state/source snapshot required and unchanged')
    y=d._labels(labels)
    if not synthetic and test_provenance.get('labels_int64le_sha256')!=d.array_sha(y,'<i8'):
        raise ValueError('official label source provenance differs from scored GT')
    if assignment['task']!=task or assignment['partition_seed']!=d.TASKS[task]['partition_seed']:
        raise ValueError('registered task partition mismatch')
    groups=_validate_assignment(y,assignment)
    if not np.array_equal(np.asarray(row_ids),np.arange(10000)):
        raise ValueError('raw TEST logits must follow exact original IDs0..9999')
    z=np.asarray(raw_logits)
    if z.dtype not in (np.float32,np.float64) or z.shape!=(10000,10) or not np.isfinite(z).all():
        raise ValueError('finite10000x10 raw final-checkpoint logits required')
    if not isinstance(head_logits_by_client,(list,tuple)) or len(head_logits_by_client)!=20:
        raise ValueError('all20 fixed TRAIN-adapted head score exports required')
    if head_row_ids_by_client is None or len(head_row_ids_by_client)!=20 or any(
            not np.array_equal(np.asarray(head_ids),ids) for head_ids,ids in zip(head_row_ids_by_client,groups)):
        raise ValueError('explicit head score original IDs must match each assigned client order')
    for ids,head in zip(groups,head_logits_by_client):
        if np.asarray(head).shape!=(len(ids),10) or not np.isfinite(head).all():
            raise ValueError('head score order/shape must match assigned client original IDs')
    state_hashes=validate_states(states,assignment['train_counts'])
    sources={str(HERE/name):d.file_sha(HERE/name) for name in ('FINAL_TEST_PLAN.md','data.py','scorer.py')}
    out.mkdir(parents=True)
    (out/'assignment.json').write_text(json.dumps(assignment,indent=2,allow_nan=False)+'\n')
    (out/'input_snapshot.json').write_text(json.dumps(input_snapshot,indent=2)+'\n')
    started=time.perf_counter();clients=[];files=[]
    transform_seconds=metric_seconds=export_seconds=0.
    for client,ids in enumerate(groups):
        t=time.perf_counter()
        scores=apply_states(z[ids],head_logits_by_client[client],states[client])
        transform_seconds+=time.perf_counter()-t
        t=time.perf_counter()
        observed={name:metrics(value,y[ids]) for name,value in scores.items()}
        metric_seconds+=time.perf_counter()-t
        t=time.perf_counter()
        target=out/f'client{client}.npz'
        np.savez_compressed(target,test_ids=ids,test_labels=y[ids],raw_logits=z[ids],
            fixed_head_logits=np.asarray(head_logits_by_client[client]),
            **{'state_'+name:value for name,value in states[client].items()},
            **{'score_'+name:value for name,value in scores.items()})
        export_seconds+=time.perf_counter()-t
        files.append({'path':target.name,'bytes':target.stat().st_size,'sha256':d.file_sha(target)})
        clients.append({'client':client,'test_rows':len(ids),'metrics':observed,
            'deployment_state_hashes':state_hashes[client]})
    if (validate_states(states,assignment['train_counts'])!=state_hashes
            or any(d.file_sha(path)!=expected for path,expected in {**input_snapshot,**sources}.items())):
        raise ValueError('source/checkpoint/TRAIN-state input changed during final export')
    report={'status':'DONE','scope':'SYNTHETIC_CPU_FIXTURE' if synthetic else 'LOCKED_OFFICIAL_FINAL_TEST',
        'task':task,'seed':seed,'method':method,'partition_seed':d.TASKS[task]['partition_seed'],
        'rounds':20,'assignment_hash':assignment['assignment_hash'],
        'assignment_manifest_hash':assignment['manifest_hash'],'clients':clients,'summaries':summarize(clients),
        'test_provenance':test_provenance,'input_snapshot_sha256_before_after_equal':input_snapshot,
        'source_identities':sources,'authorization':authorization,'exports':files,
        'fixed_variants':list(VARIANTS),'state_bytes':{name:dimension*4 for name,dimension in STATE_DIMENSIONS.items()},
        'scoring_arithmetic':'exact FP32 deployment states promoted to FP64; no FP64 optimizer state substitution',
        'test_protocol_scope':'TRAIN-count-defined new label-skew client assignment, not official benchmark client partition or exact TRAIN prior matching',
        'metric_definitions':{'nll':'mean categorical negative log likelihood',
            'brier':'mean sample sum of ten class squared probability errors',
            'ece15':'15 equal-width top-label confidence bins, last includes1, empty0',
            'weighted_ece15':'sample-weighted mean client ECE; pooled ECE additionally reported'},
        'cost':{'new_fit_calls':0,'new_cnn_backwards':0,'new_training_rounds':0,
            'inference_driver_cost':inference_cost or {'status':'NOT_SUPPLIED'},
            'transform_seconds':transform_seconds,'metric_seconds':metric_seconds,'export_seconds':export_seconds,
            'scorer_elapsed_seconds':time.perf_counter()-started,'npz_bytes':sum(row['bytes'] for row in files),
            'raw_logit_bytes':z.nbytes,'scope':'local observed timings; no network/energy measurement'},
        'runtime':d.runtime(),'judgment':'no final-test gate or method/seed selection in scorer; report all fixed results'}
    (out/'result.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report


def summarize_three_seeds(reports,task,method):
    """Paired comparisons/mean/sample SD only, with no test-driven decision."""
    expected=d.TASKS[task]['seeds']
    if (len(reports)!=3 or sorted(row['seed'] for row in reports)!=expected
            or any(row['task']!=task or row['method']!=method or row['status']!='DONE' for row in reports)
            or len({row['assignment_hash'] for row in reports})!=1):
        raise ValueError('exactly the three fixed paired seeds and identical assignment required')
    aggregate={}
    for variant in VARIANTS:
        aggregate[variant]={key:{'mean':float(np.mean(values)),'sample_sd':float(np.std(values,ddof=1))}
            for key in reports[0]['summaries'][variant]
            for values in [[row['summaries'][variant][key] for row in reports]]}
    differences={}
    for control in VARIANTS:
        if control=='prioranchored_vector':continue
        differences[control]={name:{'mean':float(np.mean(values)),'sample_sd':float(np.std(values,ddof=1))}
            for name in ('macro_accuracy','macro_nll','macro_brier','macro_ece15')
            for values in [[row['summaries']['prioranchored_vector'][name]-row['summaries'][control][name] for row in reports]]}
    return {'task':task,'method':method,'seeds':expected,'assignment_hash':reports[0]['assignment_hash'],
        'summaries':aggregate,'paired_main_minus_control':differences,'sd_definition':'sample SD, ddof1, n3',
        'judgment':'descriptive fixed-three-seed summary; no selection or gate'}


HERE=Path(__file__).resolve().parent
