"""Decision-free TRAIN-only calibration and predictive metrics on saved arrays."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import sys
import time
import tracemalloc

import numpy as np
import solver as s

HERE=Path(__file__).resolve().parent
SOLVER_SHA256='9c3196eb3e6ff7f5f32073e3bc4aae8d986ac7e44bdf0d719258343847d4dbae'
READER=HERE.parent/'calibration_baselines/probe.py'
READER_SHA256='d1fab28871824163a6d2fc2ee44e043698537b00dabc0e02af2618531adcd0c7'
OLD_MODELS=('ce_uniform3','fedlc_uniform3','balanced_uniform3','balanced_dense')
CONFIRM_MODELS=('ce_uniform3','fedlc_uniform3','fedlc_dense')
ECE_BINS=15
METRICS=('accuracy','nll','brier','ece15')
VARIANTS=('raw','analytic_prior','prioranchor_bias','zero_anchor_vector',
    'prioranchored_vector','learned_head','learned_head_plus_prior')


def atomic_json(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    temporary.replace(path)


def read_saved_function():
    if s.sha(s.__file__)!=SOLVER_SHA256 or s.sha(READER)!=READER_SHA256:
        raise ValueError('frozen solver/array reader identity mismatch')
    spec=importlib.util.spec_from_file_location('_predictive_saved_array_reader',READER)
    module=importlib.util.module_from_spec(spec)
    previous=sys.modules.get('solver')
    sys.modules['solver']=s.base
    try:
        spec.loader.exec_module(module)
    finally:
        if previous is None:
            del sys.modules['solver']
        else:
            sys.modules['solver']=previous
    return module.read_saved


def predictive_metrics(logits,labels):
    z=np.asarray(logits,dtype=np.float64)
    y=np.asarray(labels)
    if (z.ndim!=2 or z.shape[1]!=10 or not len(z) or y.shape!=(len(z),)
            or y.dtype.kind not in 'iu' or np.any(y<0) or np.any(y>=10)
            or not np.isfinite(z).all()):
        raise ValueError('finite nonempty ten-class logits and integer labels required')
    maximum=z.max(axis=1)
    exp=np.exp(z-maximum[:,None])
    p=exp/exp.sum(axis=1,keepdims=True)
    nll=maximum+np.log(exp.sum(axis=1))-z[np.arange(len(y)),y]
    prediction=p.argmax(axis=1)
    correct=prediction==y
    confidence=p.max(axis=1)
    residual=p.copy()
    residual[np.arange(len(y)),y]-=1.
    brier=np.sum(residual**2,axis=1)
    # Fixed equal-width [k/15,(k+1)/15) bins; the last includes confidence 1.
    bucket=np.minimum((confidence*ECE_BINS).astype(np.int64),ECE_BINS-1)
    counts=np.bincount(bucket,minlength=ECE_BINS)
    confidence_sum=np.bincount(bucket,weights=confidence,minlength=ECE_BINS)
    correct_sum=np.bincount(bucket,weights=correct.astype(np.float64),minlength=ECE_BINS)
    ece=float(np.abs(correct_sum-confidence_sum).sum()/len(y))
    bins=[{'index':k,'lower':k/ECE_BINS,'upper':(k+1)/ECE_BINS,
        'count':int(counts[k]),'confidence_sum':float(confidence_sum[k]),
        'correct_sum':int(correct_sum[k])} for k in range(ECE_BINS)]
    return {'accuracy':float(correct.mean()),'nll':float(nll.mean()),
        'loss':float(nll.mean()),'brier':float(brier.mean()),'ece15':ece,
        'samples':len(y),'ece15_bins':bins}


def summarize(clients):
    weights=np.array([row['holdout_samples'] for row in clients],dtype=np.float64)
    weights/=weights.sum()
    result={}
    for variant in VARIANTS:
        summary={}
        for name in METRICS:
            values=np.array([row['metrics'][variant][name] for row in clients])
            summary['macro_'+name]=float(values.mean())
            summary['weighted_'+name]=float(values@weights)
        summary['macro_loss']=summary['macro_nll']
        summary['weighted_loss']=summary['weighted_nll']
        summary['client_accuracy_q10']=float(np.quantile(
            [row['metrics'][variant]['accuracy'] for row in clients],.1))
        pooled_counts=np.sum([[entry['count'] for entry in row['metrics'][variant]['ece15_bins']]
            for row in clients],axis=0)
        pooled_confidence=np.sum([[entry['confidence_sum'] for entry in row['metrics'][variant]['ece15_bins']]
            for row in clients],axis=0)
        pooled_correct=np.sum([[entry['correct_sum'] for entry in row['metrics'][variant]['ece15_bins']]
            for row in clients],axis=0)
        summary['pooled_ece15']=float(np.abs(pooled_correct-pooled_confidence).sum()/pooled_counts.sum())
        result[variant]=summary
    return result


def run(input_root,out,protocol_plan,protocol_registration=None):
    input_root,out,protocol_plan=map(lambda p:Path(p).resolve(),(input_root,out,protocol_plan))
    if out.exists():
        raise ValueError('fresh evaluator output required')
    attempts=s.install_data_guard()
    read_saved=read_saved_function()
    # PLAN is provenance data; this module never executes its text or decides
    # research gates. The external stage gate helper owns every judgment.
    protocol_text=protocol_plan.read_text()
    if not protocol_text.strip():
        raise ValueError('nonempty explicit protocol PLAN required')
    source_paths=[Path(__file__).resolve(),Path(s.__file__).resolve(),s.BASE_PATH,READER,protocol_plan]
    if protocol_registration is not None:
        source_paths.append(Path(protocol_registration).resolve())
    sources={str(path):s.sha(path) for path in source_paths}
    source_result=input_root/'result.json'
    inputs={str(source_result):s.sha(source_result)}
    existing=json.loads(source_result.read_text())
    keys=set(existing.get('results',{}))
    models=CONFIRM_MODELS if keys==set(CONFIRM_MODELS) else OLD_MODELS if keys==set(OLD_MODELS) else None
    if existing.get('status')!='DONE' or models is None:
        raise ValueError('complete DONE confirm-three or old-four saved-array schema required')
    tracemalloc.start()
    all_saved={}
    source_bytes=selected_bytes=0
    for method in models:
        if len(existing['results'][method]['clients'])!=20:
            raise ValueError('all 20 clients must be present for every model')
        for client in range(20):
            path=input_root/f'{method}_client{client}.npz'
            inputs[str(path)]=s.sha(path)
            saved,bytes_cached=read_saved(path)
            if saved['learned'] is None:
                raise ValueError('matched learned-head saved scores required')
            if method!=models[0]:
                reference=all_saved[(models[0],client)]
                for name in ('train_ids','train_labels','holdout_ids','labels','prior'):
                    if not np.array_equal(saved[name],reference[name]):
                        raise ValueError('models do not share identical client TRAIN/holdout membership and labels')
            all_saved[(method,client)]=saved
            source_bytes+=bytes_cached
            selected_bytes+=sum(value.nbytes for value in saved.values() if isinstance(value,np.ndarray))
    first_ids=np.concatenate([np.r_[all_saved[(models[0],client)]['train_ids'],
        all_saved[(models[0],client)]['holdout_ids']] for client in range(20)])
    if len(np.unique(first_ids))!=len(first_ids):
        raise ValueError('client memberships overlap or duplicate original IDs')
    out.mkdir(parents=True)
    atomic_json(out/'inputs_before.json',{**sources,**inputs})
    results={}
    totals={'prior_vector_fits':0,'zero_vector_fits':0,
        'prior_vector_steps':0,'zero_vector_steps':0,'prior_vector_fit_seconds':0.,
        'zero_vector_fit_seconds':0.,'train_rows_used_per_recipe':0,
        'source_arrays_uncompressed_bytes':source_bytes,'selected_input_array_bytes_cached':selected_bytes,
        'peak_single_design_cache_bytes':0,'new_cnn_forwards':0,'new_cnn_backwards':0,'extra_uplink_bytes':0}
    start=time.perf_counter()
    for method in models:
        clients=[]
        for client in range(20):
            saved=all_saved[(method,client)]
            anchor=s.train_prior_anchor(saved['train_labels'])
            prior_vector,prior_optimization=s.fit(saved['train_logits'],saved['train_labels'],anchor)
            zero_vector,zero_optimization=s.base.fit(saved['train_logits'],saved['train_labels'],'vector_scaling')
            # State A,b are float32 once; every final score applies these exact
            # deployed values with float64 arithmetic, including the controls.
            raw=saved['raw'].astype(np.float64)
            learned=saved['learned'].astype(np.float64)
            scores={'raw':raw,'analytic_prior':raw+saved['prior'],
                'prioranchor_bias':raw+saved['prioranchor_bias'],
                'zero_anchor_vector':s.transform(raw,zero_vector),
                'prioranchored_vector':s.transform(raw,prior_vector),
                'learned_head':learned,'learned_head_plus_prior':learned+saved['prior']}
            metrics={name:predictive_metrics(value,saved['labels']) for name,value in scores.items()}
            original_client=existing['results'][method]['clients'][client]
            for variant,old_metric in original_client['metrics'].items():
                key='prioranchor_bias' if variant=='fitted_bias' else variant
                if key in metrics:
                    old_nll=old_metric.get('nll',old_metric.get('loss'))
                    if (abs(metrics[key]['accuracy']-old_metric['accuracy'])>1e-12
                            or old_nll is None or abs(metrics[key]['nll']-old_nll)>1e-9):
                        raise ValueError('saved original baseline metrics do not reproduce')
            np.savez_compressed(out/f'{method}_client{client}.npz',
                train_ids=saved['train_ids'],holdout_ids=saved['holdout_ids'],
                train_bias_anchor=anchor,prioranchored_vector=prior_vector,
                zero_anchor_vector=zero_vector,prioranchor_bias=saved['prioranchor_bias'],
                prioranchored_vector_holdout_logits=scores['prioranchored_vector'],
                zero_anchor_vector_holdout_logits=scores['zero_anchor_vector'])
            inherited={key:value for key,value in original_client.items()
                if key not in ('metrics','client','train_samples','holdout_samples','optimization')}
            clients.append({'client':client,'train_samples':len(saved['train_labels']),
                'holdout_samples':len(saved['labels']),'metrics':metrics,
                'prioranchored_vector_optimization':prior_optimization,
                'zero_anchor_vector_optimization':zero_optimization,
                'state_bytes':{'prioranchored_vector':80,'zero_anchor_vector':80,'prioranchor_bias':40},
                'incremental_cost':{'new_cnn_forwards':0,'new_cnn_backwards':0,'extra_uplink_bytes':0},
                'inherited_source_client_cost_and_metadata':inherited})
            totals['prior_vector_fits']+=1
            totals['zero_vector_fits']+=1
            totals['prior_vector_steps']+=prior_optimization['accepted_steps']
            totals['zero_vector_steps']+=zero_optimization['accepted_steps']
            totals['prior_vector_fit_seconds']+=prior_optimization['seconds']
            totals['zero_vector_fit_seconds']+=zero_optimization['seconds']
            totals['train_rows_used_per_recipe']+=len(saved['train_labels'])
            totals['peak_single_design_cache_bytes']=max(totals['peak_single_design_cache_bytes'],
                prior_optimization['design_cache_bytes'])
        results[method]={'clients':clients,'summaries':summarize(clients)}
        print(json.dumps({'model':method,'summaries':results[method]['summaries']}),flush=True)
    current,peak=tracemalloc.get_traced_memory()
    tracemalloc.stop()
    totals.update({'tracemalloc_observed_current_bytes':current,'tracemalloc_observed_peak_bytes':peak,
        'memory_scope':'traced Python/NumPy input, fitting and output allocations; not process RSS or exact peak bound'})
    frozen={**sources,**inputs}
    if any(s.sha(path)!=digest for path,digest in frozen.items()) or attempts:
        raise ValueError('source/input changed or dataset/TEST access attempted')
    source_cost_keys=('bias_total_train_image_forwards','bias_total_cnn_backward',
        'matched_head_total_gradient_evaluations','head_total_gradient_evaluations',
        'official_test_byte_read_attempts','partition_seed','seed','task','model_root')
    report={'status':'DONE','schema':'prior_vector_saved_predictive_v1','results':results,
        'protocol_plan':str(protocol_plan),'protocol_plan_sha256':sources[str(protocol_plan)],
        'protocol_registration':None if protocol_registration is None else str(Path(protocol_registration).resolve()),
        'judgment_scope':'no gate or decision in this evaluator; a separately frozen protocol and stage helper own judgments. Historical STOPs are not revised.',
        'input_root':str(input_root),'input_schema':'confirm_three' if models==CONFIRM_MODELS else 'old_four',
        'input_task':existing.get('task','not specified by source result'),
        'source_identities':sources,'input_sha256_before_after_equal':frozen,
        'source_result_metadata_and_collection_cost':{key:existing[key] for key in source_cost_keys if key in existing},
        'metric_definitions':{'nll':'mean per-sample categorical negative log likelihood; loss is an exact alias',
            'brier':'mean per-sample sum over 10 classes of (p-onehot)^2; not divided by number of classes',
            'ece15':'top-label ECE, fixed 15 equal-width confidence bins [k/15,(k+1)/15); last includes 1; empty bins contribute zero',
            'macro':'unweighted mean over all 20 clients','weighted':'mean weighted by clientholdout sample counts; weighted ECE is mean client ECE',
            'pooled_ece15':'ECE after merging client bin counts/confidence/correct sums; distinct from mean client ECE'},
        'score_application':'saved FP32 A,b promoted to FP64 with saved raw logits; optimization FP64 theta is diagnostic only; same arithmetic across vector controls',
        'label_usage':'only saved TRAIN labels enter prior anchor and both fits; holdout labels only final metrics',
        'recipe':'sum TRAIN CE + ridge10 on both A-anchor1 and b-anchorcenter(log(TRAINcount+1)); zero-vector changes only b-anchor to zero',
        'deployment_state_bytes':{'prioranchored_vector':80,'zero_anchor_vector':80,'prioranchor_bias':40},
        'cost_scope':'states and logits already saved; incremental NumPy solves/cache reported; inherited image/head collection costs separately preserved; no network timing claim',
        'known_method_scope':'known vector scaling plus prior regularization; no atomic novelty or official FedCal reproduction claim',
        'new_cnn_forwards':0,'new_cnn_backwards':0,'new_dataset_bytes':0,'extra_uplink_bytes':0,
        'forbidden_dataset_open_attempts':attempts,'totals':totals,'elapsed_seconds':time.perf_counter()-start,
        'numpy':np.__version__,'python':sys.version}
    atomic_json(out/'result.json',report)
    print(json.dumps({'status':'DONE','models':list(models),'gates_evaluated':False,'totals':totals}),flush=True)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument('--input-root',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--protocol-plan',type=Path,required=True)
    parser.add_argument('--protocol-registration',type=Path)
    args=parser.parse_args()
    run(args.input_root,args.out,args.protocol_plan,args.protocol_registration)


if __name__=='__main__':
    main()
