"""Fit only the 11d control; reuse paired frozen 20d states on fresh saved arrays."""
from pathlib import Path
import argparse
import json
import time
import tracemalloc
import types

import numpy as np
import bcts_control as c
import evaluate_saved as e

HERE=Path(__file__).resolve().parent
ADDENDUM_SHA256='9ea93ea8496f33f4f7330a1ea7a3764e3aa101432c04c0bb916903370116d1cc'
EVALUATOR_SHA256='7b45a055df8c3e03421fd9edd6749c1e2bfba1203864e0b41695413db1fb8b05'
PROTOCOL_SHA256='703d5956c9edd24d5a03501a75f678a43222d19bd3dd1ede515ed77545b0830b'
VARIANTS=(*e.VARIANTS,'prioranchored_bcts')


def summarize(clients):
    namespace=dict(e.summarize.__globals__)
    namespace['VARIANTS']=VARIANTS
    bound=types.FunctionType(e.summarize.__code__,namespace,e.summarize.__name__,
        e.summarize.__defaults__,e.summarize.__closure__)
    return bound(clients)


def run(input_root,vector_root,out,protocol_plan,protocol_registration=None):
    input_root,vector_root,out,protocol_plan=map(lambda p:Path(p).resolve(),
        (input_root,vector_root,out,protocol_plan))
    if out.exists():
        raise ValueError('fresh control output required')
    attempts=c.s.install_data_guard()
    if c.s.sha(HERE/'BCTS_CONTROL_PLAN.md')!=ADDENDUM_SHA256 or c.s.sha(e.__file__)!=EVALUATOR_SHA256:
        raise ValueError('fixed control addendum or main evaluator identity changed')
    if c.s.sha(protocol_plan)!=PROTOCOL_SHA256:
        raise ValueError('this control requires the declared prospective PLAN')
    read_saved=e.read_saved_function()
    sources={str(path):c.s.sha(path) for path in (
        HERE/'BCTS_CONTROL_PLAN.md',HERE/'bcts_control.py',Path(__file__).resolve(),
        Path(c.s.__file__).resolve(),c.s.BASE_PATH,Path(e.__file__).resolve(),e.READER,protocol_plan)}
    if protocol_registration is not None:
        registration=Path(protocol_registration).resolve()
        sources[str(registration)]=c.s.sha(registration)
    source_result=input_root/'result.json'
    vector_result=vector_root/'result.json'
    existing=json.loads(source_result.read_text())
    vector=json.loads(vector_result.read_text())
    inputs={str(source_result):c.s.sha(source_result),str(vector_result):c.s.sha(vector_result)}
    if (existing.get('status')!='DONE' or vector.get('status')!='DONE'
            or set(existing.get('results',{}))!=set(e.CONFIRM_MODELS)
            or set(vector.get('results',{}))!=set(e.CONFIRM_MODELS)
            or vector.get('schema')!='prior_vector_saved_predictive_v1'):
        raise ValueError('complete source confirm-three and locked vector evaluator output required')
    if (Path(vector.get('input_root','')).resolve()!=input_root
            or vector.get('protocol_plan_sha256')!=PROTOCOL_SHA256
            or vector.get('input_sha256_before_after_equal',{}).get(str(source_result))!=inputs[str(source_result)]):
        raise ValueError('vector states do not bind the exact source evaluation and protocol')
    if protocol_registration is not None and vector.get('protocol_registration')!=str(registration):
        raise ValueError('control and vector must bind identical registration path')
    # Retain and check the original vector fit's entire source/input snapshot.
    for path,expected in vector['input_sha256_before_after_equal'].items():
        if c.s.sha(path)!=expected:
            raise ValueError('paired vector provenance changed: '+path)
        inputs[path]=expected
    fixture=HERE/'BCTS_CONTROL_FIXTURES.json'
    checked=json.loads(fixture.read_text())
    if checked['status']!='PASS' or c.s.sha(HERE/'bcts_control.py')!=checked['source_identities_unchanged'][str((HERE/'bcts_control.py').resolve())]:
        raise ValueError('passing current control toy fixture required')
    inputs[str(fixture)]=c.s.sha(fixture)
    tracemalloc.start()
    saved_inputs={}
    input_bytes=0
    for model in e.CONFIRM_MODELS:
        if len(existing['results'][model]['clients'])!=20 or len(vector['results'][model]['clients'])!=20:
            raise ValueError('all 20 clients required')
        for client in range(20):
            source_path=input_root/f'{model}_client{client}.npz'
            vector_path=vector_root/f'{model}_client{client}.npz'
            inputs[str(source_path)]=c.s.sha(source_path)
            inputs[str(vector_path)]=c.s.sha(vector_path)
            saved,bytes_cached=read_saved(source_path)
            input_bytes+=bytes_cached
            with np.load(vector_path,allow_pickle=False) as arrays:
                paired={key:arrays[key].copy() for key in arrays.files}
            for name in ('train_ids','holdout_ids'):
                if not np.array_equal(saved[name],paired[name]):
                    raise ValueError('paired vector original membership differs')
            if not np.array_equal(saved['prioranchor_bias'],paired['prioranchor_bias']):
                raise ValueError('paired original ten-dimensional state differs')
            for kind in ('prioranchored_vector','zero_anchor_vector'):
                state=paired[kind]
                if state.dtype!=np.float32 or state.shape!=(20,) or state.nbytes!=80:
                    raise ValueError('exact paired deployed FP32 vector state required')
            saved_inputs[(model,client)]=(saved,paired)
    out.mkdir(parents=True)
    e.atomic_json(out/'inputs_before.json',{**sources,**inputs})
    started=time.perf_counter()
    results={}
    totals={'new_bcts_fits':0,'new_bcts_steps':0,'new_bcts_linear_solves':0,
        'new_bcts_objective_evaluations':0,'new_bcts_fit_seconds':0.,
        'new_vector_fits':0,'train_rows_used':0,'source_arrays_uncompressed_bytes':input_bytes,
        'peak_single_design_cache_bytes':0,'new_cnn_forwards':0,'new_cnn_backwards':0,'extra_uplink_bytes':0}
    for model in e.CONFIRM_MODELS:
        clients=[]
        for client in range(20):
            saved,paired=saved_inputs[(model,client)]
            anchor=c.s.train_prior_anchor(saved['train_labels'])
            state,optimization=c.fit(saved['train_logits'],saved['train_labels'],anchor)
            raw=saved['raw'].astype(np.float64)
            learned=saved['learned'].astype(np.float64)
            scores={'raw':raw,'analytic_prior':raw+saved['prior'],
                'prioranchor_bias':raw+saved['prioranchor_bias'],
                'zero_anchor_vector':c.s.transform(raw,paired['zero_anchor_vector']),
                'prioranchored_vector':c.s.transform(raw,paired['prioranchored_vector']),
                'learned_head':learned,'learned_head_plus_prior':learned+saved['prior'],
                'prioranchored_bcts':c.transform(raw,state)}
            for name in ('zero_anchor_vector','prioranchored_vector'):
                if not np.array_equal(scores[name],paired[name+'_holdout_logits']):
                    raise ValueError('paired deployed vector scores do not reproduce')
            metrics={name:e.predictive_metrics(z,saved['labels']) for name,z in scores.items()}
            for name in e.VARIANTS:
                baseline=vector['results'][model]['clients'][client]['metrics'][name]
                if any(abs(metrics[name][metric]-baseline[metric])>1e-12 for metric in e.METRICS):
                    raise ValueError('paired vector/control baseline metrics changed')
            np.savez_compressed(out/f'{model}_client{client}.npz',
                train_ids=saved['train_ids'],holdout_ids=saved['holdout_ids'],train_bias_anchor=anchor,
                prioranchored_bcts=state,prioranchored_vector=paired['prioranchored_vector'],
                zero_anchor_vector=paired['zero_anchor_vector'],prioranchor_bias=saved['prioranchor_bias'],
                prioranchored_bcts_holdout_logits=scores['prioranchored_bcts'])
            clients.append({'client':client,'train_samples':len(saved['train_labels']),
                'holdout_samples':len(saved['labels']),'metrics':metrics,'optimization':optimization,
                'incremental_state_bytes':44,'new_cnn_forwards':0,'new_cnn_backwards':0,'new_uplink_bytes':0})
            totals['new_bcts_fits']+=1
            totals['new_bcts_steps']+=optimization['accepted_steps']
            totals['new_bcts_linear_solves']+=optimization['newton_linear_solves']
            totals['new_bcts_objective_evaluations']+=optimization['objective_evaluations']
            totals['new_bcts_fit_seconds']+=optimization['seconds']
            totals['train_rows_used']+=len(saved['train_labels'])
            totals['peak_single_design_cache_bytes']=max(totals['peak_single_design_cache_bytes'],optimization['design_cache_bytes'])
        results[model]={'clients':clients,'summaries':summarize(clients)}
        print(json.dumps({'model':model,'summaries':results[model]['summaries']}),flush=True)
    current,peak=tracemalloc.get_traced_memory()
    tracemalloc.stop()
    totals.update({'tracemalloc_observed_current_bytes':current,'tracemalloc_observed_peak_bytes':peak,
        'memory_scope':'traced Python/NumPy allocations; not process RSS or exact peak bound'})
    frozen={**sources,**inputs}
    if any(c.s.sha(path)!=expected for path,expected in frozen.items()) or attempts:
        raise ValueError('control provenance changed or dataset/TEST bytes attempted')
    comparisons={}
    for model,result in results.items():
        scalar=result['summaries']['prioranchored_bcts']
        vector_summary=result['summaries']['prioranchored_vector']
        comparisons[model]={'bcts_minus_vector_accuracy_pp':100*(scalar['macro_accuracy']-vector_summary['macro_accuracy']),
            **{'bcts_minus_vector_'+name:scalar['macro_'+name]-vector_summary['macro_'+name]
               for name in ('nll','brier','ece15')}}
    report={'status':'DONE','role':'mechanism control only; no new candidate, gate or decision',
        'results':results,'comparisons':comparisons,
        'input_root':str(input_root),'vector_root':str(vector_root),
        'protocol_plan':str(protocol_plan),'protocol_plan_sha256':PROTOCOL_SHA256,
        'control_addendum_sha256':ADDENDUM_SHA256,'source_identities':sources,
        'input_sha256_before_after_equal':frozen,
        'recipe':'sum TRAIN CE +10/2(a-1)^2+10/2||b-center(log(TRAINcount+1))||²; a>=0.001; original BCTS bound/KKT/Newton/Armijo schedule',
        'interpretation':'one scalar vs ten scales differs in dimension and total ridge penalty; no automatic attribution to class-specific scales; report both favorable and unfavorable controls',
        'score_application':'FP32 saved scalar/bias and paired FP32 vector states promoted to FP64; FP64 optimizer theta is diagnostic only',
        'deployment_state_bytes':{'prioranchored_bcts':44,'prioranchored_vector':80,'zero_anchor_vector':80,'prioranchor_bias':40},
        'label_usage':'only saved TRAIN labels enter anchor and 11d fit; holdout only final scoring; cached 20d states reused without refit',
        'new_cnn_forwards':0,'new_cnn_backwards':0,'new_dataset_bytes':0,'extra_uplink_bytes':0,
        'forbidden_dataset_open_attempts':attempts,'totals':totals,'elapsed_seconds':time.perf_counter()-started}
    e.atomic_json(out/'result.json',report)
    print(json.dumps({'status':'DONE','new_bcts_fits':totals['new_bcts_fits'],'new_vector_fits':0,'gates_evaluated':False}),flush=True)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument('--input-root',type=Path,required=True)
    parser.add_argument('--vector-root',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--protocol-plan',type=Path,required=True)
    parser.add_argument('--protocol-registration',type=Path)
    args=parser.parse_args()
    run(args.input_root,args.vector_root,args.out,args.protocol_plan,args.protocol_registration)


if __name__=='__main__':
    main()
