"""One fixed old-array comparison: TRAIN-prior anchored vs zero vector scaling."""
from pathlib import Path
import argparse
import importlib.util
import json
import sys
import time
import tracemalloc

import numpy as np
import solver as s

HERE=Path(__file__).resolve().parent
INPUT=HERE.parent/'prior_bias/diagnostic_v1'
MODELS=('ce_uniform3','fedlc_uniform3','balanced_uniform3','balanced_dense')
CANDIDATE='fedlc_uniform3'
PINNED={
    'work/calibration_baselines/PLAN.md':'33864fe167b461f5d9d7aa8777b932fdfecc752f9553fda14d1fb23c8ad091a9',
    'work/calibration_baselines/solver.py':'10b11405e6662cb06d08a20549db92e2948b730c426aa1f0b912b958f1211f1c',
    'work/calibration_baselines/probe.py':'d1fab28871824163a6d2fc2ee44e043698537b00dabc0e02af2618531adcd0c7',
    'work/prior_bias/solver.py':'6b8dda5e86438f78f3918f8601ec3ce9e7099417d3b42ac55bfbeb4014132b96'}
INPUT_RESULT_SHA256='fad176a08e6979c7ff7731ef23e5b3017423f69a5d95e4f7d59d75dc5c4fc0e3'


def atomic_json(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    temporary.replace(path)


def read_frozen_reader():
    for relative,expected in PINNED.items():
        if s.sha(s.ROOT/relative)!=expected:
            raise ValueError('frozen dependency differs: '+relative)
    path=HERE.parent/'calibration_baselines/probe.py'
    spec=importlib.util.spec_from_file_location('_prior_vector_frozen_array_reader',path)
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


def diagnostic_gates(summaries):
    new=summaries['prioranchored_vector']
    ten=summaries['prioranchor_bias']
    zero=summaries['zero_anchor_vector']
    prior=summaries['analytic_prior']
    return {
        'vs_ten_dim_accuracy_plus_half_pp':new['macro_accuracy']-ten['macro_accuracy']>=.005,
        'vs_ten_dim_nll_within_0005':new['macro_loss']-ten['macro_loss']<=.005,
        'vs_zero_vector_accuracy_noninferior_02pp':new['macro_accuracy']-zero['macro_accuracy']>=-.002,
        'vs_zero_vector_nll_improvement_001':zero['macro_loss']-new['macro_loss']>=.01,
        'vs_prior_accuracy_plus_half_pp':new['macro_accuracy']-prior['macro_accuracy']>=.005,
        'vs_prior_lower_nll':new['macro_loss']<prior['macro_loss']}


def main():
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument('--out',type=Path,default=HERE/'diagnostic_v1')
    args=parser.parse_args()
    out=args.out.resolve()
    if out.exists():
        raise ValueError('fresh output required; fixed old-array diagnostic is not overwritten')
    attempts=s.install_data_guard()
    reader=read_frozen_reader()
    source_paths=[HERE/name for name in ('PLAN.md','solver.py','probe.py')]
    source_paths+=[s.ROOT/relative for relative in PINNED]
    sources={str(path.resolve()):s.sha(path) for path in source_paths}
    fixture_path=HERE/'LOCAL_FIXTURES.json'
    fixtures=json.loads(fixture_path.read_text())
    if fixtures['status']!='PASS' or any(s.sha(path)!=expected for path,expected in fixtures['source_identities_unchanged'].items()):
        raise ValueError('passing current-source toy fixtures required before saved-array probe')
    inputs={str(INPUT/'result.json'):s.sha(INPUT/'result.json'),str(fixture_path):s.sha(fixture_path)}
    if inputs[str(INPUT/'result.json')]!=INPUT_RESULT_SHA256:
        raise ValueError('fixed old prior-bias input result differs')
    existing=json.loads((INPUT/'result.json').read_text())
    if existing['status']!='DONE' or set(existing['results'])!=set(MODELS):
        raise ValueError('exact completed old four-model input required')
    tracemalloc.start()
    all_saved={}
    source_bytes=selected_bytes=0
    for method in MODELS:
        for client in range(20):
            path=INPUT/f'{method}_client{client}.npz'
            inputs[str(path)]=s.sha(path)
            saved,bytes_cached=reader(path)
            all_saved[(method,client)]=saved
            source_bytes+=bytes_cached
            selected_bytes+=sum(value.nbytes for value in saved.values() if isinstance(value,np.ndarray))
    out.mkdir(parents=True)
    atomic_json(out/'inputs_before.json',{**sources,**inputs})
    results={}
    totals={'new_prior_vector_fits':0,'paired_zero_vector_fits':0,
        'new_accepted_newton_steps':0,'paired_zero_accepted_newton_steps':0,
        'new_newton_linear_solves':0,'new_objective_evaluations':0,
        'new_fit_seconds':0.,'paired_zero_fit_seconds':0.,
        'train_rows_used_per_recipe':0,'source_arrays_uncompressed_bytes':source_bytes,
        'selected_input_array_bytes_cached':selected_bytes,'peak_single_design_cache_bytes':0}
    start=time.perf_counter()
    for method in MODELS:
        clients=[]
        for client in range(20):
            saved=all_saved[(method,client)]
            anchor=s.train_prior_anchor(saved['train_labels'])
            # Both optimizers receive only this client's saved TRAIN arrays.
            new,new_info=s.fit(saved['train_logits'],saved['train_labels'],anchor)
            zero,zero_info=s.base.fit(saved['train_logits'],saved['train_labels'],'vector_scaling')
            raw,labels=saved['raw'].astype(np.float64),saved['labels']
            scores={'raw':raw,'analytic_prior':raw+saved['prior'],
                'prior_only':np.broadcast_to(saved['prior'],raw.shape),
                'prioranchor_bias':raw+saved['prioranchor_bias'],
                'zero_anchor_vector':s.transform(raw,zero),
                'prioranchored_vector':s.transform(raw,new)}
            if saved['learned'] is not None:
                scores.update(learned_head=saved['learned'].astype(np.float64),
                    learned_head_plus_prior=saved['learned'].astype(np.float64)+saved['prior'])
            metrics={name:s.stats(logits,labels) for name,logits in scores.items()}
            for variant,old_metric in existing['results'][method]['clients'][client]['metrics'].items():
                key='prioranchor_bias' if variant=='fitted_bias' else variant
                if key in metrics and (abs(metrics[key]['accuracy']-old_metric['accuracy'])>1e-12
                        or abs(metrics[key]['loss']-old_metric['loss'])>1e-9):
                    raise ValueError('fixed saved baseline failed to reproduce')
            np.savez_compressed(out/f'{method}_client{client}.npz',
                train_ids=saved['train_ids'],holdout_ids=saved['holdout_ids'],
                train_bias_anchor=anchor,prioranchored_vector=new,zero_anchor_vector=zero,
                prioranchor_bias=saved['prioranchor_bias'],
                prioranchored_vector_holdout_logits=scores['prioranchored_vector'],
                zero_anchor_vector_holdout_logits=scores['zero_anchor_vector'])
            clients.append({'client':client,'train_samples':len(saved['train_labels']),
                'holdout_samples':len(labels),'metrics':metrics,
                'prioranchored_vector_optimization':new_info,'zero_anchor_vector_optimization':zero_info,
                'train_prior_counts':np.bincount(saved['train_labels'],minlength=10).tolist(),
                'deployment_state_bytes':80,'new_cnn_forwards':0,'new_cnn_backwards':0,
                'new_uplink_bytes':0})
            totals['new_prior_vector_fits']+=1
            totals['paired_zero_vector_fits']+=1
            totals['new_accepted_newton_steps']+=new_info['accepted_steps']
            totals['paired_zero_accepted_newton_steps']+=zero_info['accepted_steps']
            totals['new_newton_linear_solves']+=new_info['newton_linear_solves']
            totals['new_objective_evaluations']+=new_info['objective_evaluations']
            totals['new_fit_seconds']+=new_info['seconds']
            totals['paired_zero_fit_seconds']+=zero_info['seconds']
            totals['train_rows_used_per_recipe']+=len(saved['train_labels'])
            totals['peak_single_design_cache_bytes']=max(totals['peak_single_design_cache_bytes'],new_info['design_cache_bytes'])
        results[method]={'clients':clients,'summaries':s.summarize(clients)}
        print(json.dumps({'model':method,'summaries':results[method]['summaries']}),flush=True)
    current_memory,peak_memory=tracemalloc.get_traced_memory()
    tracemalloc.stop()
    totals.update({'tracemalloc_observed_current_bytes':current_memory,
        'tracemalloc_observed_peak_bytes':peak_memory,
        'memory_measurement_scope':'traced Python/NumPy allocations after input-load start; includes outputs; not process RSS or exact peak bound'})
    frozen={**sources,**inputs}
    if any(s.sha(path)!=digest for path,digest in frozen.items()):
        raise ValueError('source or saved-array input changed')
    if attempts:
        raise ValueError('forbidden dataset/TEST access attempted')
    comparisons={}
    for method,result in results.items():
        candidate=result['summaries']['prioranchored_vector']
        comparisons[method]={variant:{
            'delta_accuracy_pp':100*(candidate['macro_accuracy']-metrics['macro_accuracy']),
            'delta_nll':candidate['macro_loss']-metrics['macro_loss']}
            for variant,metrics in result['summaries'].items() if variant!='prioranchored_vector'}
    gates=diagnostic_gates(results[CANDIDATE]['summaries'])
    decision='FOLLOW_UP' if all(gates.values()) else 'STOP'
    report={'status':'DONE','candidate':CANDIDATE,'decision':decision,'gates':gates,
        'decision_scope':'fixed known-vector-plus-TRAIN-prior-regularization recipe only; success is an exploratory lead requiring a fresh split',
        'results':results,'comparisons':comparisons,'source_identities':sources,
        'input_sha256_before_after_equal':frozen,'input_root':str(INPUT),
        'source_decision_unchanged':existing.get('decision'),
        'original_ten_dim_six_gates':'unchanged; this diagnostic neither replaces nor relaxes them',
        'method_scope':'known vector scaling plus prior regularization; no atomic novelty, server aggregation, new scoring model or FedCal reproduction claim',
        'regularization':'sum TRAIN CE + 10/2||A-1||² + 10/2||b-center(log(TRAINcount+1))||²',
        'sole_mechanism_change':'same frozen 20-dimensional affine design, replace only zero b-anchor with local TRAIN centered log-count prior',
        'constraints':'unrestricted real A; centered b; fixed max50 Newton/Armijo and no grid',
        'deployment_state_bytes':{'prioranchored_vector':80,'zero_anchor_vector':80,'prioranchor_bias':40},
        'state_scope':'20 float32 values per client; FP64 solve trajectory/cache excluded and reported separately',
        'label_usage':'TRAIN labels only in anchor/fit; holdout only final scores; no model/client/coefficient selection',
        'holdout_scope':'old clientholdout includes old globaldev; not fresh test evidence',
        'new_cnn_forwards':0,'new_cnn_backwards':0,'new_dataset_bytes':0,'extra_uplink_bytes':0,
        'forbidden_dataset_open_attempts':attempts,'totals':totals,
        'elapsed_seconds':time.perf_counter()-start,'numpy':np.__version__,'python':sys.version}
    atomic_json(out/'result.json',report)
    print(json.dumps({'status':'DONE','candidate':CANDIDATE,'decision':decision,'gates':gates,'totals':totals}),flush=True)


if __name__=='__main__':
    main()
