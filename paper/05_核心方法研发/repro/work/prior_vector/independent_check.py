"""Independent saved-state transform/metric/objective checks; no new fitting."""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np

HERE=Path(__file__).resolve().parent
OUT=HERE/'diagnostic_v1'
record=json.loads((OUT/'result.json').read_text())
guard=[]
def audit(event,args):
    if event=='open' and isinstance(args[0],(str,bytes)):
        path=str(args[0])
        if '/dataset/' in path or '/datasets/' in path or 'test_batch' in path or 't10k' in path:
            guard.append(path)
            raise AssertionError('independent check may read saved states only')
sys.addaudithook(audit)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metric(z,y):
    z=np.asarray(z,dtype=np.float64)
    peak=np.max(z,axis=1)
    ce=peak+np.log(np.sum(np.exp(z-peak[:,None]),axis=1))-z[np.arange(len(y)),y]
    return {'accuracy':float(np.mean(np.argmax(z,axis=1)==y)),
        'loss':float(np.mean(ce)),'samples':len(y)}


before={path:sha(path) for path in record['input_sha256_before_after_equal']}
assert before==record['input_sha256_before_after_equal']
output_before={str(path):sha(path) for path in OUT.glob('*.npz')}
source=Path(record['input_root'])
max_metric_error=max_objective_error=max_gradient_error=0.
max_fp64_optimization_vs_fp32_state_scores=0.
max_fp32_arithmetic_vs_fp64_application_scores=0.
max_fp32_arithmetic_accuracy_delta=0.
max_fp32_arithmetic_loss_delta=0.
verified_states=verified_array_elements=0
summaries={}
stop_reasons={}
for model,result in record['results'].items():
    clients=[]
    for client,item in enumerate(result['clients']):
        with np.load(source/f'{model}_client{client}.npz',allow_pickle=False) as raw:
            saved={name:raw[name].copy() for name in raw.files}
        with np.load(OUT/f'{model}_client{client}.npz',allow_pickle=False) as arrays:
            new={name:arrays[name].copy() for name in arrays.files}
        assert np.array_equal(new['train_ids'],saved['train_ids'])
        assert np.array_equal(new['holdout_ids'],saved['holdout_ids'])
        assert np.array_equal(new['prioranchor_bias'],saved['fitted_bias'])
        counts=np.bincount(saved['train_labels'],minlength=10)
        anchor=np.log(counts.astype(np.float64)+1.)
        anchor-=anchor.mean()
        assert np.array_equal(anchor,new['train_bias_anchor'])
        z=saved['holdout_raw'].astype(np.float64)
        scores={'raw':z,'analytic_prior':z+saved['prior_shift'],
            'prior_only':np.broadcast_to(saved['prior_shift'],z.shape),
            'prioranchor_bias':z+saved['fitted_bias'],
            'learned_head':saved['holdout_learned'].astype(np.float64),
            'learned_head_plus_prior':saved['holdout_learned'].astype(np.float64)+saved['prior_shift']}
        for kind,info_key,zero in (
                ('prioranchored_vector','prioranchored_vector_optimization',False),
                ('zero_anchor_vector','zero_anchor_vector_optimization',True)):
            state=new[kind]
            assert state.dtype==np.float32 and state.shape==(20,) and state.nbytes==80
            parameters64=np.asarray(item[info_key]['parameters_float64'],dtype=np.float64)
            assert np.array_equal(parameters64.astype(np.float32),state)
            deploy64=state.astype(np.float64)
            transformed=z*deploy64[:10]+deploy64[10:]
            assert np.array_equal(transformed,new[kind+'_holdout_logits'])
            scores[kind]=transformed
            max_fp64_optimization_vs_fp32_state_scores=max(max_fp64_optimization_vs_fp32_state_scores,
                float(np.max(np.abs(z*parameters64[:10]+parameters64[10:]-transformed))))
            fp32_scores=saved['holdout_raw'].astype(np.float32)*state[:10]+state[10:]
            max_fp32_arithmetic_vs_fp64_application_scores=max(max_fp32_arithmetic_vs_fp64_application_scores,
                float(np.max(np.abs(fp32_scores.astype(np.float64)-transformed))))
            expected_metric=metric(transformed,saved['holdout_labels'])
            fp32_metric=metric(fp32_scores,saved['holdout_labels'])
            max_fp32_arithmetic_accuracy_delta=max(max_fp32_arithmetic_accuracy_delta,
                abs(expected_metric['accuracy']-fp32_metric['accuracy']))
            max_fp32_arithmetic_loss_delta=max(max_fp32_arithmetic_loss_delta,
                abs(expected_metric['loss']-fp32_metric['loss']))
            train=saved['train_logits'].astype(np.float64)
            prediction=train*parameters64[:10]+parameters64[10:]
            peak=prediction.max(axis=1)
            exp=np.exp(prediction-peak[:,None])
            p=exp/exp.sum(axis=1,keepdims=True)
            ce=peak+np.log(exp.sum(axis=1))-prediction[np.arange(len(train)),saved['train_labels']]
            target=np.r_[np.ones(10),np.zeros(10) if zero else anchor]
            # The configurable wrapper re-centers supplied anchors once.
            if not zero:
                target[10:]-=target[10:].mean()
            delta=parameters64-target
            objective=float(np.sum(ce)+5*np.dot(delta,delta))
            max_objective_error=max(max_objective_error,abs(objective-item[info_key]['final_objective']))
            error=p.copy()
            error[np.arange(len(train)),saved['train_labels']]-=1.
            gradient=np.r_[np.sum(error*train,axis=0),np.sum(error,axis=0)]+10*delta
            max_gradient_error=max(max_gradient_error,
                abs(float(np.abs(gradient).max())-item[info_key]['projected_gradient_inf']))
            reason=item[info_key]['stop_reason']
            stop_reasons[reason]=stop_reasons.get(reason,0)+1
            verified_states+=1
            verified_array_elements+=transformed.size
        observed={name:metric(value,saved['holdout_labels']) for name,value in scores.items()}
        for name,values in observed.items():
            for key,value in values.items():
                max_metric_error=max(max_metric_error,abs(value-item['metrics'][name][key]))
        clients.append(observed)
    weights=np.array([row['holdout_samples'] for row in result['clients']],dtype=np.float64)
    weights/=weights.sum()
    summaries[model]={}
    for name in clients[0]:
        accuracy=np.array([row[name]['accuracy'] for row in clients])
        loss=np.array([row[name]['loss'] for row in clients])
        summary={'macro_accuracy':float(np.mean(accuracy)),'macro_loss':float(np.mean(loss)),
            'weighted_accuracy':float(accuracy@weights),'weighted_loss':float(loss@weights),
            'client_accuracy_q10':float(np.quantile(accuracy,.1))}
        for key,value in summary.items():
            max_metric_error=max(max_metric_error,abs(value-result['summaries'][name][key]))
        summaries[model][name]=summary
candidate=summaries['fedlc_uniform3']
n=candidate['prioranchored_vector'];t=candidate['prioranchor_bias'];v=candidate['zero_anchor_vector'];p=candidate['analytic_prior']
gates=[n['macro_accuracy']-t['macro_accuracy']>=.005,n['macro_loss']-t['macro_loss']<=.005,
    n['macro_accuracy']-v['macro_accuracy']>=-.002,v['macro_loss']-n['macro_loss']>=.01,
    n['macro_accuracy']-p['macro_accuracy']>=.005,n['macro_loss']<p['macro_loss']]
assert gates==list(record['gates'].values())
assert ('FOLLOW_UP' if all(gates) else 'STOP')==record['decision']=='STOP'
assert max_metric_error==0. and max_objective_error<1e-9 and max_gradient_error<1e-9
assert not guard
assert {path:sha(path) for path in before}==before
assert {path:sha(path) for path in output_before}==output_before
report={'status':'PASS','decision_verified':'STOP','candidate_fixed':'fedlc_uniform3',
    'verified_output_files':len(output_before),'verified_vector_states':verified_states,
    'bitwise_verified_holdout_score_elements':verified_array_elements,
    'score_application':'FP32 saved A,b promoted to FP64; FP64 saved raw logits application and scoring for every vector baseline. FP64 optimization theta is diagnostic only.',
    'state_rounding':'saved FP32 state equals optimization float64 theta cast once to float32',
    'max_metric_error':max_metric_error,'max_objective_formula_error':max_objective_error,
    'max_gradient_inf_formula_error':max_gradient_error,
    'max_fp64_optimization_vs_fp32_state_score_difference':max_fp64_optimization_vs_fp32_state_scores,
    'fp32_arithmetic_sensitivity_only':{'max_score_difference':max_fp32_arithmetic_vs_fp64_application_scores,
        'max_client_accuracy_difference':max_fp32_arithmetic_accuracy_delta,
        'max_client_nll_difference':max_fp32_arithmetic_loss_delta,
        'scope':'diagnostic arithmetic sensitivity; official metrics use promoted FP32 states with FP64 arithmetic'},
    'optimizer_stop_reasons':stop_reasons,'verified_source_input_identities':len(before),
    'source_input_output_hashes_unchanged':True,'result_sha256':sha(OUT/'result.json'),
    'dataset_open_attempts':guard,'new_cnn_forwards':0,'new_cnn_backwards':0,
    'new_fits':0,'harness_sha256':sha(__file__)}
(HERE/'INDEPENDENT_CHECK.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
