"""Synthetic confirm-three integration and independent predictive metrics."""
from pathlib import Path
import contextlib
import importlib.util
import io
import json
import tempfile

import numpy as np
import evaluate_saved as e

HERE=Path(__file__).resolve().parent
TMP=Path(tempfile.mkdtemp(prefix='prior_vector_saved_evaluator_fixture_'))
source=TMP/'source'
source.mkdir()
plan=TMP/'SYNTHETIC_PLAN.md'
plan.write_text('Synthetic 20-client evaluator fixture only; no scientific gates or prospective data.\n')
sources={str(path):e.s.sha(path) for path in (HERE/'solver.py',HERE/'evaluate_saved.py',e.s.BASE_PATH,e.READER)}


def independent(logits,labels):
    z=np.asarray(logits,dtype=np.float64)
    ex=np.exp(z-z.max(axis=1)[:,None])
    p=ex/ex.sum(axis=1)[:,None]
    y=np.asarray(labels)
    loss=z.max(axis=1)+np.log(ex.sum(axis=1))-z[np.arange(len(y)),y]
    squared=(p-np.eye(10)[y])**2
    correct=z.argmax(axis=1)==y
    confidence=p.max(axis=1)
    ece=0.
    for bin_index in range(15):
        selected=(confidence>=bin_index/15)&(
            confidence<(bin_index+1)/15 if bin_index<14 else confidence<=1.)
        if selected.any():
            ece+=selected.mean()*abs(correct[selected].mean()-confidence[selected].mean())
    return {'accuracy':float(correct.mean()),'nll':float(loss.mean()),
        'brier':float(squared.sum(axis=1).mean()),'ece15':float(ece)}


uniform=e.predictive_metrics(np.zeros((10,10)),np.arange(10))
assert abs(uniform['accuracy']-.1)<1e-15 and abs(uniform['nll']-np.log(10))<1e-15
assert abs(uniform['brier']-.9)<1e-15 and uniform['ece15']<1e-15
perfect=e.predictive_metrics(np.eye(10)*1000,np.arange(10))
assert perfect['accuracy']==1 and perfect['nll']==perfect['brier']==perfect['ece15']==0
wrong=e.predictive_metrics(np.array([[-1000,1000]+[0.]*8]),np.array([0]))
assert wrong['accuracy']==0 and wrong['nll']==2000 and wrong['brier']==2 and wrong['ece15']==1
assert sum(bin['count'] for bin in wrong['ece15_bins'])==1
assert wrong['ece15_bins'][-1]['count']==1

ten_path=HERE.parent/'prior_bias/solver.py'
assert e.s.sha(ten_path)=='6b8dda5e86438f78f3918f8601ec3ce9e7099417d3b42ac55bfbeb4014132b96'
spec=importlib.util.spec_from_file_location('_fixture_frozen_ten_dim',ten_path)
ten=importlib.util.module_from_spec(spec)
spec.loader.exec_module(ten)
all_arrays={}
source_report={'status':'DONE','task':'synthetic fixture only','results':{}}
for model_index,model in enumerate(e.CONFIRM_MODELS):
    clients=[]
    for client in range(20):
        label_rng=np.random.default_rng(20261291+client)
        train_labels=label_rng.integers(0,4,size=8,dtype=np.int64)
        holdout_labels=label_rng.integers(0,4,size=4,dtype=np.int64)
        rng=np.random.default_rng(20261391+model_index*100+client)
        train=rng.normal(size=(8,10)).astype(np.float32)
        raw=rng.normal(size=(4,10)).astype(np.float32)
        learned=(raw+rng.normal(size=(4,10))*.1).astype(np.float32)
        counts=np.bincount(train_labels,minlength=10)
        prior=np.log((counts+1.)/(len(train_labels)+10)).astype(np.float32)
        fitted,_=ten.fit(train,train_labels)
        ids=np.arange(client*12,client*12+12,dtype=np.uint32)
        arrays={'train_logits':train,'train_labels':train_labels,'train_ids':ids[:8],
            'raw':raw,'learned':learned,'labels':holdout_labels,'holdout_ids':ids[8:],
            'prior_shift':prior,'fitted_bias':fitted}
        all_arrays[(model,client)]=arrays
        np.savez_compressed(source/f'{model}_client{client}.npz',**arrays)
        scores={'raw':raw.astype(np.float64),'analytic_prior':raw.astype(np.float64)+prior,
            'prioranchor_bias':raw.astype(np.float64)+fitted,'learned_head':learned.astype(np.float64),
            'learned_head_plus_prior':learned.astype(np.float64)+prior}
        metrics={kind:{**independent(z,holdout_labels),'loss':independent(z,holdout_labels)['nll']}
            for kind,z in scores.items()}
        clients.append({'client':client,'metrics':metrics,'bias_train_image_forwards':8,
            'bias_new_cnn_backward':0,'matched_head_training':{'gradient_evaluations':1}})
    source_report['results'][model]={'clients':clients}
(source/'result.json').write_text(json.dumps(source_report,indent=2)+'\n')
with contextlib.redirect_stdout(io.StringIO()):
    report=e.run(source,TMP/'evaluation',plan)
assert report['status']=='DONE' and report['input_schema']=='confirm_three'
assert 'decision' not in report and 'gates' not in report
assert report['protocol_plan_sha256']==e.s.sha(plan)
assert report['totals']['prior_vector_fits']==report['totals']['zero_vector_fits']==60
assert not report['forbidden_dataset_open_attempts']
max_error=0.
for model,result in report['results'].items():
    recomputed=[]
    for client,row in enumerate(result['clients']):
        arrays=all_arrays[(model,client)]
        raw=arrays['raw'].astype(np.float64)
        learned=arrays['learned'].astype(np.float64)
        with np.load(TMP/'evaluation'/f'{model}_client{client}.npz',allow_pickle=False) as saved:
            assert np.array_equal(saved['train_ids'],arrays['train_ids'])
            assert np.array_equal(saved['holdout_ids'],arrays['holdout_ids'])
            scores={'raw':raw,'analytic_prior':raw+arrays['prior_shift'],
                'prioranchor_bias':raw+arrays['fitted_bias'],'learned_head':learned,
                'learned_head_plus_prior':learned+arrays['prior_shift']}
            for kind in ('prioranchored_vector','zero_anchor_vector'):
                state=saved[kind]
                assert state.dtype==np.float32 and state.nbytes==80
                scores[kind]=raw*state[:10].astype(np.float64)+state[10:].astype(np.float64)
                assert np.array_equal(scores[kind],saved[kind+'_holdout_logits'])
        values={kind:independent(z,arrays['labels']) for kind,z in scores.items()}
        recomputed.append(values)
        for kind,metrics in values.items():
            for name,value in metrics.items():
                max_error=max(max_error,abs(value-row['metrics'][kind][name]))
    for kind in e.VARIANTS:
        for name in e.METRICS:
            manual=float(np.mean([row[kind][name] for row in recomputed]))
            max_error=max(max_error,abs(manual-result['summaries'][kind]['macro_'+name]))
        pooled=np.concatenate([all_arrays[(model,client)]['labels'] for client in range(20)])
        # Independent pooled ECE comes from the retained per-client bin totals.
        counts=np.zeros(15);correct=np.zeros(15);confidence=np.zeros(15)
        for row in result['clients']:
            for index,bin in enumerate(row['metrics'][kind]['ece15_bins']):
                counts[index]+=bin['count'];correct[index]+=bin['correct_sum'];confidence[index]+=bin['confidence_sum']
        expected=float(np.abs(correct-confidence).sum()/len(pooled))
        max_error=max(max_error,abs(expected-result['summaries'][kind]['pooled_ece15']))
assert max_error<1e-14
try:
    e.run(source,TMP/'evaluation',plan)
except ValueError as error:
    assert 'fresh' in str(error)
else:
    raise AssertionError('existing output accepted')
broken=TMP/'incomplete'
broken.mkdir()
(broken/'result.json').write_text(json.dumps({'status':'DONE','results':{'ce_uniform3':{}}}))
try:
    e.run(broken,TMP/'not_created',plan)
except ValueError as error:
    assert 'complete DONE' in str(error)
else:
    raise AssertionError('incomplete model set accepted')
assert not (TMP/'not_created').exists()
assert all(e.s.sha(path)==digest for path,digest in sources.items())
fixture={'status':'PASS','scope':'synthetic 3-model/20-client saved-array integration only; no existing/fresh real split evaluated',
    'fixture_root':str(TMP),'metric_known_cases':['uniform balanced','perfect saturated','confident wrong saturated'],
    'max_independent_metric_summary_error':max_error,'states_float32_score_application_float64':True,
    'complete_schema_and_fresh_output_negative_checks':True,'evaluator_contains_no_gate_or_decision':True,
    'synthetic_vector_fits':120,'new_cnn_forwards':0,'new_cnn_backwards':0,'real_dataset_bytes':0,
    'source_identities_unchanged':sources,'harness_sha256':e.s.sha(__file__),
    'real_array_probe_started':False,'formal_ML_training_started':False}
(HERE/'EVALUATE_SAVED_FIXTURES.json').write_text(json.dumps(fixture,indent=2)+'\n')
print(json.dumps(fixture,indent=2))
