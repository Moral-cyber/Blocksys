"""CPU-only BCTS control math and complete synthetic saved-array dispatch."""
from pathlib import Path
import contextlib
import io
import json
import tempfile

import numpy as np
import bcts_control as c
import bcts_control_probe as probe
import evaluate_saved as e

HERE=Path(__file__).resolve().parent
core_paths=(HERE/'BCTS_CONTROL_PLAN.md',HERE/'bcts_control.py',HERE/'bcts_control_probe.py',
    Path(c.s.__file__).resolve(),c.s.BASE_PATH,Path(e.__file__).resolve())
sources={str(path.resolve()):c.s.sha(path) for path in core_paths}
original_problem=c.s.base.fit.__globals__['Problem']
original_variants=e.VARIANTS
rng=np.random.default_rng(20261317)
z=rng.normal(size=(37,10))
y=rng.choice([0,2,7],size=37)
anchor=c.s.train_prior_anchor(y)
p=c.Problem(z,y,anchor)
theta=p.project(p.anchor+rng.normal(size=11)*.1)
objective,gradient,hessian=p.objective(theta)
eps,ng,nh=1e-5,[],[]
for column in range(11):
    offset=np.eye(11)[column]*eps
    plus,minus=p.objective(theta+offset),p.objective(theta-offset)
    ng.append((plus[0]-minus[0])/(2*eps))
    nh.append((plus[1]-minus[1])/(2*eps))
gerror=float(np.max(np.abs(gradient-ng)))
herror=float(np.max(np.abs(hessian-np.array(nh).T)))
assert gerror<1e-7 and herror<1e-7
assert np.linalg.eigvalsh(hessian).min()>=10-1e-9
zero_problem=c.Problem(z,y,np.zeros(10))
_,g0,h0=zero_problem.objective(theta)
assert np.array_equal(hessian,h0)
assert np.max(np.abs(gradient-g0+10*(p.anchor-zero_problem.anchor)))<1e-12
gauge=np.r_[0.,np.ones(10)]
assert np.max(np.abs(hessian@gauge-10*gauge))<1e-12
assert abs(p.objective(theta+.7*gauge,False)-objective-5*10*.7**2)<1e-10
equivalence,cases=[],[]
inverted_labels=np.tile(np.arange(10),20)
inverted=-10*np.eye(10)[inverted_labels]
for name,logits,labels in (
        ('imbalanced',rng.normal(size=(103,10)),rng.choice([0,2,7],size=103)),
        ('single_class',rng.normal(size=(21,10))*20,np.full(21,3)),
        ('one_row',np.full((1,10),1000.),np.array([9])),
        ('balanced_zero',np.zeros((100,10)),np.tile(np.arange(10),10)),
        ('positive_bound',inverted,inverted_labels)):
    removed,removed_info=c.fit(logits,labels,np.zeros(10))
    frozen,frozen_info=c.s.base.fit(logits,labels,'bcts')
    assert np.array_equal(removed,frozen)
    assert removed_info['parameters_float64']==frozen_info['parameters_float64']
    assert removed_info['trace']==frozen_info['trace']
    equivalence.append({'case':name,'state_bitwise_equal':True,'float64_parameters_trace_equal':True})
    state,info=c.fit(logits,labels,c.s.train_prior_anchor(labels))
    assert state.dtype==np.float32 and state.shape==(11,) and state.nbytes==44
    assert state[0]>=.001 and info['accepted_steps']<=50
    assert all(right['objective']<=left['objective'] for left,right in zip(info['trace'],info['trace'][1:]))
    manual=logits.astype(np.float64)*state[0].astype(np.float64)+state[1:].astype(np.float64)
    assert np.array_equal(c.transform(logits,state),manual)
    if name=='positive_bound':
        assert abs(info['parameters_float64'][0]-.001)<1e-15
        assert info['projected_gradient_inf']<1e-8
    cases.append({'case':name,'steps':info['accepted_steps'],'a':float(state[0]),
        'projected_gradient_inf':info['projected_gradient_inf'],'stop_reason':info['stop_reason']})
assert c.s.base.fit.__globals__['Problem'] is original_problem
assert e.VARIANTS==original_variants
assert all(c.s.sha(path)==digest for path,digest in sources.items())
toy={'status':'PASS','scope':'CPU NumPy synthetic logits only; no real source arrays or images',
    'gradient_fd_max_error':gerror,'hessian_fd_max_error':herror,
    'convex_hessian_ridge10':True,'anchor_changes_gradient_only':True,'gauge_checks':True,
    'anchor_deletion_equivalence':equivalence,'optimization_and_bound_cases':cases,
    'deployment_state_bytes':44,'fp32_state_fp64_application_bitwise_checked':True,
    'original_solver_globals_and_evaluator_variants_unchanged':True,
    'source_identities_unchanged':sources,'harness_sha256':c.s.sha(__file__),
    'real_saved_array_fits':0,'new_cnn_forwards':0,'new_cnn_backwards':0,'formal_ML_training_started':False}
(HERE/'BCTS_CONTROL_FIXTURES.json').write_text(json.dumps(toy,indent=2)+'\n')

# Full source-schema/provenance/metric/NPZ path on 3x20 synthetic clients.
# The earlier core fixture JSON is a frozen input of this run; do not rewrite it.
tmp=Path(tempfile.mkdtemp(prefix='bcts_control_integration_'))
source=tmp/'source'
source.mkdir()
plan=HERE.parent/'predictive_pfl_confirm/PLAN.md'
registration=HERE.parent/'predictive_pfl_confirm/PREREGISTRATION.json'
source_result={'status':'DONE','task':'synthetic control fixture only','results':{}}
all_saved={}
for model_index,model in enumerate(e.CONFIRM_MODELS):
    clients=[]
    for client in range(20):
        seed_rng=np.random.default_rng(20261331+client)
        train_labels=seed_rng.integers(0,4,size=4,dtype=np.int64)
        holdout_labels=seed_rng.integers(0,4,size=2,dtype=np.int64)
        logits_rng=np.random.default_rng(20261431+100*model_index+client)
        train=logits_rng.normal(size=(4,10)).astype(np.float32)
        raw=logits_rng.normal(size=(2,10)).astype(np.float32)
        learned=(raw+logits_rng.normal(size=(2,10))*.1).astype(np.float32)
        counts=np.bincount(train_labels,minlength=10)
        prior=np.log((counts+1.)/(len(train_labels)+10)).astype(np.float32)
        # This is a declared synthetic supplied 10d state, not a real fit result.
        ten=c.s.train_prior_anchor(train_labels).astype(np.float32)
        ids=np.arange(client*6,client*6+6,dtype=np.uint32)
        saved={'train_logits':train,'train_labels':train_labels,'train_ids':ids[:4],
            'raw':raw,'learned':learned,'labels':holdout_labels,'holdout_ids':ids[4:],
            'prior_shift':prior,'fitted_bias':ten}
        all_saved[(model,client)]=saved
        np.savez_compressed(source/f'{model}_client{client}.npz',**saved)
        scores={'raw':raw.astype(np.float64),'analytic_prior':raw.astype(np.float64)+prior,
            'prioranchor_bias':raw.astype(np.float64)+ten,'learned_head':learned.astype(np.float64),
            'learned_head_plus_prior':learned.astype(np.float64)+prior}
        clients.append({'client':client,'metrics':{kind:e.predictive_metrics(value,holdout_labels)
            for kind,value in scores.items()}})
    source_result['results'][model]={'clients':clients}
(source/'result.json').write_text(json.dumps(source_result,indent=2)+'\n')
with contextlib.redirect_stdout(io.StringIO()):
    vector=e.run(source,tmp/'vector',plan,registration)
    control=probe.run(source,tmp/'vector',tmp/'control',plan,registration)
assert control['status']=='DONE' and 'decision' not in control and 'gates' not in control
assert control['totals']['new_bcts_fits']==60 and control['totals']['new_vector_fits']==0
assert not control['forbidden_dataset_open_attempts']
metric_error=0.
for model,result in control['results'].items():
    for client,row in enumerate(result['clients']):
        source_saved=all_saved[(model,client)]
        with np.load(tmp/'control'/f'{model}_client{client}.npz',allow_pickle=False) as saved:
            state=saved['prioranchored_bcts']
            assert state.dtype==np.float32 and state.shape==(11,) and state.nbytes==44
            score=source_saved['raw'].astype(np.float64)*state[0].astype(np.float64)+state[1:].astype(np.float64)
            assert np.array_equal(score,saved['prioranchored_bcts_holdout_logits'])
            metrics=e.predictive_metrics(score,source_saved['labels'])
            for name in e.METRICS:
                metric_error=max(metric_error,abs(metrics[name]-row['metrics']['prioranchored_bcts'][name]))
        for variant in e.VARIANTS:
            for name in e.METRICS:
                assert row['metrics'][variant][name]==vector['results'][model]['clients'][client]['metrics'][variant][name]
assert metric_error==0.
assert c.s.base.fit.__globals__['Problem'] is original_problem and e.VARIANTS==original_variants
assert all(c.s.sha(path)==digest for path,digest in sources.items())
integration={'status':'PASS','scope':'synthetic confirm-three/20 clients only; actual fresh-array control not run',
    'fixture_root':str(tmp),'new_synthetic_bcts_fits':60,'cached_vector_states_reused_without_refit':True,
    'all_seven_original_baseline_metrics_equal':True,'bcts_state_score_metric_reconstruction_error':metric_error,
    'no_candidate_gates_or_decision':True,'source_identities_unchanged':sources,
    'core_fixture_sha256':c.s.sha(HERE/'BCTS_CONTROL_FIXTURES.json'),
    'real_saved_array_fits':0,'real_dataset_bytes':0,'new_cnn_forwards':0,'new_cnn_backwards':0,
    'formal_ML_training_started':False,'main_evaluator_sha256':c.s.sha(e.__file__),
    'main_solver_sha256':c.s.sha(c.s.__file__)}
(HERE/'BCTS_CONTROL_INTEGRATION.json').write_text(json.dumps(integration,indent=2)+'\n')
print(json.dumps({'status':'PASS','gradient_fd_error':gerror,'hessian_fd_error':herror,
    'anchor_deletion_cases':len(equivalence),'bound_case':cases[-1],
    'synthetic_control_fits':60,'real_saved_array_fits':0,'main_sources_unchanged':True},indent=2))
