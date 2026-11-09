"""Synthetic derivatives, descent and removal-of-anchor equivalence fixtures."""
from pathlib import Path
import json
import numpy as np
import solver as s

HERE=Path(__file__).resolve().parent
before={str(path):s.sha(path) for path in (HERE/'PLAN.md',HERE/'solver.py',s.BASE_PATH)}
rng=np.random.default_rng(20261271)
logits=rng.normal(size=(37,10))
labels=rng.choice([0,2,7],size=37)
anchor=s.train_prior_anchor(labels)
counts=np.bincount(labels,minlength=10)
manual=np.log((counts+1.)/(len(labels)+10.))
manual-=manual.mean()
assert np.max(np.abs(anchor-manual))<1e-15
problem=s.Problem(logits,labels,anchor)
theta=problem.project(problem.anchor+rng.normal(size=20)*.2)
objective,gradient,hessian=problem.objective(theta)
eps,numerical_gradient,numerical_hessian=1e-5,[],[]
for column in range(20):
    offset=np.eye(20)[column]*eps
    plus,minus=problem.objective(theta+offset),problem.objective(theta-offset)
    numerical_gradient.append((plus[0]-minus[0])/(2*eps))
    numerical_hessian.append((plus[1]-minus[1])/(2*eps))
gerror=float(np.max(np.abs(gradient-numerical_gradient)))
herror=float(np.max(np.abs(hessian-np.array(numerical_hessian).T)))
assert gerror<1e-7 and herror<1e-7
assert np.linalg.eigvalsh(hessian).min()>=10-1e-9
assert np.max(np.abs(hessian-hessian.T))<1e-12
zero=s.Problem(logits,labels,np.zeros(10))
zero_objective,zero_gradient,zero_hessian=zero.objective(theta)
anchor_delta=problem.anchor-zero.anchor
assert np.max(np.abs(gradient-zero_gradient+10*anchor_delta))<1e-12
assert np.array_equal(hessian,zero_hessian)
expected_difference=5*(np.dot(theta-problem.anchor,theta-problem.anchor)-np.dot(theta-zero.anchor,theta-zero.anchor))
assert abs(objective-zero_objective-expected_difference)<1e-11
other=problem.project(problem.anchor+rng.normal(size=20)*.3)
assert problem.objective((theta+other)/2,False)<=(problem.objective(theta,False)+problem.objective(other,False))/2+1e-12
gauge=np.r_[np.zeros(10),np.ones(10)]
assert np.max(np.abs(hessian@gauge-10*gauge))<1e-12
assert abs(problem.objective(theta+.7*gauge,False)-objective-5*10*.7**2)<1e-10
equivalence,cases=[],[]
for name,z,y in (
    ('imbalanced',rng.normal(size=(103,10)),rng.choice([0,3,8],size=103)),
    ('single_class',rng.normal(size=(21,10))*20,np.full(21,3)),
    ('one_row',np.full((1,10),1000.),np.array([9])),
    ('balanced_zero',np.zeros((100,10)),np.tile(np.arange(10),10))):
    removed,removed_info=s.fit(z,y,np.zeros(10))
    frozen,frozen_info=s.base.fit(z,y,'vector_scaling')
    assert np.array_equal(removed,frozen)
    assert removed_info['parameters_float64']==frozen_info['parameters_float64']
    assert removed_info['trace']==frozen_info['trace']
    equivalence.append({'case':name,'state_bitwise_equal':True,'float64_parameters_trace_equal':True})
    state,info=s.fit(z,y,s.train_prior_anchor(y))
    assert state.dtype==np.float32 and state.shape==(20,) and state.nbytes==80
    assert info['accepted_steps']<=50 and info['hessian_min_eigenvalue']>=10-1e-9
    assert info['final_objective']<=info['initial_objective']
    assert all(right['objective']<=left['objective'] for left,right in zip(info['trace'],info['trace'][1:]))
    cases.append({'case':name,'steps':info['accepted_steps'],'state_bytes':state.nbytes,
        'gradient_inf':info['projected_gradient_inf'],'stop_reason':info['stop_reason']})
    if name=='balanced_zero':
        assert np.max(np.abs(s.train_prior_anchor(y)))<1e-15
        assert np.max(np.abs(state-np.r_[np.ones(10),np.zeros(10)]))<1e-12
inverted_labels=np.tile(np.arange(10),20)
inverted=-4*np.eye(10)[inverted_labels]
state,negative=s.fit(inverted,inverted_labels,s.train_prior_anchor(inverted_labels))
assert np.all(state[:10]<0)
attempts=s.install_data_guard()
for path in ('/tmp/prior_vector_guard/dataset/blocked','/tmp/prior_vector_guard_test_batch','/tmp/prior_vector_guard_t10k'):
    try:
        Path(path).read_bytes()
    except ValueError:
        pass
    else:
        raise AssertionError('saved-array guard failed')
assert len(attempts)==3
assert all(s.sha(path)==digest for path,digest in before.items())
report={'status':'PASS','scope':'synthetic NumPy arrays only; no CNN/dataset bytes',
    'gradient_fd_max_error':gerror,'hessian_fd_max_error':herror,
    'positive_definite_hessian':True,'anchor_gradient_delta_only_and_hessian_identical':True,
    'centered_train_log_count_anchor':True,'convexity_gauge_checks':True,
    'anchor_deletion_equivalence':equivalence,'optimization_cases':cases,
    'negative_scales_permitted':state[:10].tolist(),'deployment_state_bytes':80,
    'intentional_dummy_guard_rejections':3,'real_dataset_open_attempts':0,
    'source_identities_unchanged':before,'harness_sha256':s.sha(__file__),
    'formal_ML_training_started':False}
(HERE/'LOCAL_FIXTURES.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
