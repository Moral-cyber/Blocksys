"""Independent toy CPU formulas, no real-data training or TEST loading."""
from pathlib import Path
import copy
import hashlib
import importlib.util
import json
import math
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset,DataLoader

torch.set_num_threads(1)
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('shadow_frozen_fixture',HERE/'run.py')
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)

class Toy(nn.Module):
 def __init__(self):
  super().__init__();self.linear=nn.Linear(3,2,bias=True,dtype=torch.float64)
 def forward(self,x):return self.linear(x)

anchor=np.array([.2,-.1,.05,-.13,.14,.03,.02,-.04],dtype=np.float64)
old=np.array([.08,-.06,.012,-.011,.015,-.035,.071,-.02],dtype=np.float64)
x=np.array([[1.,.2,-.3],[-.5,.8,.4],[.7,-.2,.5],[-.3,-.6,.9]],dtype=np.float64)
y=np.array([0,1,1,0],dtype=np.int64)
model=Toy();a.r.assign_vector(model,torch.tensor(anchor))
res=torch.tensor(old);ds=TensorDataset(torch.tensor(x),torch.tensor(y));k=3

def grad(flat,xb,yb):
 logits=xb@flat[:6].reshape(2,3).T+flat[6:]
 ex=np.exp(logits-logits.max(1,keepdims=True));pr=ex/ex.sum(1,keepdims=True)
 pr[np.arange(len(yb)),yb]-=1;pr/=len(yb)
 return np.r_[(pr.T@xb).ravel(),pr.sum(0)]

def reference(theta,xb,yb,method):
 half=theta-(.03 if method=='standard' else .015)*grad(theta,xb,yb)
 if method=='standard':return half,None
 if method=='samebatch':return half-.015*grad(half,xb,yb),None
 u=half-anchor+old
 ids=np.argsort(-np.abs(u),kind='stable')[:k]
 assert min(abs(u[ids]))>max(abs(u[np.setdiff1d(np.arange(8),ids)]))+1e-5
 mask=np.zeros(8);mask[ids]=1
 shadow_point=anchor+mask*u
 expected=half-.015*mask*grad(shadow_point,xb,yb)
 return expected,{'half':half,'shadow_point':shadow_point,'support':ids.tolist(),
   'projected_gradient':mask*grad(shadow_point,xb,yb),'unprojected_gradient':grad(shadow_point,xb,yb)}

checks={};single={}
for method in ['standard','samebatch','shadow']:
 expected,detail=reference(anchor.copy(),x,y,method)
 delta,counts=a.train(model,ds,77,res,k,method,'cpu')
 actual=anchor+delta.numpy()
 np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-15)
 np.testing.assert_array_equal(a.r.vectorize(model).numpy(),anchor)
 np.testing.assert_array_equal(res.numpy(),old)
 assert counts=={'optimizer_steps':1 if method=='standard' else 2,
  'gradient_evaluations':1 if method=='standard' else 2,'batches':1}
 single[method]={'max_abs_parameter_error':float(np.max(np.abs(actual-expected))),**counts}
 if detail is not None:
  assert not np.allclose(actual,detail['shadow_point'])
  assert not np.allclose(actual,detail['half']-.015*detail['unprojected_gradient'])
  selected=set(detail['support']);outside=list(set(range(8))-selected)
  np.testing.assert_allclose(actual[outside],detail['half'][outside],rtol=0,atol=1e-15)
  single[method]['support']=detail['support'];single[method]['actual_not_overwritten_by_shadow']=True
  single[method]['outside_support_second_half_unchanged']=True
checks['single_batch_independent_numpy_ce_formula']=single

# Two real loader minibatches with only synthetic data; old residual is fixed.
# Reconstruct shuffled minibatches independently; no optimizer/reference torch autograd.
xx=np.tile(x,(17,1));yy=np.tile(y,17);ds2=TensorDataset(torch.tensor(xx),torch.tensor(yy))
loader=DataLoader(ds2,batch_size=64,shuffle=True,generator=torch.Generator(device='cpu').manual_seed(99))
theta=anchor.copy()
for xb,yb in loader:
 theta,_=reference(theta,xb.numpy(),yb.numpy(),'shadow')
delta,counts=a.train(model,ds2,99,res,k,'shadow','cpu')
np.testing.assert_allclose(anchor+delta.numpy(),theta,rtol=0,atol=1e-15)
np.testing.assert_array_equal(a.r.vectorize(model).numpy(),anchor);np.testing.assert_array_equal(res.numpy(),old)
assert counts=={'optimizer_steps':4,'gradient_evaluations':4,'batches':2}
checks['two_batch_fixed_residual']={'synthetic_rows':68,'max_abs_parameter_error':float(np.max(np.abs(anchor+delta.numpy()-theta))),
 'global_unchanged':True,'old_residual_unchanged':True,**counts}
# Sole final compression uses actual theta+old residual; no repair or shadow overwrite.
u=delta+res
ids=torch.argsort(u.abs(),descending=True,stable=True)[:k]
q=torch.zeros_like(u);q[ids]=u[ids];new=u-q
assert torch.equal(q+new,u)
expected_u=theta-anchor+old
np.testing.assert_allclose(u.numpy(),expected_u,rtol=0,atol=1e-15)
checks['terminal_upload_ef']={'selected_coordinates':k,'ef_conservation_exact':True,
 'candidate_is_actual_theta_minus_anchor_plus_old_residual':True,'no_extra_repair':True}

cfg=json.loads((HERE/'development_s20261161/standard/config.json').read_text());m=cfg['data_manifest']
t=np.concatenate(m['train_indices']);h=np.concatenate(m['holdout_indices']);v=np.array(m['validation_indices'])
assert len(t)==37500 and len(h)==12500 and len(v)==2000 and not np.intersect1d(t,h).size
assert np.array_equal(np.sort(np.r_[t,h]),np.arange(50000)) and np.isin(v,h).all()
assert cfg['seed']==20261161 and cfg['selected_coordinates']==6843
checks['data_budget_identity']={'fresh_seed':20261161,'train_n':37500,'dev_n':2000,'holdout_pool_n':12500,
 'k_coordinates':6843,'per_client_analytical_upload_bytes':54808,'per_round_upload_bytes':1096160,'round20_upload_bytes':21923200}
checks['source_hashes']={name:hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in ['run.py','PLAN.md']}
assert cfg['source_sha256']==checks['source_hashes']['run.py'] and cfg['plan_sha256']==checks['source_hashes']['PLAN.md']
checks['status']='PASS';checks['external_cross_model_audit']='REVIEW_UNAVAILABLE'
(HERE/'LOCAL_FIXTURE_EVIDENCE.json').write_text(json.dumps(checks,indent=2)+'\n')
print(json.dumps(checks,indent=2))
