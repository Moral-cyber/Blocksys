"""Finite CPU algebra/codec fixtures; no training, image loading or TEST calls."""
from pathlib import Path
import hashlib
import importlib.util
import json
import math
import numpy as np
import torch
from torch.utils.data import DataLoader,TensorDataset

torch.set_num_threads(1)
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('normalized_frozen_fixture',HERE/'run.py')
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
folder=HERE/'development_s20261151'
cfg=json.loads((folder/'topk/config.json').read_text());m=cfg['data_manifest']
checks={}

# Actual one-epoch DataLoader minibatch enumeration; no model/gradient invocation.
sizes=m['client_sizes'];tau=[];batch_sizes=[]
for n in sizes:
 loader=DataLoader(TensorDataset(torch.arange(n)),batch_size=64,shuffle=True,
  generator=torch.Generator().manual_seed(7))
 seen=[len(x[0]) for x in loader]
 assert len(seen)==math.ceil(n/64) and sum(seen)==n
 tau.append(len(seen));batch_sizes.append(seen[-1])
assert cfg['local_steps']==tau
assert a.accel.controls._ACTIVE_MU==0.
weights=np.asarray(sizes,dtype=float)/sum(sizes)
tau_eff=float(weights@np.asarray(tau))
checks['tau_loader_fixture']={'tau':tau,'last_batch_n':batch_sizes,'tau_eff':tau_eff,
 'train_n':sum(sizes),'active_fedprox_mu':float(a.accel.controls._ACTIVE_MU),
 'no_training_called':True}

# FedNova no-momentum constant-gradient special case and equal-tau reduction.
ww=np.array([.25,.75]);tt=np.array([2.,5.]);eta=.03
grad=np.array([[1.,2.,-3.],[-2.,4.,1.]])
delta=-eta*tt[:,None]*grad
nova=(ww@tt)*np.sum(ww[:,None]*delta/tt[:,None],axis=0)
expected=-eta*(ww@tt)*np.sum(ww[:,None]*grad,axis=0)
np.testing.assert_allclose(nova,expected,rtol=0,atol=1e-16)
assert not np.allclose(nova,np.sum(ww[:,None]*delta,axis=0))
for equal_tau in [1.,3.]:
 np.testing.assert_allclose(equal_tau*np.sum(ww[:,None]*delta/equal_tau,axis=0),np.sum(ww[:,None]*delta,axis=0),atol=1e-16)
checks['nova_algebra_fixture']={'dense_update':nova.tolist(),'dense_formula_matches':True,
 'equal_tau_reduces_to_FedAvg':True,'different_tau_differs_from_FedAvg':True}

# RMS normalization should change support, preserve original selected values, and
# be invariant to positive per-tensor scaling apart from floating point ties.
sections=[slice(0,2),slice(2,4),slice(4,6)]
u=torch.tensor([100.,50.,2.,2.,0.,0.])
sent_raw,res_raw,_=a.sparse(u,3,sections,False)
sent_rms,res_rms,counts=a.sparse(u,3,sections,True)
assert sent_raw.tolist()==[100.,50.,2.,0.,0.,0.]
assert sent_rms.tolist()==[100.,0.,2.,2.,0.,0.] and counts==[1,2,0]
assert torch.equal(sent_rms+res_rms,u)
scaled=u*torch.tensor([.01,.01,100.,100.,5.,5.])
sent_scaled,_,scaled_counts=a.sparse(scaled,3,sections,True)
assert counts==scaled_counts and (sent_rms!=0).tolist()==(sent_scaled!=0).tolist()
z,rz,cz=a.sparse(torch.zeros(6),3,sections,True)
assert torch.isfinite(z).all() and not z.any() and torch.equal(z+rz,torch.zeros(6))
checks['rms_sparse_fixture']={'raw_sent':sent_raw.tolist(),'rms_sent':sent_rms.tolist(),
 'rms_counts':counts,'tensor_scale_invariant_support':True,'allzero_finite':True,
 'raw_selected_values_sent':True,'coordinate_not_pack_selection':True}

# Multiround normalized EF: residual stays in delta/tau units and telescopes.
# Verify actual production sparse() path plus independent FP64 conservation.
taus=np.array([2,5]);weights=np.array([.25,.75]);eff=float(weights@taus)
deltas=[np.array([[1.,2.,3.,4.],[-2.,3.,1.,2.]]),
 np.array([[2.,-1.,1.,-3.],[4.,-2.,0.,1.]]),np.array([[0.,3.,2.,1.],[-3.,0.,1.,2.]])]
residual=[torch.zeros(4,dtype=torch.float64) for _ in range(2)]
server_total=np.zeros(4);normalized_delta_total=np.zeros(4)
for ds in deltas:
 sent_sum=np.zeros(4)
 for i in range(2):
  d=torch.tensor(ds[i]/taus[i],dtype=torch.float64)
  candidate=d+residual[i]
  sent,residual[i],_=a.sparse(candidate,2,[slice(0,2),slice(2,4)],True)
  assert torch.equal(sent+residual[i],candidate)
  sent_sum+=weights[i]*sent.numpy()
  normalized_delta_total+=weights[i]*d.numpy()
 server_total+=eff*sent_sum
remaining=sum(weights[i]*residual[i].numpy() for i in range(2))
np.testing.assert_allclose(server_total,eff*(normalized_delta_total-remaining),rtol=0,atol=2e-15)
checks['normalized_ef_telescoping_fixture']={'server_update_total':server_total.tolist(),
 'telescoping_max_abs_error':float(np.max(np.abs(server_total-eff*(normalized_delta_total-remaining)))),
 'residual_units':'delta/tau, before server multiplier','status':'PASS'}

# Manifest indexes, global dev2000 semantics, fresh seed and source identity.
t=np.concatenate(m['train_indices']);h=np.concatenate(m['holdout_indices']);v=np.array(m['validation_indices'])
assert len(t)==37500 and len(h)==12500 and len(v)==2000
assert not np.intersect1d(t,h).size and np.array_equal(np.sort(np.r_[t,h]),np.arange(50000))
assert len(np.unique(v))==2000 and np.isin(v,h).all() and not np.isin(v,t).any()
assert cfg['seed']==20261151 and cfg['selected_coordinates']==6843 and cfg['parameter_count']==136874
checks['data_and_budget']={'train_n':len(t),'holdout_pool_n':len(h),'actual_dev_n':len(v),
 'client_train_min':min(sizes),'k_coordinates':6843,'per_client_analytical_upload_bytes':64+8*6843,
 'round_upload_bytes':20*(64+8*6843),'rounds20_upload_bytes':20*20*(64+8*6843),
 'fresh_seed':cfg['seed'],'no_warmup':True}
checks['source_hashes']={x:hashlib.sha256((HERE/x).read_bytes()).hexdigest() for x in ['run.py','PLAN.md']}
assert cfg['source_sha256']==checks['source_hashes']['run.py'] and cfg['plan_sha256']==checks['source_hashes']['PLAN.md']
checks['status']='PASS'
(HERE/'LOCAL_FIXTURE_EVIDENCE.json').write_text(json.dumps(checks,indent=2)+'\n')
print(json.dumps(checks,indent=2))
