"""Read-only local fixtures for frozen probe; no training or official TEST reads."""
from pathlib import Path
import copy
import hashlib
import importlib.util
import json
import struct
import sys
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset

torch.set_num_threads(1)
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('frozen_probe_checked', HERE/'probe_fusion.py')
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)
run = HERE/'dev_seed20261143'
cfg = json.loads((run/'config.json').read_text())
checks = {}

# Verify actual manifest memberships, not its precomputed integrity booleans.
m = cfg['data_manifest']
tg = [np.asarray(x,dtype=np.int64) for x in m['train_indices']]
hg = [np.asarray(x,dtype=np.int64) for x in m['holdout_indices']]
t = np.concatenate(tg); h = np.concatenate(hg)
v = np.asarray(m['validation_indices'],dtype=np.int64)
assert len(tg)==len(hg)==20
assert len(t)==37500 and len(h)==12500
assert len(np.intersect1d(t,h))==0
assert np.array_equal(np.sort(np.r_[t,h]),np.arange(50000))
assert np.all(np.isin(v,h)) and not np.any(np.isin(v,t)) and len(np.unique(v))==2000
assert [len(x) for x in tg]==m['client_sizes']
assert [len(x) for x in hg]==m['client_holdout_sizes']
def gh(groups):
    z=hashlib.sha256()
    for a in groups:
        z.update(struct.pack('<I',len(a))); z.update(np.asarray(a,dtype='<u4').tobytes())
    return z.hexdigest()
assert gh(tg)==m['train_indices_sha256'] and gh(hg)==m['holdout_indices_sha256']
checks['manifest_recomputed']={'train_count':len(t),'dev_count':len(h),
  'train_dev_intersection':0,'coverage_exact':True,'client_train_min':min(map(len,tg)),
  'client_dev_min':min(map(len,hg)),'unused_stratified_subset_size':len(v)}
checks['source_hash_matches']={q:hashlib.sha256(Path(q).read_bytes()).hexdigest()==z for q,z in cfg['source_sha256'].items()}
assert all(checks['source_hash_matches'].values())

# Independent, hand-computed masked average: absent upload is not numerical zero.
vec=[torch.tensor([2.,4.,8.,16.]),torch.tensor([10.,20.,30.,40.]),torch.tensor([100.,200.,300.,400.])]
mask=[torch.tensor([1,1,0,0],dtype=torch.bool),torch.tensor([1,0,1,0],dtype=torch.bool),torch.tensor([0,1,1,0],dtype=torch.bool)]
w=np.array([.2,.3,.5])
a,valid=p.donor_average(vec,mask,w)
np.testing.assert_allclose(a.numpy(),[6.8,100.8/.7,159/.8,0.],rtol=1e-6)
np.testing.assert_array_equal(valid.numpy(),[1,1,1,0])
a0,v0=p.donor_average(vec,mask,w,exclude=0)
np.testing.assert_allclose(a0.numpy(),[10.,200.,159/.8,0.],rtol=1e-6)
a2,v2=p.donor_average(vec,mask,w,exclude=2)
np.testing.assert_allclose(a2.numpy(),[6.8,4.,30.,0.],rtol=1e-6)
# A genuinely uploaded zero remains available.
az,vz=p.donor_average([torch.zeros(2)],[torch.tensor([1,0],dtype=torch.bool)],[1.])
assert vz.tolist()==[True,False]
local=torch.tensor([7.,7.,7.,7.])
assert torch.where(v0,a0,local)[3].item()==7.
checks['masked_average_fixture']={'inclusive':a.tolist(),'exclude_0':a0.tolist(),
  'exclude_2':a2.tolist(),'zero_upload_distinct_from_missing':True}

# Block gate uses -g^T(donor-local), zeros unavailable positions and nonpositive blocks.
local=torch.tensor([1.,1.,1.,1.,1.])
donor=torch.tensor([2.,2.,2.,2.,999.])
avail=torch.tensor([1,1,1,1,0],dtype=torch.bool)
grad=torch.tensor([-2.,1.,2.,1.,-999.])
direction,diag=p.gate_direction(local,donor,avail,grad,p.pack_slices(5,2))
np.testing.assert_array_equal(direction.numpy(),[1,1,0,0,0])
assert diag=={'selected_packs':1,'predicted_train_gain':1.}
checks['block_gate_fixture']={'direction':direction.tolist(),**diag}

# Vector order, independence, final head, short pack and complete coverage.
model=p.base.SmallCNN()
original=p.base.vectorize(model)
assert original.numel()==136874
sections=p.pack_slices(original.numel(),128)
assert len(sections)==1070 and sections[-1].stop-sections[-1].start==42
covered=np.concatenate([np.arange(s.start,s.stop) for s in sections])
assert np.array_equal(covered,np.arange(original.numel()))
sentinel=torch.arange(original.numel(),dtype=torch.float32)/1000
p.base.assign_vector(model,sentinel)
assert torch.equal(p.base.vectorize(model),sentinel)
head_count=sum(q.numel() for q in model.classifier[-1].parameters())
assert head_count==650
assert torch.equal(torch.cat([q.detach().flatten() for q in model.classifier[-1].parameters()]),sentinel[-head_count:])
original.zero_()
assert torch.equal(p.base.vectorize(model),sentinel)
checks['vector_pack_fixture']={'parameter_count':sentinel.numel(),'pack_count':len(sections),
  'last_pack_values':42,'k_5pct':max(1,int(len(sections)*.05)),'last_linear_parameters':head_count,
  'roundtrip_and_no_alias':True,'complete_nonoverlapping_coverage':True}

# Independent training-batch membership fixtures; no dataset loading needed.
batch_checks=[]
for n in [2,3,104,128,129,130]:
 ds=TensorDataset(torch.arange(n,dtype=torch.float32).reshape(n,1),torch.arange(n,dtype=torch.long))
 fit,check,ids=p.train_batches(ds,20261143)
 assert not set(ids['gradient_positions']) & set(ids['check_positions'])
 assert len(fit[1])==len(check[1])==min(64,n//2)
 batch_checks.append({'train_n':n,'gradient_n':len(fit[1]),'check_n':len(check[1])})
checks['disjoint_train_batches_fixture']=batch_checks

# Count actual forwards and expose tie ordering without any backward/training call.
class CountLinear(nn.Linear):
 def __init__(self): super().__init__(1,2); self.calls=0
 def forward(self,x): self.calls+=1; return super().forward(x)
cm=CountLinear(); cm.eval(); z=p.base.vectorize(cm)
b=(torch.tensor([[1.],[2.]]),torch.tensor([0,1]))
out,diag=p.line_search(cm,z,torch.zeros_like(z),b)
assert torch.equal(out,z) and diag['alpha']==0 and cm.calls==4
checks['line_search_zero_direction_fixture']={'actual_forward_calls':cm.calls,
  'tie_alpha':diag['alpha'],'all_losses_equal':len(set(diag['train_check_ce']))==1,
  'nominal_backward_metadata':diag['extra_backward']}

# Current saved state, CPU read only: recompute masks, donor aggregation and upload lengths.
state_path=run/'shared_state.pt'
if state_path.exists():
 st=torch.load(state_path,map_location='cpu',weights_only=False)
 assert st['signature']==cfg['signature']
 warm=torch.load(run/'warmup.pt',map_location='cpu',weights_only=False)
 assert warm['signature']==cfg['signature'] and warm['round']==cfg['warmup']==10
 assert torch.equal(torch.cat([x.flatten() for x in warm['model'].values()]),st['anchor'])
 anchor=st['anchor']; vectors=st['vectors']; masks=st['masks']; weights=st['weights']
 assert len(vectors)==len(masks)==len(weights)==20
 np.testing.assert_allclose(weights,np.array(m['client_sizes'])/sum(m['client_sizes']),atol=0,rtol=0)
 packs=p.pack_slices(anchor.numel(),128); k=max(1,int(len(packs)*.05))
 upload_stats=[]
 for i,(loc,mask_) in enumerate(zip(vectors,masks)):
  assert loc.shape==anchor.shape and mask_.dtype==torch.bool and torch.isfinite(loc).all()
  de=loc-anchor
  # Reconstructed subtraction can alter last-bit energy after anchor+delta rounding;
  # verify selected/nonselected separation rather than asserting exact top-k ties.
  energies=np.array([float(de[s].double().square().sum()) for s in packs])
  selected=[j for j,s in enumerate(packs) if bool(mask_[s].any())]
  assert len(selected)==k and all(bool(mask_[packs[j]].all()) for j in selected)
  assert all(not bool(mask_[packs[j]].any()) for j in set(range(len(packs)))-set(selected))
  unselected=list(set(range(len(packs)))-set(selected))
  min_selected=float(energies[selected].min()); max_other=float(energies[unselected].max())
  assert min_selected>=max_other*(1-1e-5)
  values=int(mask_.sum()); payload=64+4*k+4*values
  upload_stats.append({'client':i,'packs':k,'values':values,'payload_bytes_formula':payload,
    'short_pack_selected':len(packs)-1 in selected,
    'min_selected_energy':min_selected,'max_unselected_energy':max_other})
 # NumPy independent masked weighted mean for every receiver and all parameters.
 vv=np.stack([x.numpy() for x in vectors]).astype(np.float64)
 mm=np.stack([x.numpy() for x in masks]); ww=np.asarray(weights,dtype=np.float64)
 worst=0.
 for i in range(20):
  donor,va=p.donor_average(vectors,masks,weights,exclude=i)
  active=mm.copy();active[i]=False
  den=(active*ww[:,None]).sum(0); numerator=(active*ww[:,None]*vv).sum(0)
  ref=np.divide(numerator,den,out=np.zeros_like(numerator),where=den>0)
  np.testing.assert_array_equal(va.numpy(),den>0)
  np.testing.assert_allclose(donor.numpy(),ref,rtol=1e-5,atol=2e-7)
  worst=max(worst,float(np.max(np.abs(donor.numpy()-ref))))
 checks['saved_state_readonly']={'signature_matches':True,'donor_numpy_max_abs_error':worst,
   'all_20_exclusions_checked':True,'warmup_checkpoint_round':warm['round'],'warmup_anchor_matches':True,'uploads':upload_stats,'sparse_upload_total_bytes_formula':sum(x['payload_bytes_formula'] for x in upload_stats),
   'dense_upload_total_bytes_formula':20*(64+4*anchor.numel())}

# Recompute every summary/GO metric without calling probe.summarize.
result_path=run/'result.json'
if result_path.exists():
 r=json.loads(result_path.read_text()); rows=r['rows']; assert len(rows)==20
 assert r['signature']==cfg['signature'] and r['status']=='DONE'
 assert sorted(x['client'] for x in rows)==list(range(20))
 actual_batches=[]
 for x in rows:
  i=x['client']; ids=x['train_batches']; n=len(tg[i]); amount=min(64,n//2)
  perm=np.random.default_rng(cfg['seed']+10000+i).permutation(n)
  assert ids['gradient_positions']==perm[:amount].tolist()
  assert ids['check_positions']==perm[amount:2*amount].tolist()
  original_fit=tg[i][ids['gradient_positions']]; original_check=tg[i][ids['check_positions']]
  assert not np.intersect1d(original_fit,original_check).size
  assert not np.intersect1d(np.r_[original_fit,original_check],h).size
  if state_path.exists():
   _,av=p.donor_average(vectors,masks,weights,exclude=i)
   assert x['available_external']==int(av.sum())
  actual_batches.append({'client':i,'gradient_n':amount,'check_n':amount})
 checks['actual_recorded_train_batches_verified']=actual_batches
 independent={}
 for name in rows[0]['metrics']:
  aa=np.array([x['metrics'][name]['accuracy'] for x in rows],dtype=float)
  ll=np.array([x['metrics'][name]['loss'] for x in rows],dtype=float)
  nn_=np.array([x['metrics'][name]['samples'] for x in rows],dtype=float)
  origin=np.array([x['metrics']['local']['loss'] for x in rows],dtype=float)
  s={'client_macro_accuracy':float(sum(aa)/len(aa)),'client_macro_ce':float(sum(ll)/len(ll)),
    'sample_weighted_accuracy':float(np.dot(aa,nn_)/sum(nn_)),
    'sample_weighted_ce':float(np.dot(ll,nn_)/sum(nn_)),
    'fraction_clients_ce_improved_vs_local':float(np.count_nonzero(ll<origin-1e-8)/len(ll)),
    'fraction_clients_ce_harmed_vs_local':float(np.count_nonzero(ll>origin+1e-8)/len(ll))}
  for key,value in s.items(): assert abs(value-r['summary'][name][key])<1e-12
  assert all(int(x['metrics'][name]['samples'])==len(hg[x['client']]) for x in rows)
  independent[name]=s
 cc=independent['gate_external']
 go=all(cc['client_macro_ce']<independent[x]['client_macro_ce'] and cc['client_macro_accuracy']>=independent[x]['client_macro_accuracy']-.005 for x in ['local','extra_sgd'])
 decision='GO_TO_INDEPENDENT_CHECK' if go else 'STOP_THIS_EXACT_EXTERNAL_GATE'
 assert r['decision']==decision
 if 'saved_state_readonly' in checks:
  by_id={x['client']:x for x in checks['saved_state_readonly']['uploads']}
  for u in r['uploads']:
   z_=by_id[u['client']]
   assert u['payload_bytes']==z_['payload_bytes_formula'] and u['values']==z_['values']
   actual_pack_ids=[j for j,sec in enumerate(packs) if bool(masks[u['client']][sec].any())]
   assert len(set(u['pack_ids']))==k and set(u['pack_ids'])==set(actual_pack_ids)
 pairs={}
 for ctrl in ['local','extra_sgd','sparse_external','sparse_external_keephead','gate_inclusive']:
  dc=np.array([x['metrics']['gate_external']['loss']-x['metrics'][ctrl]['loss'] for x in rows])
  da=np.array([x['metrics']['gate_external']['accuracy']-x['metrics'][ctrl]['accuracy'] for x in rows])
  pairs[ctrl]={'macro_ce_delta_gate_minus_control':float(dc.mean()),'macro_accuracy_delta_pp':float(100*da.mean()),
    'ce_better_clients':int((dc<-1e-8).sum()),'ce_worse_clients':int((dc>1e-8).sum()),'ce_tied_clients':int((np.abs(dc)<=1e-8).sum())}
 checks['result_recomputed']={'status':r['status'],'rows':len(rows),'summary_matches_atol':1e-12,
  'independent_decision':decision,'summary':independent,'paired_client_differences':pairs,
  'actual_minimum_ties':{name:[x['client'] for x in rows if sum(y==min(x[name]['train_check_ce']) for y in x[name]['train_check_ce'])>1] for name in ['gate_external','gate_inclusive','extra_sgd']},
  'available_external_parameter_min_mean_max':[min(x['available_external'] for x in rows),float(np.mean([x['available_external'] for x in rows])),max(x['available_external'] for x in rows)],
  'gate_external_alpha_counts':{str(a):sum(x['gate_external']['alpha']==a for x in rows) for a in [0.,1.,.5,.25]},
  'extra_sgd_alpha_counts':{str(a):sum(x['extra_sgd']['alpha']==a for x in rows) for a in [0.,1.,.5,.25]}}
else:
 checks['result_recomputed']={'status':'RESULT_NOT_YET_PRESENT'}

checks['fixture_status']='PASS'
(HERE/'LOCAL_CHECK_FIXTURES.json').write_text(json.dumps(checks,indent=2,allow_nan=False)+'\n')
print(json.dumps({k:v for k,v in checks.items() if k not in ['source_hash_matches','saved_state_readonly']},indent=2))
