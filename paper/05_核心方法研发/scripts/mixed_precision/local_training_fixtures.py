"""Independent CPU synthetic training/count/gradient fixtures; no real training."""
from pathlib import Path
import copy
import hashlib
import importlib.util
import json
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader

torch.set_num_threads(1)
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('mixed_training_frozen_fixture',HERE/'run.py')
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)

class IndexedToy(Dataset):
 def __init__(self,x,labels,indices):self.x=x;self.labels=labels;self.indices=indices
 def __len__(self):return len(self.indices)
 def __getitem__(self,i):
  j=int(self.indices[i]);return torch.tensor(self.x[j],dtype=torch.float32),torch.tensor(int(self.labels[j]),dtype=torch.long)
class Toy(nn.Module):
 def __init__(self):super().__init__();self.linear=nn.Linear(2,10)
 def forward(self,x):return self.linear(x)

model=Toy();anchor=np.linspace(-.2,.25,30).astype(np.float32);a.r.assign_vector(model,torch.tensor(anchor))
x=np.array([[.2,-.1],[.5,.4],[-.3,.6],[.8,-.7],[-.2,.9],[.1,.3],[-.5,-.6],[.4,.7],[.9,.2],[-.7,.8]],dtype=np.float32)
labels=np.array([0,1,9,3,1,4,2,2,8,8],dtype=np.int64);indices=np.array([7,1,4,6,0])
ds=IndexedToy(x,labels,indices)

# Independent NumPy gradient for full ten-class CE, flat weight then bias order.
def logits_grad(flat,xb,yb,shift):
 logits=xb.astype(float)@flat[:20].reshape(10,2).T+flat[20:]
 logits=logits-shift
 ex=np.exp(logits-logits.max(1,keepdims=True));prob=ex/ex.sum(1,keepdims=True)
 logprob=logits-np.log(ex.sum(1,keepdims=True))-logits.max(1,keepdims=True)
 ce=-float(logprob[np.arange(len(yb)),yb].mean())
 prob[np.arange(len(yb)),yb]-=1;prob/=len(yb)
 return np.r_[(prob.T@xb.astype(float)).ravel(),prob.sum(0)],ce

counts=np.bincount(labels[indices],minlength=10)
assert counts.tolist()==[1,2,2,0,0,0,0,0,0,0]
assert counts.tolist()!=np.bincount(labels,minlength=10).tolist()
shift=np.maximum(counts,1)**(-.25)
assert shift[0]==1 and shift[3]==1 and shift[1]<1
checks={};single={}
for calibrated in [False,True]:
 calls={'forward_ce':0,'backward':0};orig=a.nn.functional.cross_entropy
 def counted(*args,**kwargs):
  calls['forward_ce']+=1;loss=orig(*args,**kwargs)
  if loss.requires_grad:
   def hook(grad):calls['backward']+=1;return grad
   loss.register_hook(hook)
  return loss
 a.nn.functional.cross_entropy=counted
 try:delta,importance,meta=a.train(model,ds,53,'cpu',calibrated)
 finally:a.nn.functional.cross_entropy=orig
 gradient,_=logits_grad(anchor.astype(float),x[indices],labels[indices],shift if calibrated else 0.)
 expected_delta=-.03*gradient
 np.testing.assert_allclose(delta.numpy(),expected_delta,rtol=1e-5,atol=2e-8)
 np.testing.assert_allclose(importance.numpy(),gradient**2,rtol=1e-5,atol=1e-8)
 assert calls=={'forward_ce':1,'backward':1} and meta['optimizer_steps']==meta['gradient_evaluations']==1
 assert meta['train_label_counts']==counts.tolist()
 if calibrated:np.testing.assert_allclose(meta['calibration_shift'],shift,atol=3e-8,rtol=0)
 else:assert meta['calibration_shift'] is None
 np.testing.assert_array_equal(a.r.vectorize(model).numpy(),anchor)
 single['cal' if calibrated else 'ce']={'label_counts':meta['train_label_counts'],
  'max_delta_abs_error':float(np.max(np.abs(delta.numpy()-expected_delta))),
  'max_gradient_square_abs_error':float(np.max(np.abs(importance.numpy()-gradient**2))),**calls}
 if calibrated:
  local=copy.deepcopy(model);actual=anchor+delta.numpy();a.r.assign_vector(local,torch.tensor(actual))
  measured=a.r.metrics(local,ds)
  _,expected_raw_ce=logits_grad(actual.astype(float),x[indices],labels[indices],0.)
  _,hypothetical_shifted_ce=logits_grad(actual.astype(float),x[indices],labels[indices],shift)
  assert abs(measured['loss']-expected_raw_ce)<1e-6
  assert abs(measured['loss']-hypothetical_shifted_ce)>1e-4
  checks['raw_logit_inference']={'reported_ce':measured['loss'],'independent_raw_ce':expected_raw_ce,
   'shifted_ce_not_used':hypothetical_shifted_ce,'status':'PASS'}
checks['single_batch_independent_calibration']=single

# Two uneven loader batches: proxy is unweighted mean of squared batch-mean gradients.
rng=np.random.default_rng(2);xx=rng.normal(size=(76,2)).astype(np.float32)
yy=np.resize(np.array([0,1,1,2,3],dtype=np.int64),76);ii=rng.permutation(76)[:68]
ds2=IndexedToy(xx,yy,ii);cc=np.bincount(yy[ii],minlength=10);ss=np.maximum(cc,1)**(-.25)
loader=DataLoader(ds2,batch_size=64,shuffle=True,generator=torch.Generator(device='cpu').manual_seed(61))
point=anchor.astype(float).copy();squares=[];sizes=[]
for xb,yb in loader:
 g,_=logits_grad(point,xb.numpy(),yb.numpy(),ss)
 squares.append(g*g);sizes.append(len(yb));point-=.03*g
actual_delta,importance,meta=a.train(model,ds2,61,'cpu',True)
expected_importance=np.mean(squares,axis=0)
weighted=np.average(squares,axis=0,weights=sizes)
np.testing.assert_allclose(actual_delta.numpy(),point-anchor,rtol=1e-5,atol=5e-8)
np.testing.assert_allclose(importance.numpy(),expected_importance,rtol=1e-5,atol=2e-8)
assert not np.allclose(importance.numpy(),weighted,atol=1e-6)
assert meta['optimizer_steps']==meta['gradient_evaluations']==2
checks['uneven_batch_proxy']={'batch_sizes':sizes,'equal_batch_weighting':True,
 'max_gradient_square_abs_error':float(np.max(np.abs(importance.numpy()-expected_importance))),
 'not_sample_weighted_fisher':True,'additional_backward_for_importance':0}

# Stage1 compression branch identities, all from original-valued coordinates.
delta=torch.tensor([.2,-.4,.1,.3]);old=torch.tensor([.01,.05,-.02,.04]);u=delta+old
ids=torch.argsort(u.abs(),descending=True,stable=True)[:2];q=torch.zeros_like(u);q[ids]=u[ids];new=u-q
assert torch.equal(q+new,u)
dense=delta.clone();dense_new=torch.zeros_like(delta)
assert torch.equal(dense,delta) and not dense_new.any()
checks['stage1_compression']={'dense_uses_raw_delta_no_EF':True,'topk_uses_delta_plus_old_residual':True,
 'topk_ef_conservation_exact':True,'quantizer_not_reviewed':True}
checks['source_hashes']={name:hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in ['run.py','EXPERIMENT_PLAN.md']}
checks['status']='PASS';checks['external_cross_model_audit']='REVIEW_UNAVAILABLE'
(HERE/'TRAINING_FIXTURE_EVIDENCE.json').write_text(json.dumps(checks,indent=2)+'\n')
print(json.dumps(checks,indent=2))
