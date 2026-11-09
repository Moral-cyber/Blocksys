"""Read-only CPU protocol/utility fixtures; never call training or main."""
from pathlib import Path
import hashlib
import importlib.util
import json
import struct
import numpy as np
import torch
from torch import nn

torch.set_num_threads(1)
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('receiver_frozen_check',HERE/'run.py')
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
checks={}

# Independent decoder for the exact encoder format. Missing values stay masked out.
def decode_model(wire):
 rnd,n,size,k=struct.unpack_from('<4I',wire,0)
 assert size==128 and n>0
 out=np.zeros(n,dtype=np.float32);mask=np.zeros(n,dtype=bool);ids=[];at=16
 for _ in range(k):
  j,=struct.unpack_from('<I',wire,at);at+=4
  assert j<(n+size-1)//size and j not in ids
  ids.append(j);lo=j*size;hi=min(lo+size,n);count=hi-lo
  out[lo:hi]=np.frombuffer(wire,dtype='<f4',count=count,offset=at)
  mask[lo:hi]=True;at+=4*count
 assert ids==sorted(ids) and at==len(wire)
 return out,mask,{'round':rnd,'n':n,'pack_size':size,'ids':ids,'bytes':at}

def decode_feedback(wire):
 rnd,n,version,count=struct.unpack_from('<4I',wire,0)
 assert version==1 and count==(n+7)//8 and len(wire)==16+count
 allbits=np.unpackbits(np.frombuffer(wire,dtype=np.uint8,offset=16),bitorder='little')
 assert not allbits[n:].any()
 return allbits[:n].astype(bool),{'round':rnd,'bits':n,'payload_bytes':count,'bytes':len(wire)}

# Nonmonotone choice order, zero and negative values, partial last pack, real model size.
N=136874; packs=r.probe.pack_slices(N,128)
vec=torch.arange(N,dtype=torch.float32)/256-100
for chosen in [[1069,0,317],[1069],[0],[],list(range(53))]:
 wire=r.encode_model(vec,chosen,packs,7)
 vals,mask,meta=decode_model(wire)
 expected=np.zeros(N,dtype=bool)
 for j in chosen:expected[packs[j]]=True
 np.testing.assert_array_equal(mask,expected)
 np.testing.assert_array_equal(vals[expected],vec.numpy()[expected])
 assert len(wire)==16+4*len(chosen)+4*int(expected.sum())
 bits=np.zeros(len(packs),dtype=bool);bits[chosen]=True
 bwire=r.encode_feedback(bits,7);bb,bmeta=decode_feedback(bwire)
 np.testing.assert_array_equal(bb,bits)
 assert bmeta['bytes']==150
checks['model_encoder_independent_decoder']={'status':'PASS','cases':5,'parameters':N,'packs':len(packs),'last_pack_parameters':42,'feedback_bytes':150,'mixed_full_short_zero_negative_values':True}
# Explicit bit order edges and two padding bits.
bits=np.zeros(1070,dtype=bool);bits[[0,7,8,1069]]=True
wire=r.encode_feedback(bits,11);bb,meta=decode_feedback(wire)
assert wire[16]==129 and wire[17]==1 and wire[-1]==32
np.testing.assert_array_equal(bb,bits)
checks['feedback_bit_order']={'edge_positions':[0,7,8,1069],'first_bytes':[wire[16],wire[17]],'last_byte':wire[-1],'padding_bits':2}

# Exact common application byte cap; neither full/short mix may exceed it.
cap=16+53*(4+128*4)
mag_full=len(r.encode_model(vec,range(53),packs,0))
fb_full=len(r.encode_model(vec,range(52),packs,0))+len(r.encode_feedback(np.zeros(1070),0))
mag_short=len(r.encode_model(vec,list(range(52))+[1069],packs,0))
fb_short=len(r.encode_model(vec,list(range(51))+[1069],packs,0))+150
assert mag_full==cap and fb_full<=cap and mag_short<=cap and fb_short<=cap
checks['upload_byte_cap']={'common_cap_53_full_packs':cap,'magnitude_53_full':mag_full,'feedback_52_full_plus_bitset':fb_full,'magnitude_including_short':mag_short,'feedback_including_short_plus_bitset':fb_short,'equal_actual_bytes':False,'common_warmup_formula_bytes_unicast':10*20*2*(16+4*N)}

# Receiver decision rule identical for both communication variants; local SGD set differs.
class CountLinear(nn.Linear):
 def __init__(self):super().__init__(1,2);self.calls=0
 def forward(self,x):self.calls+=1;return super().forward(x)
model=CountLinear();model.eval();local=r.base.vectorize(model)
grad=torch.tensor([.1,-.2,.3,-.4]);direction=torch.tensor([-.2,.1,.4,-.3])
check=(torch.tensor([[1.],[2.]]),torch.tensor([0,1]))
outputs=[]
for method in ['magnitude_gate','receiver_feedback','local_sgd']:
 model.calls=0
 point,choice,losses=r.choose(model,local,grad,direction,check,method)
 assert model.calls==4
 outputs.append((point,choice,losses))
assert torch.equal(outputs[0][0],outputs[1][0]) and outputs[0][1:]==outputs[1][1:]
model.calls=0
_,tie_choice,tie_losses=r.choose(model,local,torch.zeros_like(grad),torch.zeros_like(direction),check,'receiver_feedback')
assert tie_choice==0 and len(set(tie_losses))==1 and model.calls==4
checks['receiver_choice_fixture']={'communication_rules_equal':True,'forwards_per_choose':4,'ties_choose_local':True,'local_sgd_has_different_candidates':True}

# Reproduce the precise update using an independent receiver-loop expectation.
# Unobserved pack stays at its old value; all-zero *observed* pack decays.
accepted=np.array([[1,1,0,0],[0,1,0,0],[1,0,0,0]],dtype=bool)
validpacks=np.array([1,1,1,0],dtype=bool)
old=np.array([[.2,.3,.4,.6],[.5,.6,.7,.8],[.8,.9,.1,.2]],dtype=float)
actual=old.copy()
for i in range(len(accepted)):
 obs=(accepted.sum(axis=0)-accepted[i].astype(int))/(len(accepted)-1)
 actual[i,validpacks]=.75*actual[i,validpacks]+.25*obs[validpacks]
expected=old.copy()
for i in range(3):
 for j in range(4):
  if validpacks[j]:
   other=[float(accepted[z,j]) for z in range(3) if z!=i]
   expected[i,j]=.75*old[i,j]+.25*sum(other)/len(other)
np.testing.assert_array_equal(actual,expected)
assert np.array_equal(actual[:,3],old[:,3]) and np.all(actual[:,2]<old[:,2])
checks['utility_fixture']={'status':'PASS','updated':actual.tolist(),'self_receiver_excluded':True,'observed_zero_decays':True,'unobserved_unchanged':True,'utility_is_aggregate_reception_proxy':True}

# Static source identity; data IDs/checkpoint read only, no image/test loading.
source=HERE.parent/'partial_fusion/dev_seed20261143'
cfg=json.loads((source/'config.json').read_text());warm=torch.load(source/'warmup.pt',map_location='cpu',weights_only=False)
assert warm['signature']==cfg['signature'] and warm['round']==10
m=cfg['data_manifest'];train=np.concatenate(m['train_indices']);dev=np.concatenate(m['holdout_indices'])
assert len(train)==37500 and len(dev)==12500 and not np.intersect1d(train,dev).size
assert np.array_equal(np.sort(np.r_[train,dev]),np.arange(50000))
checks['source_warmup']={'round':10,'signature_matches':True,'train_dev_disjoint':True,'train_n':len(train),'dev_n':len(dev),'warmup_sha256':r.probe.sha(source/'warmup.pt')}
checks['current_source_hashes']={'run.py':r.probe.sha(HERE/'run.py'),'PLAN.md':r.probe.sha(HERE/'PLAN.md')}
checks['status']='PASS'
(HERE/'LOCAL_FIXTURE_EVIDENCE.json').write_text(json.dumps(checks,indent=2)+'\n')
print(json.dumps(checks,indent=2))
