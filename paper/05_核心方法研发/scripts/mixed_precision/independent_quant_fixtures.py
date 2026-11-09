"""Small independent wire/quantization fixtures; no training or author tests."""
from pathlib import Path
import hashlib
import importlib.util
import json
import struct
import numpy as np
import torch

torch.set_num_threads(1)
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('frozen_codec_independent',HERE/'codec.py')
c=importlib.util.module_from_spec(spec)
import sys
sys.modules[spec.name]=c;spec.loader.exec_module(c)
EXPECTED='46f3601e5c107f192d359227b21aea2734cc5e798674f2dccbbdf46fc796d863'
assert hashlib.sha256((HERE/'codec.py').read_bytes()).hexdigest()==EXPECTED

# Parse each block as a little-endian integer bitstream; no codec unpack helper.
def independent_decode(wire):
 magic=wire[:8];version,n,size,nblocks=struct.unpack_from('<4I',wire,8)
 assert magic==b'MPQCODEC' and version==1 and size==256 and wire[24:64]==bytes(40)
 assert nblocks==(n+255)//256
 at=64;parts=[];blocks=[]
 for j in range(nblocks):
  count=min(256,n-j*256);bits=wire[at];scale=np.float32(struct.unpack_from('<f',wire,at+1)[0]);at+=5
  nb=(count*bits+7)//8;payload=wire[at:at+nb];at+=nb
  integer=int.from_bytes(payload,'little');codes=[(integer>>(i*bits))&((1<<bits)-1) for i in range(count)]
  assert integer>>(count*bits)==0 and max(codes)<(1<<bits)-1
  levels=np.array(codes,dtype=np.int32)-((1<<(bits-1))-1)
  decoded=levels.astype(np.float32)*scale;parts.append(decoded)
  blocks.append({'count':count,'bits':bits,'scale':float(scale),'codes':codes,'value_bytes':nb,
    'decoded':decoded})
 assert at==len(wire)
 return np.concatenate(parts),blocks

# Independently enumerate FP32-scale initialization and four LS/refit candidates.
def ref_scalar_quant(values,bits):
 x=np.array(values,dtype=np.float32).astype(float);qmax=2**(bits-1)-1
 scale=np.float32(np.abs(x).max()/qmax);possibilities=[]
 for _ in range(5):
  level=np.zeros(len(x),dtype=np.int32) if scale==0 else np.minimum(qmax,np.maximum(-qmax,np.rint(x/float(scale)))).astype(np.int32)
  decoded=level.astype(np.float32)*scale
  distortion=float(np.mean((x-decoded.astype(float))**2))
  possibilities.append((distortion,float(scale),decoded.copy()))
  denominator=float(np.sum(level.astype(float)**2))
  scale=np.float32(np.sum(x*level)/denominator) if denominator else np.float32(0)
 return min(possibilities,key=lambda z:z[0]),possibilities

checks={}
hand=np.array([-1.8,-1.0,-.23,.02,.31,.74,1.4],dtype=np.float32)
wire=c.encode(torch.tensor(hand),64+5+(len(hand)*2+7)//8,'mixed_mse')
actual,blocks=independent_decode(wire.wire);best,allc=ref_scalar_quant(hand,2)
np.testing.assert_array_equal(actual,best[2]);assert blocks[0]['scale']==best[1]
assert best[0]<=allc[0][0] and max(blocks[0]['codes'])<=2
checks['scalar_endpoint_and_scale']={'input':hand.tolist(),'decoded':actual.tolist(),'scale':blocks[0]['scale'],
 'decoded_mse':best[0],'maxabs_initial_mse':allc[0][0],'symmetric_levels_2bit':[-1,0,1],
 'best_of_initial_plus_four_refits':True,'clipping_after_refit_allowed':True}

# One affordable upgrade across three different full blocks. Independently verify
# the chosen gain/byte, and that importance leaves each candidate quantizer intact.
b0=np.linspace(-1.3,1.1,256,dtype=np.float32)
b1=(4*np.sin(np.arange(256)*.17)).astype(np.float32)
b2=(.2*np.cos(np.arange(256)*.31)).astype(np.float32)
u=np.r_[b0,b1,b2].astype(np.float32);base=64+3*5+3*64;budget=base+32
enc_mse=c.encode(torch.tensor(u),budget,'mixed_mse')
importance=np.r_[np.full(256,100.,dtype=np.float32),np.zeros(512,dtype=np.float32)]
enc_sens=c.encode(torch.tensor(u),budget,'mixed_sensitive',torch.tensor(importance))
parsed=[];ratios=[]
for j,block in enumerate([b0,b1,b2]):
 old,_=ref_scalar_quant(block,2);new,_=ref_scalar_quant(block,3)
 ratios.append((old[0]-new[0])*(256/len(u))/32)
for result in [enc_mse,enc_sens]:
 dec,meta=independent_decode(result.wire)
 np.testing.assert_array_equal(dec,result.sent.numpy())
 parsed.append(meta)
 assert len(result.wire)==budget
winner=max(range(3),key=lambda j:ratios[j]);assert winner==1
assert [x['bits'] for x in parsed[0]]==[2,3,2]
assert [x['bits'] for x in parsed[1]]==[3,2,2]
for meta in parsed:
 for j,block in enumerate([b0,b1,b2]):
  best,_=ref_scalar_quant(block,meta[j]['bits'])
  assert meta[j]['scale']==best[1]
  np.testing.assert_array_equal(meta[j]['decoded'],best[2])
checks['greedy_and_importance_only_allocation']={'unweighted_gain_per_byte':ratios,
 'mse_widths':[x['bits'] for x in parsed[0]],'sensitive_widths':[x['bits'] for x in parsed[1]],
 'same_candidate_scale_and_decoded_at_each_fixed_bitwidth':True,'actual_upgrade_bytes':32}

# Actual model-length payload: independently sum all headers and partial blocks.
N=136874;B=54808
rng=np.random.default_rng(66);large=(rng.normal(size=N)*.03).astype(np.float32)
full=[]
for mode in ['uniform3','mixed_mse','mixed_sensitive']:
 weights=torch.tensor(np.linspace(.2,2,N,dtype=np.float32)) if mode=='mixed_sensitive' else None
 result=c.encode(torch.tensor(large),B,mode,weights)
 decoded,meta=independent_decode(result.wire)
 np.testing.assert_array_equal(decoded,result.sent.numpy())
 expected_bytes=64+sum(5+(m['count']*m['bits']+7)//8 for m in meta)
 assert expected_bytes==len(result.wire)<=B and len(meta)==535 and meta[-1]['count']==170
 if mode=='uniform3':assert len(result.wire)==54067
 expected_residual=large-decoded
 np.testing.assert_array_equal(result.residual.numpy(),expected_residual)
 ef=((decoded+expected_residual).astype(float)-large.astype(float))
 tolerance=4*np.finfo(np.float32).eps*max(1.,float(np.abs(large).max()))
 assert float(np.max(np.abs(ef)))<=tolerance
 assert result.diagnostics['ef_conservation_max_abs_error']==float(np.max(np.abs(ef)))
 full.append({'mode':mode,'wire_bytes':len(result.wire),'blocks':len(meta),'tail_values':170,
  'ef_max_abs_error':float(np.max(np.abs(ef))),'ef_tolerance':float(tolerance),
  'bitwidths':{str(b):sum(m['bits']==b for m in meta) for b in [2,3,4]}})
checks['full_model_integer_decoder_and_byte_sum']=full
checks['codec_sha256']=EXPECTED;checks['status']='PASS';checks['external_cross_model_audit']='REVIEW_UNAVAILABLE'
(HERE/'QUANTIZATION_FIXTURE_EVIDENCE.json').write_text(json.dumps(checks,indent=2)+'\n')
print(json.dumps(checks,indent=2))
