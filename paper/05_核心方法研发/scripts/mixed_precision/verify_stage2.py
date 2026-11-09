"""Independent Stage2 saved-encoding-statistic check, no training or author tests."""
from pathlib import Path
import hashlib
import json
import math
import numpy as np
import torch

torch.set_num_threads(1)
HERE=Path(__file__).resolve().parent;folder=HERE/'development_s20261181'
SHA='46f3601e5c107f192d359227b21aea2734cc5e798674f2dccbbdf46fc796d863'
assert hashlib.sha256((HERE/'codec.py').read_bytes()).hexdigest()==SHA
methods=['cal_topk','cal_uniform3','cal_mixed_mse','cal_mixed_sensitive']
report={'status':'PENDING','runs':{},'codec_sha256':SHA,'external_cross_model_audit':'REVIEW_UNAVAILABLE'}
manifest0=None;lengths=[256]*534+[170];N=136874;B=54808
for method in methods:
 d=folder/method
 if not (d/'result.json').exists():report['runs'][method]={'status':'NOT_STARTED'};continue
 cfg=json.loads((d/'config.json').read_text());result=json.loads((d/'result.json').read_text());quant=method!='cal_topk'
 c=cfg.copy();sig=c.pop('signature');assert hashlib.sha256(json.dumps(c,sort_keys=True).encode()).hexdigest()==sig==result['signature']
 assert cfg['source_sha256']==hashlib.sha256((HERE/'run.py').read_bytes()).hexdigest()
 assert cfg['plan_sha256']==hashlib.sha256((HERE/'EXPERIMENT_PLAN.md').read_bytes()).hexdigest()
 assert cfg['quantizer_sha256']==(SHA if quant else None)
 assert cfg['seed']==20261181 and cfg['rounds']==20 and cfg['topk']==6843 and cfg['application_budget_bytes']==B
 assert not cfg['official_test_loaded'] and not result['test_evaluated']
 m=cfg['data_manifest']
 if manifest0 is None:manifest0=m
 else:assert m==manifest0
 counts=m['client_label_counts'];tau=[math.ceil(x/64) for x in m['client_sizes']]
 rows=result['rows'];assert all(row['round']==i+1 for i,row in enumerate(rows))
 previous_norm=[0.]*20;payloads=[];eferrors=[];all_bits={2:0,3:0,4:0};code_seconds=[];candidatebounds=[]
 for row in rows:
  assert row['validation']['samples']==2000 and np.isfinite(row['validation']['loss'])
  assert row['byte_accounting']==('generated decoded quantizer bytes' if quant else 'analytical fixed layout')
  assert len(row['clients'])==20 and sorted(c['client'] for c in row['clients'])==list(range(20))
  for client in row['clients']:
   i=client['client'];assert client['optimizer_steps']==client['gradient_evaluations']==tau[i]
   assert client['train_label_counts']==counts[i]
   shift=(np.maximum(counts[i],1)**(-.25)).astype(np.float32)
   np.testing.assert_array_equal(np.array(client['calibration_shift'],dtype=np.float32),shift)
   payload=client['payload_bytes'];assert payload<=B
   payloads.append(payload);code_seconds.append(client['codec_seconds'])
   if not quant:assert payload==B
   else:
    dg=client['codec'];bits=dg['bitwidths'];assert len(bits)==535 and all(x in [2,3,4] for x in bits)
    expected=64+sum(5+(n*b+7)//8 for n,b in zip(lengths,bits))
    assert payload==expected==dg['wire_bytes']<=B
    assert dg['header_bytes']==64 and dg['block_metadata_bytes']==2675
    assert dg['packed_value_bytes']==payload-64-2675 and dg['unused_budget_bytes']==B-payload
    for b in [2,3,4]:
     count=bits.count(b);assert dg['bitwidth_counts'][str(b)]==count;all_bits[b]+=count
    if method=='cal_uniform3':assert payload==54067 and all(b==3 for b in bits)
    assert len(dg['scales'])==len(dg['block_mse'])==535
    assert np.isfinite(dg['scales']).all() and min(dg['scales'])>=0
    assert np.isfinite(dg['block_mse']).all() and min(dg['block_mse'])>=0
    np.testing.assert_allclose(dg['mse'],np.dot(lengths,dg['block_mse'])/N,rtol=1e-12,atol=1e-20)
    # u infinity norm <= delta L2 + previous residual L2; conservative FP32 bound.
    bound=4*np.finfo(np.float32).eps*max(1.,client['delta_norm']+previous_norm[i])
    err=dg['ef_conservation_max_abs_error'];assert np.isfinite(err) and 0<=err<=bound
    eferrors.append(err);candidatebounds.append(float(bound))
    assert all(x['normalized_mse_gain']>0 and x['added_bytes']>=0 for x in dg['upgrades'])
    # Every listed increment is to the next bitwidth with the exact packed byte delta.
    if method!='cal_uniform3':
     state=[2]*535;spent=64+sum(5+(n*2+7)//8 for n in lengths)
     for upgrade in dg['upgrades']:
      j=upgrade['block'];old=state[j];new=upgrade['to']
      assert upgrade['from']==old and new==old+1 and new<=4
      cost=(lengths[j]*new+7)//8-(lengths[j]*old+7)//8
      assert cost==upgrade['added_bytes'];spent+=cost;state[j]=new
     assert state==bits and spent==payload
    if method=='cal_mixed_mse':assert not dg['weighted_allocation'] and all(w==1 for w in dg['normalized_block_weights'])
    if method=='cal_mixed_sensitive':assert dg['weighted_allocation'] or dg['zero_importance_fallback']
   previous_norm[i]=client['residual_norm']
  assert row['upload_bytes']==sum(c['payload_bytes'] for c in row['clients'])
 entry={'status':result['status'],'rounds':len(rows),'gradients_total':sum(c['gradient_evaluations'] for row in rows for c in row['clients']),
  'upload_bytes_total':sum(row['upload_bytes'] for row in rows),'per_client_payload_min_max':[min(payloads),max(payloads)],
  'codec_seconds_total':sum(code_seconds),'round_iteration_seconds_total':sum(row['seconds'] for row in rows),
  'historical_wire_saved':False,'checks':'saved metadata and source/runtime reconstruction assertion; independent wire decoder checked on synthetic payloads'}
 if quant:entry.update({'ef_max_abs_error':max(eferrors),'ef_conservative_tolerance_min':min(candidatebounds),'bitwidth_counts_all_client_rounds':all_bits})
 ck=torch.load(d/'checkpoint.pt',map_location='cpu',weights_only=False)
 if len(ck['rows'])==len(rows):
  assert ck['signature']==sig and all(torch.isfinite(t).all() for t in ck['model'].values())
  assert len(ck['residuals'])==20 and all(torch.isfinite(t).all() for t in ck['residuals'])
  for i,res in enumerate(ck['residuals']):assert float(res.norm())==previous_norm[i]
  entry['checkpoint']='MATCH'
 else:entry['checkpoint']='ADVANCED_DURING_READ'
 if result['status']=='DONE' and len(rows)==20:
  entry['last5']={'rounds':[16,17,18,19,20],
   'ce':sum(row['validation']['loss'] for row in rows[-5:])/5,
   'accuracy':sum(row['validation']['accuracy'] for row in rows[-5:])/5}
 report['runs'][method]=entry
if all(report['runs'][m]['status']=='DONE' and report['runs'][m]['rounds']==20 for m in methods):
 report['status']='ALL_DONE';means={m:report['runs'][m]['last5'] for m in methods}
 C,U,M,T=[means[m] for m in ['cal_mixed_sensitive','cal_uniform3','cal_mixed_mse','cal_topk']]
 cp=C['ce']<min(U['ce'],M['ce']);ap=C['accuracy']>=U['accuracy']+.005;tp=C['accuracy']>=T['accuracy']-.005
 report['frozen_stage2_go']={'definition':'CE(sensitive)<min(CE(uniform3),CE(mse)); acc(sensitive)>=acc(uniform3)+.005; acc(sensitive)>=acc(TopK)-.005',
  'ce_pass':bool(cp),'accuracy_vs_uniform_pass':bool(ap),'accuracy_vs_topk_pass':bool(tp),
  'decision':'GO_TO_INDEPENDENT_CONFIRMATION' if cp and ap and tp else 'STOP_THIS_SENSITIVITY_ALLOCATION_RULE'}
 report['sensitive_minus_controls']={m:{'ce_delta':C['ce']-means[m]['ce'],'accuracy_delta_pp':100*(C['accuracy']-means[m]['accuracy'])} for m in ['cal_uniform3','cal_mixed_mse','cal_topk']}
(HERE/'QUANTIZATION_STATS_CHECK.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
