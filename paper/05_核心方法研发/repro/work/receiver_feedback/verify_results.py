"""Independent read-only current-result checker. No training/evaluation calls."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import torch

torch.set_num_threads(1)
HERE=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=HERE/'development_s20261144');args=p.parse_args()
N=136874;P=1070;C=20;K={'local_sgd':0,'magnitude_gate':53,'receiver_feedback':52}
source=json.loads((HERE.parent/'partial_fusion/dev_seed20261143/config.json').read_text())
original=source['data_manifest'];train=[np.array(x) for x in original['train_indices']];dev=[np.array(x) for x in original['holdout_indices']]
warm_sha=hashlib.sha256((HERE.parent/'partial_fusion/dev_seed20261143/warmup.pt').read_bytes()).hexdigest()
status={};details={};all_done=True

def model_bytes(ids):
 return 16+sum(4+4*min(128,N-j*128) for j in ids)

def unpack(s):
 raw=bytes.fromhex(s);assert len(raw)==134
 bits=np.unpackbits(np.frombuffer(raw,dtype=np.uint8),bitorder='little')
 assert not bits[P:].any()
 return bits[:P].astype(bool)

for method in K:
 folder=args.out/method;rp=folder/'result.json';cp=folder/'config.json'
 if not rp.exists():
  status[method]={'status':'NOT_STARTED'};all_done=False;continue
 r=json.loads(rp.read_text());cfg=json.loads(cp.read_text());rows=r['rows']
 # Recompute config identity; do not trust matching labels alone.
 c=cfg.copy();signature=c.pop('signature')
 assert hashlib.sha256(json.dumps(c,sort_keys=True).encode()).hexdigest()==signature==r['signature']
 assert cfg['method']==method and cfg['rounds']==20 and cfg['seed']==20261144
 assert cfg['source_sha256']==hashlib.sha256((HERE/'run.py').read_bytes()).hexdigest()
 assert cfg['plan_sha256']==hashlib.sha256((HERE/'PLAN.md').read_bytes()).hexdigest()
 assert cfg['warmup_sha256']==warm_sha
 assert cfg['data_manifest']['train_indices']==original['train_indices']
 assert cfg['data_manifest']['holdout_indices']==original['holdout_indices']
 assert all(x['round']==i+1 for i,x in enumerate(rows))
 status[method]={'status':r['status'],'completed_rounds':len(rows)}
 all_done=all_done and r['status']=='DONE' and len(rows)==20
 macro=[];up_total=down_total=0;utility=np.full((C,P),.5)
 for row in rows:
  clients=row['clients'];assert sorted(x['client'] for x in clients)==list(range(C))
  ns=np.array([x['samples'] for x in clients]);acc=np.array([x['accuracy'] for x in clients]);ce=np.array([x['ce'] for x in clients])
  summary={'client_macro_accuracy':float(acc.sum()/C),'client_macro_ce':float(ce.sum()/C),
    'sample_weighted_accuracy':float(np.dot(ns,acc)/ns.sum()),'sample_weighted_ce':float(np.dot(ns,ce)/ns.sum())}
  for key,value in summary.items():assert abs(value-row[key])<1e-12
  macro.append(summary)
  masks=np.stack([unpack(s) for s in row['accepted_masks_packbits_hex']]);assert masks.shape==(C,P)
  for client in clients:
   i=client['client'];assert client['samples']==len(dev[i])
   assert client['accepted_packs']==int(masks[i].sum())
   losses=client['training_candidate_ce'];assert len(losses)==4
   selected=min(range(4),key=lambda j:losses[j]);assert client['choice']==selected
   if method=='local_sgd' or selected in (0,1):assert not masks[i].any()
   n=len(train[i]);amount=min(64,n//2)
   permutation=np.random.default_rng(cfg['seed']+200000+(row['round']-1)*100+i).permutation(n)
   ids=client['train_batches']
   assert ids['gradient_positions']==permutation[:amount].tolist()
   assert ids['check_positions']==permutation[amount:2*amount].tolist()
   assert not np.intersect1d(train[i][ids['gradient_positions']],train[i][ids['check_positions']]).size
  if method=='local_sgd':
   assert not row['uploads'] and not row['upload_payload_sha256']
   expect_up=expect_down=0
  else:
   uploads=row['uploads'];assert len(uploads)==C and len(row['upload_payload_sha256'])==C
   for ids in uploads:assert len(ids)==K[method] and len(set(ids))==len(ids) and all(0<=x<P for x in ids)
   union=sorted(set(x for ids in uploads for x in ids));valid=np.zeros(P,dtype=bool);valid[union]=True
   assert not np.any(masks[:,~valid])
   # A receiver's accepted pack must include at least one external uploader.
   for i in range(C):
    others=set(j for z,ids in enumerate(uploads) if z!=i for j in ids)
    assert set(np.flatnonzero(masks[i])).issubset(others)
   expect_up=sum(model_bytes(ids) for ids in uploads)+(C*150 if method=='receiver_feedback' else 0)
   expect_down=C*model_bytes(union)
   assert all(model_bytes(ids)+(150 if method=='receiver_feedback' else 0)<=27364 for ids in uploads)
   if method=='receiver_feedback':
    # Independent pack/receiver loop, excluding receiver i; update only global observed support.
    for i in range(C):
     observations=np.array([np.mean([float(masks[z,j]) for z in range(C) if z!=i]) for j in union])
     utility[i,union]=.75*utility[i,union]+.25*observations
  assert row['upload_bytes']==expect_up and row['download_bytes_unicast']==expect_down
  up_total+=expect_up;down_total+=expect_down
  assert row['cumulative_upload_bytes']==up_total and row['cumulative_download_bytes_unicast']==down_total
  assert row['common_warmup_bytes_unicast']==219004800
 # Only load a checkpoint whose completed rows match the inspected result snapshot.
 ck=torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=False)
 if len(ck['rows'])==len(rows):
  assert ck['signature']==signature and ck['total_up']==up_total and ck['total_down']==down_total
  np.testing.assert_allclose(ck['utility'],utility,rtol=0,atol=1e-15)
  utility_status='MATCH'
 else:
  utility_status='CHECKPOINT_ADVANCED_DURING_READ'
 details[method]={'all_observed_round_summaries_match_atol':1e-12,'utility_checkpoint':utility_status,
  'warmup_bytes_formula':219004800,'post_warmup_upload_bytes':up_total,'post_warmup_download_bytes_unicast':down_total,
  'total_communication_bytes_including_warmup_formula':219004800+up_total+down_total}
 if r['status']=='DONE' and len(rows)==20:
  last5={key:float(np.mean([x[key] for x in macro[-5:]])) for key in macro[0]}
  details[method]['last5_rounds']=[16,17,18,19,20];details[method]['last5']=last5
  details[method]['round20']=macro[-1]
  # Fixed-client last5 primary pairing; useful description, not independent replications.
  client_last5={}
  for i in range(C):
   client_last5[str(i)]={key:float(np.mean([next(c for c in row['clients'] if c['client']==i)[key] for row in rows[-5:]])) for key in ['ce','accuracy']}
  details[method]['client_last5']=client_last5
report={'status':'ALL_DONE' if all_done else 'PARTIAL','runs':status,'checks':details,'external_cross_model_audit':'REVIEW_UNAVAILABLE'}
if all_done:
 fb=details['receiver_feedback']['last5'];mag=details['magnitude_gate']['last5']
 dce=fb['client_macro_ce']-mag['client_macro_ce'];dacc=fb['client_macro_accuracy']-mag['client_macro_accuracy']
 ce_pass=fb['client_macro_ce']<mag['client_macro_ce'];acc_pass=fb['client_macro_accuracy']>=mag['client_macro_accuracy']+.005
 report['independent_frozen_go']={'definition':'last5_mean_macro_CE(feedback)<magnitude AND last5_mean_macro_accuracy(feedback)>=magnitude+.005',
  'ce_pass':bool(ce_pass),'accuracy_pass':bool(acc_pass),'ce_delta_feedback_minus_magnitude':dce,'accuracy_delta_pp':100*dacc,
  'decision':'GO_TO_INDEPENDENT_REPLICATIONS' if ce_pass and acc_pass else 'STOP_THIS_FIXED_FEEDBACK_RULE'}
 dc=np.array([details['receiver_feedback']['client_last5'][str(i)]['ce']-details['magnitude_gate']['client_last5'][str(i)]['ce'] for i in range(C)])
 report['paired_last5_clients']={'ce_better':int((dc<-1e-8).sum()),'ce_worse':int((dc>1e-8).sum()),'ce_tied':int((np.abs(dc)<=1e-8).sum())}
(HERE/'LOCAL_STATS_CHECK.json').write_text(json.dumps(report,indent=2)+'\n')
compact={'status':report['status'],'runs':status,'checks':{k:{a:b for a,b in v.items() if a!='client_last5'} for k,v in details.items()}}
if 'independent_frozen_go' in report:compact['independent_frozen_go']=report['independent_frozen_go']
print(json.dumps(compact,indent=2))
