"""Synthetic-only final TEST protocol fixture; never opens official TEST bytes."""
from pathlib import Path
import copy
import gzip
import json
import pickle
import struct
import sys
import tempfile
import time

import numpy as np
import data as d
import scorer as s

HERE=Path(__file__).resolve().parent
started=time.perf_counter()
tmp=Path(tempfile.mkdtemp(prefix='predictive_final_test_cpu_'))
test_opens=[]
def audit(event,args):
    if event=='open' and isinstance(args[0],(str,bytes)):
        path=str(args[0])
        if 'test_batch' in path or 't10k-' in path or '/dataset/' in path or '/datasets/' in path:
            test_opens.append(path)
            raise AssertionError('fixture must not open official TEST/data paths')
sys.addaudithook(audit)
sources={str(HERE/name):d.file_sha(HERE/name) for name in ('FINAL_TEST_PLAN.md','data.py','scorer.py')}


def rejects(function):
    try:function()
    except (ValueError,pickle.UnpicklingError):return
    raise AssertionError('invalid input accepted')


rejects(lambda:d.load_cifar_test('/tmp/closed_final_loader'))
rejects(lambda:d.load_fashion_test('/tmp/closed_final_loader'))
bad_lock=tmp/'not_authorized.json'
bad_lock.write_text(json.dumps({'status':'AWAITING_ROOT_REVIEW'}))
rejects(lambda:d.load_cifar_test('/tmp/closed_final_loader',root_lock=bad_lock,execute=True))
rejects(lambda:d.load_fashion_test('/tmp/closed_final_loader',root_lock=bad_lock,execute=True))
assert not test_opens
rng=np.random.default_rng(20261301)
labels=np.repeat(np.arange(10),1000).astype(np.int64)
rng.shuffle(labels)
counts=rng.integers(0,501,size=(20,10),dtype=np.int64)
counts[0,:4]=0;counts[3,7]=0
manifest_hash=d.canonical_sha({'synthetic_train_counts':counts.tolist()})
groups,assignment=d.assign_clients(labels,counts,task='cifar',train_manifest_sha256=manifest_hash)
again,replayed=d.assign_clients(labels,counts,task='cifar',train_manifest_sha256=manifest_hash)
assert assignment['assignment_hash']==replayed['assignment_hash']
assert all(np.array_equal(a,b) for a,b in zip(groups,again))
assert np.array_equal(np.sort(np.concatenate(groups)),np.arange(10000))
quota,weights,denominator=d.largest_remainder(counts)
assert np.array_equal(quota.sum(axis=0),np.full(10,1000))
assert np.all(np.abs(quota*denominator-1000*weights)<denominator)
assert np.all(weights[counts==0]==1)
tie,_,_=d.largest_remainder(np.ones((20,10),dtype=np.int64),per_class=999)
assert np.all(tie[:19]==50) and np.all(tie[19]==49)
realized=np.array([np.bincount(labels[ids],minlength=10) for ids in groups])
assert np.array_equal(realized,quota)
train_prior=counts/counts.sum(axis=1)[:,None]
test_prior=quota/quota.sum(axis=1)[:,None]
assert np.array_equal(test_prior-train_prior,np.array(assignment['test_minus_train_class_frequency']))
rejects(lambda:d.assign_clients(labels[:-1],counts,task='cifar',train_manifest_sha256=manifest_hash))
rejects(lambda:d.largest_remainder(np.zeros((20,10),dtype=np.int64)))
empty_counts=np.ones((20,10),dtype=np.int64)
empty_counts[0]=1_000_000
rejects(lambda:d.assign_clients(labels,empty_counts,task='cifar',train_manifest_sha256=manifest_hash))

# Strict parsers use only in-memory synthetic payloads of official row shapes.
cifar_pixels=np.zeros((10000,3072),dtype=np.uint8)
cifar_pixels[:,0]=np.arange(10000,dtype=np.int64)%256
cifar_payload=pickle.dumps({'data':cifar_pixels,'labels':labels.tolist()},protocol=2)
cifar,cifar_labels,cifar_meta=d.parse_cifar_bytes(cifar_payload)
assert np.array_equal(cifar_labels,labels) and cifar.shape==(10000,32,32,3)
assert np.array_equal(cifar[:,0,0,0],cifar_pixels[:,0])
rejects(lambda:d.parse_cifar_bytes(cifar_payload+b'extra'))
rejects(lambda:d.parse_cifar_bytes(pickle.dumps({'data':cifar_pixels[:10],'labels':labels[:10].tolist()},protocol=2)))
rejects(lambda:d.parse_cifar_bytes(pickle.dumps(Path('/tmp/never_opened'),protocol=2)))
fashion_pixels=np.zeros((10000,28,28),dtype=np.uint8)
fashion_pixels[:,0,0]=np.arange(10000,dtype=np.int64)%256
image_idx=struct.pack('>IIII',2051,10000,28,28)+fashion_pixels.tobytes()
label_idx=struct.pack('>II',2049,10000)+labels.astype(np.uint8).tobytes()
fashion,fashion_labels,fashion_meta=d.parse_fashion_gzip_bytes(gzip.compress(image_idx),gzip.compress(label_idx))
assert np.array_equal(fashion,fashion_pixels) and np.array_equal(fashion_labels,labels)
rejects(lambda:d.parse_idx_bytes(image_idx[:-1],2051,(10000,28,28)))
rejects(lambda:d.parse_idx_bytes(image_idx,9999,(10000,28,28)))
rejects(lambda:d.parse_fashion_gzip_bytes(gzip.compress(image_idx+b'X'),gzip.compress(label_idx)))
for task,pixels in (('cifar',cifar[:2]),('fashion',fashion[:2])):
    untouched=pixels.copy()
    normalized=d.normalize_batch(pixels,task)
    if task=='cifar':reference=pixels.transpose(0,3,1,2).astype(np.float32)
    else:reference=np.repeat(np.pad(pixels,((0,0),(2,2),(2,2)))[:,None],3,axis=1).astype(np.float32)
    reference=(reference/np.float32(255)-np.float32(.5))/np.float32(.5)
    assert np.array_equal(normalized,reference) and np.array_equal(pixels,untouched)
    assert normalized.dtype==np.float32 and normalized.shape==(2,3,32,32)
    if task=='fashion':assert np.all(normalized[:,:,:2]==-1)

# Stable scores and last-closed confidence bins, including confidence1.
uniform=s.metrics(np.zeros((10,10)),np.arange(10))
assert abs(uniform['nll']-np.log(10))<1e-15 and abs(uniform['brier']-.9)<1e-15
assert uniform['ece15']<1e-15
perfect=s.metrics(np.eye(10)*1000,np.arange(10))
assert perfect['nll']==perfect['brier']==perfect['ece15']==0 and perfect['ece_bins']['counts'][-1]==10
wrong=s.metrics(np.array([[-1000.,1000.]+[0.]*8]),np.array([0]))
assert wrong['nll']==2000 and wrong['brier']==2 and wrong['ece15']==1
random_z=rng.normal(size=(127,10))*3;random_y=rng.integers(0,10,size=127)
observed=s.metrics(random_z,random_y)
exp=np.exp(random_z-random_z.max(axis=1)[:,None]);prob=exp/exp.sum(axis=1)[:,None]
manual_nll=float(np.mean(-np.log(prob[np.arange(127),random_y])))
manual_brier=float(np.mean(np.sum((prob-np.eye(10)[random_y])**2,axis=1)))
conf=prob.max(axis=1);correct=prob.argmax(axis=1)==random_y
manual_ece=0.
for k in range(15):
    selected=(conf>=k/15)&((conf<(k+1)/15) if k<14 else (conf<=1))
    if selected.any():manual_ece+=selected.mean()*abs(correct[selected].mean()-conf[selected].mean())
assert abs(observed['nll']-manual_nll)<1e-14 and abs(observed['brier']-manual_brier)<1e-14 and abs(observed['ece15']-manual_ece)<1e-14

states=[]
for client in range(20):
    prior=np.log((counts[client]+1.)/(counts[client].sum()+10)).astype(np.float32)
    centered=(prior.astype(np.float64)-prior.astype(np.float64).mean()).astype(np.float32)
    states.append({'analytic_prior':prior,'prioranchor_bias':centered,'zero_anchor_bias':np.zeros(10,dtype=np.float32),
        'temperature_scaling':np.ones(1,dtype=np.float32),'bcts':np.r_[np.float32(1),np.zeros(10,dtype=np.float32)],
        'zero_anchor_vector':np.r_[np.ones(10,dtype=np.float32),np.zeros(10,dtype=np.float32)],
        'prioranchored_vector':np.r_[np.ones(10,dtype=np.float32),centered],
        'prioranchored_bcts':np.r_[np.float32(1),centered]})
state_file=tmp/'synthetic_frozen_train_states.npz'
np.savez_compressed(state_file,**{f'client{i}_{k}':v for i,st in enumerate(states) for k,v in st.items()})
snapshot={str(state_file):d.file_sha(state_file)}
raw=rng.normal(size=(10000,10)).astype(np.float32)
heads=[raw[ids]+np.float32(.01)*client for client,ids in enumerate(groups)]
bad=copy.deepcopy(states);bad[0]['temperature_scaling']=bad[0]['temperature_scaling'].astype(np.float64)
rejects(lambda:s.validate_states(bad,counts))
bad=copy.deepcopy(states);bad[0].pop('bcts')
rejects(lambda:s.validate_states(bad,counts))
report=s.score_export(tmp/'export',task='cifar',seed=20261231,method='fedlc_uniform3',
    row_ids=np.arange(10000),raw_logits=raw,head_logits_by_client=heads,
    head_row_ids_by_client=groups,
    labels=labels,assignment=assignment,states=states,
    test_provenance={'scope':'SYNTHETIC_CPU_FIXTURE'},input_snapshot=snapshot,synthetic=True,
    inference_cost={'scope':'synthetic logits; no inference','cnn_forwards':0})
assert report['cost']['new_fit_calls']==report['cost']['new_cnn_backwards']==0
max_metric_error=0.;verified_scores=0
for row in report['clients']:
    client=row['client'];ids=groups[client]
    path=tmp/'export'/f'client{client}.npz'
    assert d.file_sha(path)==report['exports'][client]['sha256']
    expected=s.apply_states(raw[ids],heads[client],states[client])
    with np.load(path,allow_pickle=False) as saved:
        assert np.array_equal(saved['test_ids'],ids) and np.array_equal(saved['test_labels'],labels[ids])
        for variant in s.VARIANTS:
            assert np.array_equal(saved['score_'+variant],expected[variant])
            verified_scores+=expected[variant].size
            observed=s.metrics(saved['score_'+variant],saved['test_labels'])
            for metric in s.METRICS:max_metric_error=max(max_metric_error,abs(observed[metric]-row['metrics'][variant][metric]))
assert max_metric_error==0
reports=[{**report,'seed':seed} for seed in d.TASKS['cifar']['seeds']]
aggregate=s.summarize_three_seeds(reports,'cifar','fedlc_uniform3')
assert aggregate['summaries']['prioranchored_vector']['macro_nll']['sample_sd']==0
rejects(lambda:s.summarize_three_seeds(reports[:2],'cifar','fedlc_uniform3'))
bad_assignment=copy.deepcopy(assignment);bad_assignment['client_ids'][0][0]=0
rejects(lambda:s._validate_assignment(labels,bad_assignment))
assert not test_opens and all(d.file_sha(path)==sha for path,sha in sources.items())
result={'status':'PASS','scope':'10000 synthetic rows;1000 per class; no official TEST bytes or model inference',
    'fixture_root':str(tmp),'source_identities_unchanged':sources,
    'assignment_hash':assignment['assignment_hash'],'train_count_hash':assignment['train_counts_int64le_sha256'],
    'checks':['root-lock/execute closed before any TEST open','exact integer quota/remainder tie','all10000 assigned once',
        'missing-TRAIN-class smoothing','deterministic per-class PCG64 replay','realized-vs-TRAIN frequency difference export',
        'empty client explicit refusal','bounded in-memory CIFAR/IDX parsers and malformed rejection',
        'both input normalizations','uniform/perfect/wrong stable metric known cases and confidence1 closed bin',
        'complete FP32 states/no fitting','20client NPZ score/ID/GT/state export','fixed-three-seed sampleSD'],
    'verified_final_score_elements':verified_scores,'recomputed_metric_max_error':max_metric_error,
    'official_test_open_attempts':test_opens,'real_cnn_forwards':0,'cnn_backwards':0,'fit_calls':0,
    'root_lock_created':False,'official_final_run_started':False,'elapsed_seconds':time.perf_counter()-started,
    'runtime':d.runtime(),'harness_sha256':d.file_sha(__file__)}
(HERE/'LOCAL_FIXTURES.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({key:result[key] for key in ('status','fixture_root','verified_final_score_elements',
    'official_test_open_attempts','real_cnn_forwards','fit_calls','root_lock_created','elapsed_seconds')},indent=2))
