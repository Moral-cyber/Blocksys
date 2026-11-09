from pathlib import Path
from types import SimpleNamespace
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import struct
import sys
import tempfile
import time

import numpy as np
import torch
from torch.utils.data import TensorDataset

ROOT=Path(__file__).resolve().parents[2]
HERE=ROOT/'work/head_control'
EXPECTED='8ee487ea523ee772a99e3555174cc9f41d2c9e1ade300f58b3df778562c63b82'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(HERE/'run.py')==EXPECTED
source_before={str(p.relative_to(ROOT)):sha(p) for p in (HERE/'run.py',HERE/'PLAN.md',HERE.parent/'mixed_precision/codec.py')}
spec=importlib.util.spec_from_file_location('_head_control_local_fixtures',HERE/'run.py')
m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
torch.set_num_threads(1)
OUT=Path(tempfile.mkdtemp(prefix='head_control_local_fixtures_'))
started=time.perf_counter()
report={'scope':'independent CPU implementation fixtures on synthetic tensors only; no official data or efficacy result', 'source_identities':source_before,'run_source_expected':EXPECTED,'fixture_out':str(OUT),'dataset_open_attempts':[],'checks':{}}
def audit(event,args):
    if event=='open' and isinstance(args[0],(str,bytes)) and '/dataset/' in str(args[0]):
        report['dataset_open_attempts'].append(str(args[0]))
        raise AssertionError('official dataset file open forbidden in local fixtures')
sys.addaudithook(audit)
def maximum(x):return float(x.abs().max())
def digest_tensor(x):return hashlib.sha256(x.detach().cpu().numpy().tobytes()).hexdigest()
class LoggedDataset(TensorDataset):
    def __init__(self,*values):super().__init__(*values);self.order=[]
    def __getitem__(self,index):self.order.append(int(index));return super().__getitem__(index)
def make_dataset(n,seed):
    gen=torch.Generator().manual_seed(seed)
    return LoggedDataset(torch.randn(n,3,32,32,generator=gen),torch.randint(0,10,(n,),generator=gen))

torch.manual_seed(20261191);model=m.r.SmallCNN();anchor=m.r.vectorize(model).clone();h=m.head_size(model);body=anchor.numel()-h
assert (h,body,anchor.numel())==(650,136224,136874)
assert [name for name,_ in list(model.named_parameters())[-2:]]==['classifier.3.weight','classifier.3.bias']
report['checks']['layout']={'head':h,'body':body,'total':anchor.numel(),'head_names':['classifier.3.weight','classifier.3.bias'],'last_coordinates':True}
# Capture backward hooks before the runner's in-place correction, then inspect
# the gradients actually consumed by SGD. No runner source is modified.
ds=make_dataset(67,801);ci=torch.linspace(-.01,.01,h);c=torch.linspace(.003,-.003,h)
raw={};steps=[];holder={};oldcopy=copy.deepcopy;oldstep=torch.optim.SGD.step

def tracked_copy(obj,*args,**kwargs):
    local=oldcopy(obj,*args,**kwargs)
    if obj is model:
        holder['local']=local
        for name,p in local.named_parameters():
            p.register_hook(lambda grad,name=name:raw.__setitem__(name,grad.detach().clone()))
    return local

def tracked_step(opt,*args,**kwargs):
    params=list(holder['local'].named_parameters());head_names={name for name,_ in params[-2:]}
    raw_head=torch.cat([raw[name].flatten() for name,p in params if name in head_names])
    adjusted=torch.cat([p.grad.flatten() for name,p in params if name in head_names])
    steps.append({'raw_head':raw_head.clone(),'body_added_max':max(maximum(p.grad-raw[name]) for name,p in params if name not in head_names),'head_added_error':maximum((adjusted-raw_head)-(c-ci))})
    return oldstep(opt,*args,**kwargs)
copy.deepcopy=tracked_copy;torch.optim.SGD.step=tracked_step
try:delta,newci,k,err=m.train(model,ds,20261191,'cpu',ci,c)
finally:copy.deepcopy=oldcopy;torch.optim.SGD.step=oldstep
rawmean=torch.stack([row['raw_head'] for row in steps]).mean(0)
assert k==2 and len(steps)==2
assert max(row['body_added_max'] for row in steps)==0.
assert max(row['head_added_error'] for row in steps)<5e-8
assert maximum(newci-rawmean)<5e-6
assert maximum(newci-(ci-c-delta[-h:]/(m.LR*k)))==0.
assert torch.equal(m.r.vectorize(model),anchor)
first_order=ds.order.copy();ds.order.clear()
repeat=m.train(model,ds,20261191,'cpu',ci,c)
assert ds.order==first_order and torch.equal(repeat[0],delta) and torch.equal(repeat[1],newci)
plain_ds=make_dataset(67,801);m.train(model,plain_ds,20261191,'cpu',None,None)
assert plain_ds.order==first_order
report['checks']['local_gradient']={'examples':67,'batch':64,'actual_steps':k,'body_gradient_added_max':max(row['body_added_max'] for row in steps),'head_gradient_added_error_max':max(row['head_added_error'] for row in steps),'ci_vs_raw_head_gradient_mean_error':maximum(newci-rawmean),'reported_telescoping_error':err,'same_seed_repeat_delta_bitwise':True,'same_seed_control_plain_batch_order':True,'order_sha256':hashlib.sha256(np.asarray(first_order,dtype='<i8').tobytes()).hexdigest(),'global_anchor_bitwise_unchanged':True}

packet_cases=[];normal_wire=None
for name,values in [('normal',torch.randn(136874,generator=torch.Generator().manual_seed(91))*.01),('zero',torch.zeros(136874)),('body_tail_impulse',torch.cat((torch.zeros(136223),torch.tensor([.125]),torch.zeros(650))))]:
    u,dh=values[:body],values[body:]
    q,residual,wire,diag=m.encode_head(u,dh,2)
    qb,qh,kread=m.decode_head(wire)
    assert len(wire)<=m.CAP and diag['budget_bytes']==52176
    assert torch.equal(q,torch.cat((qb,qh))) and torch.equal(qh,dh) and kread==2
    eferror=maximum(qb+residual-u)
    assert eferror<1e-7
    packet_cases.append({'name':name,'wrapper_bytes':m.HEADER.size,'head_bytes':4*h,'body_budget_bytes':diag['budget_bytes'],'actual_body_wire_bytes':diag['wire_bytes'],'actual_total_wire_bytes':len(wire),'cap_bytes':m.CAP,'unused_bytes':m.CAP-len(wire),'head_bitwise_exact':True,'decoder_equals_encoded_update':True,'body_ef_max_error':eferror})
    if name=='normal':normal_wire=wire
assert m.HEADER.size==32 and m.CAP-32-4*h==52176
negative={}
def rejected(name,wire):
    try:m.decode_head(wire)
    except (ValueError,struct.error):negative[name]=True
    else:raise AssertionError('wrapper accepted '+name)
rejected('short_header',normal_wire[:31]);rejected('truncated_packet',normal_wire[:-1]);rejected('trailing_byte',normal_wire+b'X')
for name,offset,value in [('wrong_version',8,2),('wrong_body_size',12,body-1),('wrong_head_size',16,h-1),('zero_steps',20,0),('wrong_length',24,1),('nonzero_reserved',28,1)]:
    wire=bytearray(normal_wire);struct.pack_into('<I',wire,offset,value);rejected(name,wire)
wire=bytearray(normal_wire);wire[0]=0;rejected('wrong_magic',wire)
wire=bytearray(normal_wire);struct.pack_into('<f',wire,len(wire)-4,float('inf'));rejected('nonfinite_head',wire)
wire=bytearray(normal_wire);wire[m.HEADER.size+m.codec.HEADER_SIZE]=7;rejected('bad_body_bitwidth',wire)
report['checks']['actual_wire']={'cases':packet_cases,'malformed_rejections':negative}

# General nonzero old-control identity at unequal fixed sample weights and K.
weights=torch.tensor([3/8,5/8]);gen=torch.Generator().manual_seed(812)
oldcis=[torch.randn(h,generator=gen)*.01 for _ in range(2)]
oldc=sum((float(w)*ci for w,ci in zip(weights,oldcis)),torch.zeros(h))
rawmeans=[torch.randn(h,generator=gen)*.03 for _ in range(2)];ks=[2,5]
newcis=[];direct=torch.zeros(h)
for weight,ci,rawmean,ki in zip(weights,oldcis,rawmeans,ks):
    dh=-m.LR*ki*(rawmean+oldc-ci)
    zeros=torch.zeros(body);_,_,wire,_=m.encode_head(zeros,dh,ki)
    _,received_head,received_k=m.decode_head(wire)
    newci=ci-oldc-received_head/(m.LR*received_k)
    newcis.append(newci);direct.add_(received_head/(-m.LR*received_k),alpha=float(weight))
weighted=sum((float(w)*ci for w,ci in zip(weights,newcis)),torch.zeros(h))
assert maximum(weighted-direct)<5e-6
assert max(maximum(ci-raw) for ci,raw in zip(newcis,rawmeans))<5e-6
report['checks']['fixed_weight_control_identity']={'weights':[3/8,5/8],'different_steps':ks,'old_c_equals_weighted_old_ci':True,'decoded_head_reconstruction':True,'weighted_new_ci_vs_server_direct_error':maximum(weighted-direct),'ci_vs_raw_mean_error_max':max(maximum(ci-raw) for ci,raw in zip(newcis,rawmeans))}

# Exercise the actual run(), wire path, aggregation and checkpoint together.
trainsets=[make_dataset(3,901),make_dataset(5,902)];dev=make_dataset(2,903)
args=SimpleNamespace(out=OUT/'one_round',seed=20261191,rounds=1,backend='cpu')
manifest={'fixture_only':True,'source':'generated TensorDataset','client_examples':[3,5],'dev_examples':2,'official_data_loaded':False}
original_train=m.train;captured={};current={}
def observed_train(model,ds,seed,backend,ci,c):
    before=m.r.vectorize(model).clone();ds.order.clear()
    result=original_train(model,ds,seed,backend,ci,c)
    assert torch.equal(before,m.r.vectorize(model))
    current['records'].append({'seed':seed,'order':ds.order.copy(),'anchor_sha256':digest_tensor(before),'delta_sha256':digest_tensor(result[0]),'delta_head':result[0][-h:].clone(),'steps':result[2]})
    return result
m.train=observed_train
try:
    for method in m.METHODS:
        current['records']=[]
        with contextlib.redirect_stdout(io.StringIO()):m.run(args,method,trainsets,dev,manifest)
        captured[method]=current['records']
finally:m.train=original_train
reference=captured['plain_uniform']
for method,rows in captured.items():
    assert [(row['seed'],row['order'],row['anchor_sha256'],row['delta_sha256']) for row in rows]==[(row['seed'],row['order'],row['anchor_sha256'],row['delta_sha256']) for row in reference]
method_rows=[]
for method in m.METHODS:
    folder=args.out/method;conf=json.loads((folder/'config.json').read_text());result=json.loads((folder/'result.json').read_text());ck=torch.load(folder/'checkpoint.pt',weights_only=False)
    signature=conf.pop('signature');assert hashlib.sha256(json.dumps(conf,sort_keys=True).encode()).hexdigest()==signature
    assert ck['signature']==signature==result['signature']
    assert conf['source_sha256']==source_before['work/head_control/run.py']
    assert conf['plan_sha256']==source_before['work/head_control/PLAN.md']
    assert conf['codec_sha256']==source_before['work/mixed_precision/codec.py']
    assert conf['weights']==[3/8,5/8] and conf['official_test_loaded'] is False
    row=result['rows'][0];assert len(row['clients'])==2 and row['upload_bytes']==sum(client['upload_bytes'] for client in row['clients'])
    entry={'method':method,'rounds':len(result['rows']),'actual_local_sgd_steps':sum(client['steps'] for client in row['clients']),'upload_per_client':[client['upload_bytes'] for client in row['clients']],'signature':signature,'source_plan_codec_hashes_bound':True,'global_anchor_unmodified_during_clients':True}
    if method.startswith('head_control'):
        weighted=sum((weight*ci for weight,ci in zip(conf['weights'],ck['cis'])),torch.zeros(h))
        direct=sum((weight*(-client['delta_head']/(m.LR*client['steps'])) for weight,client in zip(conf['weights'],captured[method])),torch.zeros(h))
        assert torch.equal(weighted,ck['c']) and maximum(weighted-direct)<5e-6
        assert row['download_bytes_analytical']==2*(64+4*136874+16+4*650)
        entry['stored_c_matches_weighted_ci']=True;entry['server_direct_reconstruction_error']=maximum(weighted-direct)
    if method=='head_control_dense':assert conf['upload_cap_bytes'] is None and all(value>m.CAP for value in entry['upload_per_client'])
    else:assert all(value<=m.CAP for value in entry['upload_per_client'])
    method_rows.append(entry)
# First-round control is zero, so learning deltas must be identical across all
# four groups; this checks matched initialization and batch RNG, not efficacy.
checkpoint_folder=args.out/'head_control';checkpoint=checkpoint_folder/'checkpoint.pt'
config=checkpoint_folder/'config.json';config_before=config.read_bytes();checkpoint_before=checkpoint.read_bytes()
train_calls=[]
def forbid_train(*args,**kwargs):train_calls.append(True);raise AssertionError('unexpected training during signature/resume fixture')
m.train=forbid_train
checkpoint_tests={}
try:
    with contextlib.redirect_stdout(io.StringIO()):m.run(args,'head_control',trainsets,dev,manifest)
    checkpoint_tests['completed_resume_no_training']=True
    for name,changed in [('seed',SimpleNamespace(**{**vars(args),'seed':args.seed+1})),('backend',SimpleNamespace(**{**vars(args),'backend':'mps'}))]:
        try:m.run(changed,'head_control',trainsets,dev,manifest)
        except ValueError as error:assert 'incompatible source/config output' in str(error);checkpoint_tests[name+'_mismatch_rejected_before_training']=True
        else:raise AssertionError(name+' mismatch accepted')
    original_sha=m.sha
    m.sha=lambda path:'0'*64 if Path(path).resolve()==(HERE/'run.py').resolve() else original_sha(path)
    try:
        try:m.run(args,'head_control',trainsets,dev,manifest)
        except ValueError as error:assert 'incompatible source/config output' in str(error);checkpoint_tests['source_identity_mismatch_rejected_before_training']=True
        else:raise AssertionError('source identity mismatch accepted')
    finally:m.sha=original_sha
    bad=torch.load(checkpoint,weights_only=False);bad['signature']='invalid-fixture-signature';torch.save(bad,checkpoint)
    try:m.run(args,'head_control',trainsets,dev,manifest)
    except ValueError as error:assert 'wrong checkpoint signature' in str(error);checkpoint_tests['checkpoint_signature_mismatch_rejected_before_training']=True
    else:raise AssertionError('checkpoint signature mismatch accepted')
finally:
    m.train=original_train;config.write_bytes(config_before);checkpoint.write_bytes(checkpoint_before)
assert not train_calls
report['checks']['one_round_integration']={'synthetic_client_examples':[3,5],'synthetic_dev_examples':2,'backend':'cpu','threads':1,'methods':method_rows,'matched_seed_batch_anchor_and_initial_raw_deltas':True,'seeds':[row['seed'] for row in reference],'example_orders':[row['order'] for row in reference],'checkpoint_checks':checkpoint_tests}
assert not report['dataset_open_attempts']
source_after={name:sha(ROOT/name) for name in source_before};assert source_after==source_before
report['source_identities_unchanged']=True
report['elapsed_seconds']=time.perf_counter()-started
report['status']='PASS';report['failures']=[]
report['harness_source']={'path':str(Path(__file__).resolve()),'sha256':sha(__file__)}
(HERE/'LOCAL_FIXTURES.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'status':report['status'],'fixture_out':str(OUT),'elapsed_seconds':report['elapsed_seconds'],'checks':list(report['checks']),'official_dataset_open_attempts':report['dataset_open_attempts'],'json':str(HERE/'LOCAL_FIXTURES.json')},indent=2))
