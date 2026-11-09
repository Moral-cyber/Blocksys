"""Finite read-only index/result check. Never launches training or loads TEST."""
from pathlib import Path
import hashlib
import json
import math
import struct
import numpy as np
import torch
from torch.utils.data import DataLoader,TensorDataset

torch.set_num_threads(1)
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent/'method_followup'
report={'status':'PENDING','conditions':{},'cross_mixing':'PENDING','external_cross_model_audit':'REVIEW_UNAVAILABLE'}
manifests={};configs={}
def grouped_hash(groups):
 d=hashlib.sha256()
 for ids in groups:d.update(struct.pack('<I',len(ids)));d.update(np.array(ids,dtype='<u4').tobytes())
 return d.hexdigest()
def find(mixing,variant):
 dirs=[p for p in (HERE/f'mixing{mixing}').glob('*') if p.is_dir() and p.name.startswith(variant+'_')]
 assert len(dirs)<=1
 return dirs[0] if dirs else None
for mixing in [0,1]:
 for variant in ['dense','topk_ef']:
  name=f'mixing{mixing}/{variant}';folder=find(mixing,variant)
  if folder is None or not (folder/'config.json').exists():
   report['conditions'][name]={'status':'NOT_STARTED'};continue
  cfg=json.loads((folder/'config.json').read_text());configs[name]=cfg
  assert cfg['mixing']==mixing and cfg['variant']==variant and cfg['seed']==20261171
  assert cfg['split_seed']==20261143 and cfg['rounds']==20 and cfg['fraction']==.05
  assert cfg['lr']==.03 and cfg['local_epochs']==1 and cfg['validation_size']==2000
  assert cfg['training_backend']=='mps' and cfg['fedprox_mu']==0 and cfg['test_final'] is False
  identity=cfg.copy();sig=identity.pop('run_signature');identity.pop('out')
  assert hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()==sig
  assert cfg['runner_source_sha256']==hashlib.sha256((ROOT/'run_followup.py').read_bytes()).hexdigest()
  assert all(hashlib.sha256((ROOT/file).read_bytes()).hexdigest()==sha for file,sha in cfg['accelerated_source_sha256'].items())
  m=cfg['data_manifest'];tr=m['train_indices'];ho=m['holdout_indices'];va=m['validation_indices']
  if mixing in manifests:assert m==manifests[mixing]
  manifests[mixing]=m
  train=np.concatenate(tr);holdout=np.concatenate(ho);val=np.asarray(va)
  assert len(train)==37500 and len(holdout)==12500 and len(val)==2000 and len(np.unique(val))==2000
  assert not np.intersect1d(train,holdout).size and np.array_equal(np.sort(np.r_[train,holdout]),np.arange(50000))
  assert np.isin(val,holdout).all() and not np.isin(val,train).any()
  assert grouped_hash(tr)==m['train_indices_sha256'] and grouped_hash(ho)==m['holdout_indices_sha256']
  assert [len(x) for x in tr]==m['client_sizes']
  # No real samples/models: enumerate actual loader sizes; one epoch, drop_last=False.
  steps=[]
  for n in m['client_sizes']:
   loader=DataLoader(TensorDataset(torch.arange(n)),batch_size=64,shuffle=True,generator=torch.Generator().manual_seed(1))
   batches=[len(x[0]) for x in loader]
   assert len(batches)==math.ceil(n/64) and sum(batches)==n
   steps.append(len(batches))
  entry={'status':'CONFIG_ONLY','actual_loader_batches_one_epoch':steps,'samples_train':37500,'samples_dev':2000,
   'test_final':cfg['test_final'],'run_signature_verified':True}
  if (folder/'result.json').exists():
   result=json.loads((folder/'result.json').read_text());rows=result['rows']
   assert 'official_test_final' not in result
   assert result['seed']==20261171 and result['variant']==variant
   assert all(row['round']==i+1 for i,row in enumerate(rows))
   total=0;expect_round=20*(64+4*136874 if variant=='dense' else 64+8*6843)
   for row in rows:
    v=row['validation'];assert v['samples']==2000 and np.isfinite(v['loss']) and 0<=v['accuracy']<=1
    total+=row['round_bytes'];assert row['round_bytes']==expect_round and row['cumulative_bytes']==total
   assert len(result['client_rows'])==20*len(rows)
   entry.update({'status':result['status'],'rounds':len(rows),'analytical_upload_bytes':total})
   ck=torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=False)
   if len(ck['rows'])==len(rows):
    assert ck['total_bytes']==total and len(ck['residuals'])==20
    assert all(torch.isfinite(t).all() for t in ck['model'].values())
    assert all(torch.isfinite(t).all() for t in ck['residuals'])
    entry['current_checkpoint']='MATCH'
   else:entry['current_checkpoint']='ADVANCED_DURING_READ'
   if result['status']=='DONE' and len(rows)==20:
    entry['last5']={'rounds':[16,17,18,19,20],
      'accuracy':sum(row['validation']['accuracy'] for row in rows[-5:])/5,
      'ce':sum(row['validation']['loss'] for row in rows[-5:])/5}
  report['conditions'][name]=entry
if 0 in manifests:
 m0=manifests[0];original=[np.asarray(x,dtype=np.int64) for x in m0['train_indices']]
 rng=np.random.default_rng(np.random.SeedSequence([20261143,0x4D495831]))
 pool=np.concatenate([ids[rng.permutation(len(ids))] for ids in original]);pool=pool[rng.permutation(len(pool))]
 expected=[];at=0
 for ids in original:expected.append(pool[at:at+len(ids)]);at+=len(ids)
 report['expected_mixing1_from_current_mixing0']={'train_indices_sha256':grouped_hash(expected),
   'counts_preserved':True,'train_global_multiset_preserved':np.array_equal(np.sort(pool),np.sort(np.concatenate(original)))}
 if 1 in manifests:
  m1=manifests[1]
  assert m0['client_sizes']==m1['client_sizes']
  assert m0['holdout_indices']==m1['holdout_indices'] and m0['validation_indices']==m1['validation_indices']
  assert np.array_equal(np.sort(np.concatenate(m0['train_indices'])),np.sort(np.concatenate(m1['train_indices'])))
  assert grouped_hash(expected)==m1['train_indices_sha256']
  assert m0['train_indices_sha256']!=m1['train_indices_sha256']
  report['cross_mixing']={'status':'PASS','same_client_train_counts':True,'same_actual_loader_steps':True,
   'same_global_train_row_multiset':True,'same_holdout':True,'same_validation':True,
   'only_train_assignment_changed_in_data':True,'mixing1_matches_independent_permutation':True,
   'runtime_optimizer_step_counts_logged':False}
if all(report['conditions'][f'mixing{m}/{v}']['status']=='DONE' and report['conditions'][f'mixing{m}/{v}']['rounds']==20 for m in [0,1] for v in ['dense','topk_ef']):
 report['status']='ALL_DONE';penalties={}
 for mixing in [0,1]:
  dense=report['conditions'][f'mixing{mixing}/dense']['last5'];sparse=report['conditions'][f'mixing{mixing}/topk_ef']['last5']
  penalties[f'mixing{mixing}']={'ce_sparse_minus_dense':sparse['ce']-dense['ce'],
    'accuracy_dense_minus_sparse_pp':100*(dense['accuracy']-sparse['accuracy'])}
 report['compression_penalties']=penalties
 report['descriptive_interaction_mixing0_minus_mixing1']={k:penalties['mixing0'][k]-penalties['mixing1'][k] for k in penalties['mixing0']}
 d0=report['conditions']['mixing0/dense']['last5'];d1=report['conditions']['mixing1/dense']['last5']
 report['dense_heterogeneity_gap_mixing0_minus_mixing1']={'ce':d0['ce']-d1['ce'],'accuracy_pp':100*(d0['accuracy']-d1['accuracy'])}
(HERE/'LOCAL_STATS_CHECK.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
# Keep the short local report in sync when root reruns this checker after completion.
lines=[]
for name,entry in report['conditions'].items():
 if 'last5' in entry:
  z=entry['last5'];lines.append(f"| {name} | {entry['status']} | {entry['rounds']} | {z['ce']:.10f} | {100*z['accuracy']:.6f}% |")
 else:lines.append(f"| {name} | {entry['status']} | {entry.get('rounds','—')} | pending | pending |")
text='''# Baseline reanchor：本地有限核对\n\n2026-11-04。仅核对冻结launcher、当前配置/索引和保存统计；没有启动训练、修改优化器或设计新算法。外部跨模型审计 **`REVIEW_UNAVAILABLE`**。\n\nlauncher调用原run_accelerated/run_partitions，固定partition20261143、seed20261171、MPS、20轮、全参与、SGD.03/localepoch1/batch64；仅改变mixing0/1与dense/topk_ef。所有已生成config的冻结源哈希及run_signature匹配，test_final=false，当前result没有official_test_final字段。\n\n索引核对：已生成条件均TRAIN37500、holdout池12500、固定global dev2000，train/holdout互斥且覆盖全部50000原始行。各客户端一epoch真实DataLoader(drop_last=False)的批次数枚举为[30,46,4,93,39,8,2,5,27,51,9,62,66,5,44,22,16,3,40,25]。runner每batch一次optimizer.step，因此同client count意味着相同步数；当前runner没有单独记录实际optimizer计数，不把该推导冒称运行时计数日志。\n\n'''
if isinstance(report['cross_mixing'],dict):
 text+='''跨mixing索引检查 **PASS**：两边训练原始行multiset及每client count相同，holdout/validation完整索引逐项相同，训练分配确实改变。mixing1实际索引与从mixing0独立执行确定性全量混池所得结果相符；因此同数据规模/步数、同全局经验目标，仅训练客户端归属改变。mixing1不保证每客户端有限样本标签严格均匀。\n\n'''
else:
 text+='''跨mixing核对 **PENDING**：mixing1尚未生成config，不能先宣称实际两边已匹配。仅已独立构造预期mixing1置换（hash=b0968c8ae420f9debf7707e387dd7af87d89172eff7e0c78e5e67e8d1339f002），保持原train multiset与client count；待实际config落盘后再比对。\n\n'''
text+='| 条件 | 状态 | 完成轮数 | last5 dev CE | last5 dev accuracy |\n|---|---|---:|---:|---:|\n'+'\n'.join(lines)+'\n\n'
if report['status']=='ALL_DONE':
 text+='''四组均DONE20；last5严格取round16–20，从保存validation行求均值，未重算模型预测或重现MPS轨迹。2000是global dev，不能称client宏平均。\n\n'''
 for mix,z in report['compression_penalties'].items():
  text+=f"- {mix}压缩CE代价(sparse−dense)={z['ce_sparse_minus_dense']:+.10f}；accuracy代价(dense−sparse)={z['accuracy_dense_minus_sparse_pp']:+.6f}pp。\n"
 z=report['descriptive_interaction_mixing0_minus_mixing1']
 text+=f"\n压缩代价交互(mixing0−mixing1)：CE={z['ce_sparse_minus_dense']:+.10f}；accuracy代价={z['accuracy_dense_minus_sparse_pp']:+.6f}pp。\n"
 z=report['dense_heterogeneity_gap_mixing0_minus_mixing1']
 text+=f"\nDense本身mixing0−mixing1：CE={z['ce']:+.10f}，accuracy={z['accuracy_pp']:+.6f}pp。以上仅当前模型/分区/seed/20轮的描述差，不作显著性、总体因果或普遍领域结论。负压缩代价不叫受损。\n"
else:
 text+='''最终last5与压缩penalty交互 **PENDING**，不使用未完成轨迹估计交互。root可在四组完成后运行verify_results.py；它会复算两mix索引、last5、CE/accuracy压缩代价及mixing0−mixing1交互，并同步本报告。不需要重新启动训练。\n'''
text+='''\n预算按已有codec解析公式核算，仅上传应用payload，不称真实wire或总网络通信。检查器和证据：verify_results.py / LOCAL_STATS_CHECK.json。此前所有STOP候选保持STOP；本检查不会复活旧候选。\n'''
(HERE/'LOCAL_CHECK.md').write_text(text)
