"""Isolated CIFAR-100 TRAIN/development adapter; never downloads or reads TEST."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import copy
import hashlib
import importlib.util
import json
import pickle
import sys
import time
import tracemalloc
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
PAPER = Path('/Users/wind/Desktop/Codex/paper')
RAW = PAPER/'02_实验工程/dataset/Cifar100/rawdata'
CLASSES,PARAMETERS,CAP,PARTITION = 100,142724,56376,20261107
SEEDS = (20261231,20261232,20261233)
TRAIN_MD5 = '16019d7e3df5f24257cddd939b257f8d'


def load(name,path):
    spec = importlib.util.spec_from_file_location(name,path)
    obj = importlib.util.module_from_spec(spec)
    sys.modules[name] = obj
    spec.loader.exec_module(obj)
    return obj


base = load('_c100_baseline_dependencies',HERE.parent/'baselines/adapter.py')
solver = load('_c100_structured_solver',HERE/'solver.py')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def signature(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()


def verify(mapping):
    for path,digest in mapping.items():
        if sha(path) != digest:
            raise ValueError('CIFAR100 locked source/input changed: '+path)


def guard(raw):
    attempts = []
    test = (Path(raw)/'cifar-100-python/test').resolve()
    def audit(event,args):
        if event == 'open' and isinstance(args[0],(str,bytes)):
            value = args[0].decode(errors='replace') if isinstance(args[0],bytes) else args[0]
            path = Path(value)
            if ('test_batch' in value or 't10k' in value or 'predictive_final_test' in value or
                    path.resolve()==test or (path.name=='test' and path.parent.name=='cifar-100-python')):
                attempts.append(value)
                raise ValueError('CIFAR100 extension cannot read official TEST')
    sys.addaudithook(audit)
    return attempts


def dependencies():
    c,_ = base.dependencies('cifar')
    partitions = load('_c100_isolated_partition_namespace',Path(c.partitions.__file__))
    partitions.NUM_CLASSES = CLASSES
    if c.codec.wire_size(PARAMETERS,3) != CAP:
        raise ValueError('CIFAR100 wire-size accounting mismatch')
    return c,partitions


def sources(c):
    paths = [HERE/name for name in ('PLAN.md','adapter.py','solver.py','fixture_check.py','run_matrix.py')]
    paths += [HERE.parent/'baselines/adapter.py']
    return {**c.source_identities(),**{str(p):sha(p) for p in paths}}


def check_membership(manifest,labels):
    train = np.concatenate(manifest['train_indices'])
    holdout = np.concatenate(manifest['holdout_indices'])
    if (len(manifest['train_indices'])!=20 or np.intersect1d(train,holdout).size or
            not np.array_equal(np.sort(np.concatenate((train,holdout))),np.arange(50000))):
        raise ValueError('CIFAR100 client coverage/disjointness failure')
    if manifest['validation_class_counts'] != [20]*CLASSES:
        raise ValueError('CIFAR100 monitor not exactly 20 examples per class')
    if not np.array_equal(np.bincount(labels,minlength=CLASSES),np.full(CLASSES,500)):
        raise ValueError('CIFAR100 fine-label counts mismatch')


def prepare(partitions,raw):
    path = Path(raw)/'cifar-100-python/train'
    content = path.read_bytes()
    if hashlib.md5(content).hexdigest() != TRAIN_MD5:
        raise ValueError('official CIFAR100 TRAIN integrity mismatch')
    value = pickle.loads(content,encoding='latin1')
    images = np.asarray(value['data'],dtype=np.uint8).reshape(50000,3,32,32).transpose(0,2,3,1)
    labels = np.asarray(value['fine_labels'],dtype=np.int64)
    if labels.shape != (50000,) or images.shape != (50000,32,32,3):
        raise ValueError('official CIFAR100 TRAIN shapes mismatch')
    tr,ho,val,manifest = partitions.partition_indices(labels,alpha=.1,split_seed=PARTITION,mixing=0.,validation_size=2000)
    check_membership(manifest,labels)
    manifest.update({'source':'CIFAR100 official TRAIN fine labels',
        'index_namespace':'zero-based original row in official CIFAR100 TRAIN fine_labels',
        'normalization':'(FP32(uint8_CHW)/255 - 0.5)/0.5',
        'official_train_file':str(path.resolve()),'official_train_md5':TRAIN_MD5,
        'official_train_sha256':hashlib.sha256(content).hexdigest(),
        'official_train_labels_sha256':hashlib.sha256(labels.astype('<i8').tobytes()).hexdigest(),
        'official_test_policy':'all CIFAR100 test, CIFAR10 test_batch and t10k byte reads rejected',
        'dataset_specific_classes':CLASSES})
    sets = [partitions.IndexedCifar10(images,labels,idx) for idx in tr]
    monitor = partitions.IndexedCifar10(images,labels,val)
    return sets,monitor,manifest,{str(path.resolve()):sha(path)},images,labels


def source_preflight():
    c,partitions = dependencies()
    fixtures = json.loads((HERE/'FIXTURES.json').read_text())
    if fixtures['status'] != 'PASS_NUMERICAL_FIXTURES':
        raise ValueError('solver fixtures missing')
    verify(fixtures['source_sha256'])
    labels = np.repeat(np.arange(CLASSES),500)
    _,_,_,manifest = partitions.partition_indices(labels,alpha=.1,split_seed=PARTITION,mixing=0.,validation_size=2000)
    check_membership(manifest,labels)
    record = {'status':'SOURCE_AND_SYNTHETIC_PARTITION_PREFLIGHT_PASS',
              'created_at':datetime.now(timezone.utc).isoformat(),'classes':CLASSES,
              'parameters_source_derived':PARAMETERS,'predictor_fp32_bytes':800,
              'uniform3_wire_bytes':CAP,'dense_wire_bytes':64+4*PARAMETERS,
              'ef_residual_bytes':4*PARAMETERS,'partition':PARTITION,'model_seeds':SEEDS,
              'synthetic_membership_checked':True,'actual_data_ready':False,
              'actual_data_status':'Requires root-downloaded official TRAIN and separate data preflight',
              'new_cnn_constructions':0,'new_cnn_training':0,'official_test_reads':0}
    write(HERE/'SOURCE_PREFLIGHT.json',record)
    write(HERE/'SOURCE_LOCK.json',{'status':'FROZEN_BEFORE_CIFAR100_TRAINING',
                                 'created_at':record['created_at'],'inputs':sources(c)})
    print(json.dumps(record),flush=True)


def data_preflight(raw):
    attempts = guard(raw)
    c,partitions = dependencies()
    lock = json.loads((HERE/'SOURCE_LOCK.json').read_text())
    verify(lock['inputs'])
    sets,monitor,manifest,data_ids,_,_ = prepare(partitions,raw)
    if attempts:
        raise ValueError('forbidden TEST access attempt')
    verify({**lock['inputs'],**data_ids})
    write(HERE/'DATA_PREFLIGHT.json',{'status':'PASS_DATA_READY','raw':str(Path(raw).resolve()),
        'data_identities':data_ids,'manifest_sha256':signature(manifest),'data_manifest':manifest,
        'clients':len(sets),'monitor_rows':len(monitor),'official_test_reads':0,
        'new_cnn_constructions':0,'new_cnn_training':0,'forbidden_open_attempts':attempts})
    print(json.dumps({'status':'PASS_DATA_READY','train_samples':manifest['train_samples'],
                      'holdout_samples':manifest['holdout_samples'],'clients':len(sets)}),flush=True)


def local_train(c,model,ds,seed):
    torch.manual_seed(seed)
    torch.mps.manual_seed(seed)
    local = copy.deepcopy(model).to('mps').train()
    anchor = c.runner.vectorize(model)
    counts = np.bincount(ds.labels[ds.indices],minlength=CLASSES)
    shift = torch.tensor(-np.maximum(counts,1)**(-.25),dtype=torch.float32,device='mps')
    loader = DataLoader(ds,batch_size=64,shuffle=True,generator=torch.Generator().manual_seed(seed))
    optimizer = torch.optim.SGD(local.parameters(),lr=.03)
    for x,y in loader:
        optimizer.zero_grad(set_to_none=True)
        nn.functional.cross_entropy(local(x.to('mps'))+shift,y.to('mps')).backward()
        optimizer.step()
    delta = torch.cat([p.detach().flatten() for p in local.parameters()]).cpu()-anchor
    if not torch.isfinite(delta).all() or not torch.equal(c.runner.vectorize(model),anchor):
        raise ValueError('CIFAR100 local update nonfinite or mutated global model')
    return delta,{'steps':len(loader),'gradient_evaluations':len(loader),'train_counts':counts.tolist()}


def forward(model,ds):
    logits,labels = [],[]
    with torch.inference_mode():
        for x,y in DataLoader(ds,batch_size=256,shuffle=False):
            logits.append(model(x).numpy().copy())
            labels.append(y.numpy().copy())
    return np.concatenate(logits),np.concatenate(labels)


def monitor_metrics(model,ds):
    """Same FP32 CE monitor as original, with explicitly K-class counters."""
    model.eval()
    hits,counts = torch.zeros(CLASSES,dtype=torch.long),torch.zeros(CLASSES,dtype=torch.long)
    loss_sum = 0.
    with torch.inference_mode():
        for x,y in DataLoader(ds,batch_size=256,shuffle=False):
            logits = model(x)
            loss_sum += float(nn.functional.cross_entropy(logits,y,reduction='sum'))
            pred = logits.argmax(1)
            counts += torch.bincount(y,minlength=CLASSES)
            hits += torch.bincount(y[pred==y],minlength=CLASSES)
    class_accuracy = hits.double()/counts.clamp_min(1)
    return {'accuracy':float(hits.sum()/counts.sum()),'loss':loss_sum/len(ds),
            'class_accuracy':class_accuracy.tolist(),'macro_accuracy':float(class_accuracy.mean()),
            'worst_class_accuracy':float(class_accuracy.min()),'samples':len(ds)}


def execute(seed,raw):
    if seed not in SEEDS:
        raise ValueError('only three predeclared CIFAR seeds allowed')
    attempts = guard(raw)
    torch.set_num_threads(1)
    if not torch.backends.mps.is_available():
        raise RuntimeError('MPS unavailable; no fallback')
    c,partitions = dependencies()
    lock = json.loads((HERE/'SOURCE_LOCK.json').read_text())
    preflight = json.loads((HERE/'DATA_PREFLIGHT.json').read_text())
    if preflight['status'] != 'PASS_DATA_READY' or preflight['raw'] != str(Path(raw).resolve()):
        raise ValueError('matching data preflight required')
    verify(lock['inputs'])
    sets,monitor,manifest,data_ids,images,labels = prepare(partitions,raw)
    if data_ids != preflight['data_identities'] or signature(manifest) != preflight['manifest_sha256']:
        raise ValueError('data/partition preflight identity mismatch')
    identities = {**lock['inputs'],**data_ids}
    weights = np.asarray([len(ds) for ds in sets],dtype=np.float64)
    weights /= weights.sum()
    config = {'version':1,'task':'cifar100','method':'fedlc_uniform3','classes':CLASSES,
              'parameters':PARAMETERS,'seed':seed,'partition_seed':PARTITION,'rounds':20,
              'backend':'mps','batch':64,'lr':.03,'local_epochs':1,'weights':weights.tolist(),
              'data_manifest':manifest,'source_and_data_identities':identities,
              'loss':'CE(z-max(n,1)^(-1/4)); same explicit FedLC-inspired tau1 variant',
              'selection':'fixed final round20; no holdout or TEST choice',
              'upload_cap_bytes':CAP,'dense_wire_bytes':64+4*PARAMETERS,
              'ef_residual_bytes':4*PARAMETERS,'predictor_fp32_bytes':800,
              'posthoc_lambda':10,'torch_version':str(torch.__version__)}
    config['signature'] = signature(config)
    out = HERE/'runs'/f's{seed}'
    out.mkdir(parents=True,exist_ok=True)
    if (out/'config.json').exists() and json.loads((out/'config.json').read_text()) != config:
        raise ValueError('CIFAR100 run full config mismatch')
    torch.manual_seed(seed)
    model = c.runner.SmallCNN(classes=CLASSES)
    anchor = c.runner.vectorize(model)
    if anchor.shape != (PARAMETERS,):
        raise ValueError('CIFAR100 SmallCNN parameter layout mismatch')
    residuals,rows = [torch.zeros_like(anchor) for _ in sets],[]
    if (out/'checkpoint.pt').exists():
        checkpoint = torch.load(out/'checkpoint.pt',map_location='cpu',weights_only=False)
        if checkpoint['config'] != config or checkpoint['signature'] != config['signature']:
            raise ValueError('CIFAR100 checkpoint config mismatch')
        model.load_state_dict(checkpoint['model'])
        residuals,rows = checkpoint['residuals'],checkpoint['rows']
        if (len(residuals)!=20 or any(r.shape!=(PARAMETERS,) or r.dtype!=torch.float32 or
                not torch.isfinite(r).all() for r in residuals) or
                [r['round'] for r in rows] != list(range(1,len(rows)+1)) or len(rows)>20):
            raise ValueError('CIFAR100 checkpoint residual/history mismatch')
    write(out/'config.json',config)
    for rnd in range(len(rows),20):
        began = time.perf_counter()
        anchor = c.runner.vectorize(model).clone()
        aggregate,clients = torch.zeros_like(anchor),[]
        for client,ds in enumerate(sets):
            local_seed = seed+100*rnd+client
            delta,counters = local_train(c,model,ds,local_seed)
            candidate = delta+residuals[client]
            encoded = c.codec.encode(candidate,CAP,mode='uniform3')
            q = c.codec.decode(encoded.wire)
            if not torch.equal(q,encoded.sent) or len(encoded.wire)!=CAP:
                raise ValueError('CIFAR100 uniform3 wire mismatch')
            residuals[client] = candidate-q
            aggregate.add_(q,alpha=float(weights[client]))
            if rnd==0 and client==0:
                (out/'first_payload.bin').write_bytes(encoded.wire)
            clients.append({'client':client,'seed':local_seed,**counters,'upload_bytes':len(encoded.wire),
                            'ef_max_error':float((q+residuals[client]-candidate).abs().max())})
        c.runner.assign_vector(model,anchor+aggregate)
        metrics = monitor_metrics(model,monitor)
        rows.append({'round':rnd+1,'validation':metrics,'clients':clients,
                     'upload_bytes':sum(x['upload_bytes'] for x in clients),
                     'global_anchor_unchanged_between_clients':True,'seconds':time.perf_counter()-began})
        temporary = out/'checkpoint.tmp'
        torch.save({'version':1,'config':config,'signature':config['signature'],'model':model.state_dict(),
                    'residuals':residuals,'rows':rows},temporary)
        temporary.replace(out/'checkpoint.pt')
        write(out/'training_result.json',{'status':'DONE' if len(rows)==20 else 'RUNNING',
                                        'rows':rows,'official_test_reads':0})
        print(json.dumps({'task':'cifar100','seed':seed,'round':rnd+1,'validation':metrics,
                          'seconds':rows[-1]['seconds']}),flush=True)
    verify(identities)
    if attempts:
        raise ValueError('TEST access attempted')
    checkpoint_hash = sha(out/'checkpoint.pt')
    if (out/'evaluation.json').exists():
        prior = json.loads((out/'evaluation.json').read_text())
        if prior['status']!='DONE' or prior['checkpoint_sha256']!=checkpoint_hash:
            raise ValueError('existing CIFAR100 evaluation mismatch')
        print(json.dumps({'status':'ALREADY_EVALUATED','task':'cifar100','seed':seed}),flush=True)
        return
    model.eval()
    clients = []
    for i,ds in enumerate(sets):
        z,y = forward(model,ds)
        anchor = solver.prior_anchor(y,CLASSES)
        states,infos = {},{}
        for variant,center in (('zero',np.zeros(CLASSES)),('prior',anchor)):
            tracemalloc.start()
            states[variant],infos[variant] = solver.fit(z,y,center)
            _,peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            infos[variant]['python_traced_peak_entire_fit_bytes'] = peak
        holdout = partitions.IndexedCifar10(images,labels,manifest['holdout_indices'][i])
        hz,hy = forward(model,holdout)
        fixed = np.concatenate((np.ones(CLASSES),anchor)).astype(np.float32)
        scored = {'raw':hz.astype(np.float64),'fixed_prior':solver.transform(hz,fixed),
                  **{v:solver.transform(hz,s) for v,s in states.items()}}
        metrics = {v:solver.stats(value,hy) for v,value in scored.items()}
        np.savez_compressed(out/f'client{i}.npz',zero=states['zero'],prior=states['prior'],fixed_prior=fixed,
                            train_ids=np.asarray(ds.indices,dtype=np.uint32),
                            holdout_ids=np.asarray(holdout.indices,dtype=np.uint32),
                            train_logits=z,train_labels=y,holdout_logits=hz,holdout_labels=hy)
        clients.append({'client':i,'train_samples':len(ds),'holdout_samples':len(holdout),
                        'train_counts':np.bincount(y,minlength=CLASSES).tolist(),'metrics':metrics,'fits':infos})
    summaries = solver.summarize(clients)
    verify(identities)
    if checkpoint_hash!=sha(out/'checkpoint.pt') or attempts:
        raise ValueError('CIFAR100 model mutation or TEST read')
    write(out/'evaluation.json',{'status':'DONE','task':'cifar100','seed':seed,'partition':PARTITION,
        'scope':'official TRAIN-fit/development-holdout only, no TEST','clients':clients,'summaries':summaries,
        'prior_minus_zero_macro_nll':summaries['prior']['macro_loss']-summaries['zero']['macro_loss'],
        'checkpoint_sha256':checkpoint_hash,'source_and_data_identities':identities,'identities_unchanged':True,
        'official_test_reads':0,'forbidden_open_attempts':attempts,'predictor_bytes':800,'shared_upload_bytes':CAP})
    print(json.dumps({'status':'RUN_DONE','task':'cifar100','seed':seed,'summaries':summaries}),flush=True)


def summarize():
    values = [json.loads((HERE/'runs'/f's{s}'/'evaluation.json').read_text()) for s in SEEDS]
    if any(v['status']!='DONE' or v['official_test_reads'] or not v['identities_unchanged'] for v in values):
        raise ValueError('all three complete development results required')
    variants = ('raw','fixed_prior','zero','prior')
    metrics = {v:{k:{'mean':float(np.mean([x['summaries'][v][k] for x in values])),
                    'sample_sd':float(np.std([x['summaries'][v][k] for x in values],ddof=1))}
                  for k in ('macro_loss','macro_accuracy','weighted_loss','weighted_accuracy','client_accuracy_q10',
                            'macro_brier','weighted_brier','macro_ece15','weighted_ece15','pooled_ece15')}
               for v in variants}
    deltas = [v['prior_minus_zero_macro_nll'] for v in values]
    write(HERE/'result.json',{'status':'DONE','task':'cifar100','scope':'three-seed development only',
        'partition':PARTITION,'seeds':SEEDS,'metrics':metrics,'individual_seed_nll_deltas':deltas,
        'all_three_seed_nll_improve':all(v<0 for v in deltas),'client_seed_nll_improved':sum(
            c['metrics']['prior']['loss']<c['metrics']['zero']['loss'] for v in values for c in v['clients']),
        'official_test_reads':0,'predictor_fp32_bytes':800,'shared_upload_bytes':CAP})
    print(json.dumps({'status':'ALL_DONE','task':'cifar100','seed_nll_deltas':deltas,'metrics':metrics}),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',required=True,choices=('source-preflight','data-preflight','run','summarize'))
    parser.add_argument('--seed',type=int)
    parser.add_argument('--raw',type=Path,default=RAW)
    args = parser.parse_args()
    if args.stage=='source-preflight': source_preflight()
    elif args.stage=='data-preflight': data_preflight(args.raw)
    elif args.stage=='summarize': summarize()
    elif args.seed is not None: execute(args.seed,args.raw)
    else: parser.error('run requires --seed')


if __name__=='__main__':
    main()
