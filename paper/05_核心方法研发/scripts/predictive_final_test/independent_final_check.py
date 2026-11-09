"""Independent exported-array final checks; never opens official TEST sources."""
from pathlib import Path
import argparse
import hashlib
import json
import struct
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
VARIANTS = ('raw','prior_only','analytic_prior','prioranchor_bias','zero_anchor_bias',
            'temperature_scaling','bcts','zero_anchor_vector','prioranchored_vector',
            'learned_head','learned_head_plus_prior','prioranchored_bcts')
METHODS = ('ce_uniform3','fedlc_uniform3','fedlc_dense')
DIMENSIONS = {'analytic_prior':10,'prioranchor_bias':10,'zero_anchor_bias':10,
              'temperature_scaling':1,'bcts':11,'zero_anchor_vector':20,
              'prioranchored_vector':20,'prioranchored_bcts':11}
SEEDS = {'cifar':[20261231,20261232,20261233],'fashion':[20261241,20261242,20261243]}
PARTITIONS = {'cifar':20261217,'fashion':20261219}


def forbidden(path):
    return 'test_batch' in str(path) or 't10k' in str(path)


def sha(path):
    if forbidden(path):
        raise AssertionError('independent check must not read official TEST file')
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_sha(a, dtype):
    return hashlib.sha256(np.ascontiguousarray(a,dtype=dtype).tobytes()).hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def independent_metrics(z,y):
    z=np.asarray(z,dtype=np.float64);y=np.asarray(y)
    assert z.shape==(len(y),10) and y.dtype.kind in 'iu' and np.isfinite(z).all()
    maxima=z.max(axis=1)
    ex=np.exp(z-maxima[:,None]);p=ex/np.sum(ex,axis=1)[:,None]
    losses=np.log(np.sum(ex,axis=1))+maxima-z[np.arange(len(y)),y]
    correct=np.argmax(z,axis=1)==y;confidence=np.max(p,axis=1)
    brier=np.mean(np.sum((p-np.eye(10)[y])**2,axis=1))
    counts=[];conf=[];hits=[]
    for k in range(15):
        mask=(confidence>=k/15)&(confidence<(k+1)/15 if k<14 else confidence<=1)
        counts.append(int(mask.sum()));conf.append(float(np.sum(confidence[mask])))
        hits.append(int(np.sum(correct[mask])))
    return {'accuracy':float(np.mean(correct)),'nll':float(np.mean(losses)),
            'loss':float(np.mean(losses)),'brier':float(brier),
            'ece15':float(sum(abs(a-b) for a,b in zip(hits,conf))/len(y)),
            'samples':len(y),'ece_bins':{'counts':counts,'confidence_sum':conf,'correct_sum':hits}}


def transforms(raw,head,s):
    z=raw.astype(np.float64);h=head.astype(np.float64)
    state={k:v.astype(np.float64) for k,v in s.items()}
    result={'raw':z,'prior_only':np.tile(state['analytic_prior'],(len(z),1)),
            'analytic_prior':z+state['analytic_prior'],'prioranchor_bias':z+state['prioranchor_bias'],
            'zero_anchor_bias':z+state['zero_anchor_bias'],
            'temperature_scaling':z*state['temperature_scaling'][0],
            'bcts':z*state['bcts'][0]+state['bcts'][1:],
            'learned_head':h,'learned_head_plus_prior':h+state['analytic_prior'],
            'prioranchored_bcts':z*state['prioranchored_bcts'][0]+state['prioranchored_bcts'][1:]}
    for name in ('zero_anchor_vector','prioranchored_vector'):
        result[name]=z*state[name][:10]+state[name][10:]
    return result


def check(root,out):
    attempts=[]
    def audit(event,args):
        if event=='open' and isinstance(args[0],(str,bytes)) and forbidden(args[0]):
            attempts.append(str(args[0]));raise AssertionError('official TEST raw read forbidden')
    sys.addaudithook(audit)
    root,out=Path(root).resolve(),Path(out).resolve()
    assignment=json.loads((root/'assignment.json').read_text());task=assignment['task']
    counts=np.asarray(assignment['train_counts'],dtype=np.int64)
    groups=[np.asarray(ids,dtype=np.uint32) for ids in assignment['client_ids']]
    assert task in SEEDS and assignment['partition_seed']==PARTITIONS[task]
    assert counts.shape==(20,10) and np.all(counts>=0)
    unsigned={k:v for k,v in assignment.items() if k!='manifest_hash'}
    assert canonical_sha(unsigned)==assignment['manifest_hash']
    assert array_sha(counts,'<i8')==assignment['train_counts_int64le_sha256']
    flat=np.concatenate(groups)
    assert len(flat)==10000 and np.array_equal(np.sort(flat),np.arange(10000))
    digest=hashlib.sha256()
    for ids in groups:digest.update(struct.pack('<I',len(ids)));digest.update(ids.astype('<u4').tobytes())
    assert digest.hexdigest()==assignment['assignment_hash']
    hashed={};input_evidence={};reports={};pending=[];labels_reference=None
    checks={'client_files':0,'state_arrays':0,'variant_score_arrays':0,'client_metric_values':0,
            'ece_bin_values':0,'summary_values':0,'three_seed_aggregate_values':0,'paired_seed_values':0}
    errors={'transform':0.,'metric':0.,'bins':0.,'summary':0.,'aggregate':0.}
    def close(value,expected,kind,tolerance=2e-12):
        error=float(np.max(np.abs(np.asarray(value,dtype=np.float64)-np.asarray(expected,dtype=np.float64))))
        errors[kind]=max(errors[kind],error)
        assert error<=tolerance,(kind,error,value,expected)
    def verify_hash(path,expected):
        if str(path) not in hashed:hashed[str(path)]=sha(path)
        assert hashed[str(path)]==expected,('hash mismatch',str(path))
    for seed in SEEDS[task]:
        for method in METHODS:
            folder=root/f'{method}_s{seed}';result_path=folder/'result.json'
            if not result_path.exists():
                pending.append(f'{method}_s{seed}');continue
            report=json.loads(result_path.read_text())
            assert report['status']=='DONE' and report['scope']=='LOCKED_OFFICIAL_FINAL_TEST'
            assert (report['task'],report['seed'],report['method'],report['partition_seed'])==(task,seed,method,PARTITIONS[task])
            assert tuple(report['fixed_variants'])==VARIANTS and report['assignment_hash']==assignment['assignment_hash']
            assert json.loads((folder/'assignment.json').read_text())==assignment
            auth=report['authorization'];lock=json.loads(Path(auth['root_lock']).read_text())
            verify_hash(auth['root_lock'],auth['root_lock_sha256'])
            verify_hash(HERE/'FINAL_TEST_PLAN.md',auth['plan_sha256'])
            assert lock['status']=='ROOT_LOCKED_FINAL_TEST' and task in lock['tasks']
            for name,value in auth['source_sha256'].items():verify_hash(HERE/name,value)
            snapshot=report['input_snapshot_sha256_before_after_equal']
            for filename,value in lock['frozen_input_snapshot'].items():assert snapshot[filename]==value
            raw_records=report['test_provenance']['files']
            assert report['test_provenance']['task']==task and report['test_provenance']['official_split']=='official TEST only'
            expected_md5={'test_batch':'40351d587109b95175f43aff81a1287e',
                          't10k-images-idx3-ubyte.gz':'bef4ecab320f06d8554ea6380940ec79',
                          't10k-labels-idx1-ubyte.gz':'bb300cfdad3c16e7a12a480ee83cd310'}
            assert len(raw_records)==(1 if task=='cifar' else 2)
            for record in raw_records:
                assert forbidden(record['path']) and record['md5']==record['expected_official_md5']==expected_md5[Path(record['path']).name]
                assert record['official_md5_verified'] is True and snapshot[record['path']]==record['sha256']
            for filename,value in snapshot.items():
                if not forbidden(filename):verify_hash(filename,value)
            source_state_paths={Path(p).name:Path(p) for p in snapshot
                                if f'train_states_{task}_p{PARTITIONS[task]}_s{seed}' in p}
            labels=np.full(10000,-1,dtype=np.int64);recomputed={name:[] for name in VARIANTS}
            score_groups={name:[] for name in VARIANTS}
            exports={row['path']:row for row in report['exports']}
            assert set(exports)=={f'client{c}.npz' for c in range(20)}
            assert [row['client'] for row in report['clients']]==list(range(20))
            for client,ids in enumerate(groups):
                target=folder/f'client{client}.npz';verify_hash(target,exports[target.name]['sha256'])
                with np.load(target,allow_pickle=False) as saved:arrays={k:saved[k].copy() for k in saved.files}
                assert np.array_equal(arrays['test_ids'],ids) and arrays['test_labels'].dtype.kind in 'iu'
                y=arrays['test_labels'];labels[ids]=y
                assert arrays['raw_logits'].dtype==arrays['fixed_head_logits'].dtype==np.float32
                states={name:arrays['state_'+name] for name in DIMENSIONS}
                state_path=source_state_paths[f'{method}_client{client}.npz']
                with np.load(state_path,allow_pickle=False) as original:
                    assert np.array_equal(original['train_counts'],counts[client])
                    for name,size in DIMENSIONS.items():
                        state=states[name]
                        assert state.dtype==np.float32 and state.shape==(size,)
                        assert np.array_equal(state,original['state_'+name])
                        checks['state_arrays']+=1
                prior=np.log((counts[client]+1.)/(counts[client].sum()+10.)).astype(np.float32)
                assert np.array_equal(states['analytic_prior'],prior)
                scores=transforms(arrays['raw_logits'],arrays['fixed_head_logits'],states)
                assert all(arrays['score_'+name].dtype==np.float64 for name in VARIANTS)
                for name in VARIANTS:
                    close(scores[name],arrays['score_'+name],'transform',0.)
                    checks['variant_score_arrays']+=1
                    metric=independent_metrics(scores[name],y)
                    recorded=report['clients'][client]['metrics'][name]
                    for key in ('accuracy','nll','loss','brier','ece15','samples'):
                        close(metric[key],recorded[key],'metric');checks['client_metric_values']+=1
                    assert recorded['loss']==recorded['nll']
                    for key in ('counts','confidence_sum','correct_sum'):
                        # Independent masked pairwise sums and sequential
                        # bincount accumulation need a sample-scaled FP64 bound.
                        close(metric['ece_bins'][key],recorded['ece_bins'][key],'bins',
                              128*np.finfo(np.float64).eps*len(y))
                        checks['ece_bin_values']+=15
                    recomputed[name].append(metric);score_groups[name].append(scores[name])
                checks['client_files']+=1
            assert np.array_equal(np.bincount(labels,minlength=10),np.full(10,1000))
            assert array_sha(labels,'<i8')==assignment['labels_int64le_sha256']==report['test_provenance']['labels_int64le_sha256']
            if labels_reference is None:labels_reference=labels.copy()
            else:assert np.array_equal(labels_reference,labels)
            weights=np.array([len(ids) for ids in groups],dtype=np.float64)/10000
            summaries={}
            for name in VARIANTS:
                summary={}
                for metric in ('accuracy','nll','brier','ece15'):
                    values=np.array([r[metric] for r in recomputed[name]])
                    summary['macro_'+metric]=float(sum(values)/20)
                    summary['weighted_'+metric]=float(sum(value*weight for value,weight in zip(values,weights)))
                summary['macro_loss']=summary['macro_nll'];summary['weighted_loss']=summary['weighted_nll']
                summary['client_accuracy_q10']=float(np.quantile([r['accuracy'] for r in recomputed[name]],.1))
                pooled=independent_metrics(np.concatenate(score_groups[name]),np.concatenate([labels[ids] for ids in groups]))
                summary['pooled_ece15']=pooled['ece15']
                for key,value in summary.items():close(value,report['summaries'][name][key],'summary');checks['summary_values']+=1
                summaries[name]=summary
            assert report['cost']['new_fit_calls']==report['cost']['new_cnn_backwards']==report['cost']['new_training_rounds']==0
            verify_hash(result_path,sha(result_path))
            reports[method,seed]=summaries
            input_evidence[f'{method}_s{seed}']={'result_sha256':sha(result_path),
                'test_source_records':raw_records,'labels_original_order_sha256':array_sha(labels,'<i8'),
                'root_lock_sha256':auth['root_lock_sha256'],'input_files_checked_without_official_raw':len(snapshot)-len(raw_records)}
    assert labels_reference is not None,'no completed exported report found'
    # Replay assignment independently from reconstructed exported GT and TRAIN counts.
    quotas=np.zeros((20,10),dtype=np.int64);replay=[[] for _ in range(20)]
    owner=np.full(10000,-1,dtype=np.int64)
    for label in range(10):
        denominator=sum(int(counts[c,label])+1 for c in range(20))
        numerators=[1000*(int(counts[c,label])+1) for c in range(20)]
        base=[n//denominator for n in numerators]
        order=sorted(range(20),key=lambda c:(-(numerators[c]%denominator),c))
        for client in order[:1000-sum(base)]:base[client]+=1
        ids=np.flatnonzero(labels_reference==label)
        np.random.Generator(np.random.PCG64(np.random.SeedSequence([20261301,label]))).shuffle(ids)
        offset=0
        for client,n in enumerate(base):
            quotas[client,label]=n;selected=ids[offset:offset+n]
            replay[client].extend(selected.tolist());owner[selected]=client;offset+=n
    assert np.array_equal(quotas,assignment['test_class_quotas'])
    assert all(np.array_equal(ids,group) for ids,group in zip(replay,groups))
    assert array_sha(owner,'<i8')==assignment['owner_int64le_sha256']
    train_prior=counts/counts.sum(axis=1,keepdims=True);test_prior=quotas/quotas.sum(axis=1,keepdims=True)
    for actual,expected in [(assignment['train_class_given_client'],train_prior),
        (assignment['test_class_given_client'],test_prior),
        (assignment['test_minus_train_class_frequency'],test_prior-train_prior),
        (assignment['per_client_total_variation'],.5*np.abs(test_prior-train_prior).sum(axis=1))]:close(actual,expected,'summary')
    task_result=root/'result.json'
    if task_result.exists() and not pending:
        task_report=json.loads(task_result.read_text())
        assert task_report['status']=='DONE' and task_report['fixed_three_seeds']==SEEDS[task]
        for method in METHODS:
            aggregate=task_report['summaries'][method]
            for name in VARIANTS:
                for key in reports[method,SEEDS[task][0]][name]:
                    values=[reports[method,seed][name][key] for seed in SEEDS[task]]
                    for stat,value in [('mean',np.mean(values)),('sample_sd',np.std(values,ddof=1))]:
                        close(value,aggregate['summaries'][name][key][stat],'aggregate');checks['three_seed_aggregate_values']+=1
            for control in VARIANTS:
                if control=='prioranchored_vector':continue
                for key in ('macro_accuracy','macro_nll','macro_brier','macro_ece15'):
                    values=[reports[method,seed]['prioranchored_vector'][key]-reports[method,seed][control][key] for seed in SEEDS[task]]
                    for stat,value in [('mean',np.mean(values)),('sample_sd',np.std(values,ddof=1))]:
                        close(value,aggregate['paired_main_minus_control'][control][key][stat],'aggregate');checks['paired_seed_values']+=1
        verify_hash(task_result,sha(task_result))
    if any(sha(path)!=digest for path,digest in hashed.items()):raise AssertionError('checked inputs changed')
    result={'status':'PASS' if not pending and task_result.exists() else 'PARTIAL_PASS',
        'scope':'local independent numerical/provenance check of exported arrays; not external review',
        'task':task,'root':str(root),'completed_models':len(reports),'expected_models':9,'pending':pending,
        'checks':checks,'max_absolute_errors':errors,'assignment_replayed_exactly':True,
        'coverage_original_ids_0_to_9999_exactly_once':True,'classes_1000_each':True,
        'GT_original_order_hash_matches_recorded_strict_loader_provenance':True,
        'official_raw_source_bytes_reopened':0,'official_raw_open_attempts':attempts,
        'new_model_forward_calls':0,'new_fit_calls':0,'new_backward_calls':0,
        'source_provenance_by_model':input_evidence,'checked_nonofficial_raw_file_sha256':hashed,
        'checker_sha256':sha(__file__),'external_review':'REVIEW_UNAVAILABLE',
        'limits':['Original official source MD5/SHA and image/GT extraction are checked as recorded strict-loader provenance, not independently reread here.',
                  'Raw/head network inference is not rerun; every final score transform and metric is recomputed from exported logits.',
                  'No gate, threshold, method, seed or training change is made.']}
    out.parent.mkdir(parents=True,exist_ok=True)
    out.with_suffix('.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    out.with_suffix('.md').write_text(f"# Independent final exported-array check\n\n{result['status']}: {task}, {len(reports)}/9 completed models.\n\n"
        f"Checks: `{checks}`.\n\nMaximum absolute errors: `{errors}`.\n\n"
        "All twelve FP64 score transforms reproduce exactly from exported raw/head logits and prelocked FP32 TRAIN states. Metrics, ECE bins, macro/sample-weighted/pooled/q10 summaries and available three-seed aggregates were independently recomputed.\n\n"
        "Original IDs cover0..9999 exactly once; per-class1000, deterministic TRAIN-count quotas/RNG assignment, GT original-order hash, locked model/state/source hashes and strict-loader provenance records agree. Official TEST raw files were not reopened. Raw network inference and extraction were not independently rerun.\n\n"
        "Local numerical/provenance verification only; external review REVIEW_UNAVAILABLE. No training, fit, backward or prediction inference was performed.\n")
    print(json.dumps({k:result[k] for k in ('status','task','completed_models','pending','checks','max_absolute_errors')},indent=2))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--out',type=Path,default=HERE/'INDEPENDENT_FINAL_CHECK')
    args=parser.parse_args();check(args.root,args.out)
