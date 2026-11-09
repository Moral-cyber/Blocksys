"""Independent saved-array/state replay after all three runs; no model inference."""
from pathlib import Path
import json
import hashlib
import numpy as np

HERE = Path(__file__).resolve().parent
SEEDS = (20261231,20261232,20261233)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metric(scores,y):
    scores = np.asarray(scores,dtype=np.float64)
    y = np.asarray(y,dtype=np.int64)
    k = scores.shape[1]
    top = scores.max(axis=1)
    e = np.exp(scores-top[:,None])
    p = e/e.sum(axis=1,keepdims=True)
    nll = float((top+np.log(e.sum(axis=1))-scores[np.arange(len(y)),y]).mean())
    correct = scores.argmax(axis=1)==y
    onehot = np.zeros((len(y),k))
    onehot[np.arange(len(y)),y] = 1.
    brier = float(np.mean(np.sum((p-onehot)**2,axis=1)))
    confidence = p.max(axis=1)
    ece = 0.
    for lo in range(15):
        ids = (confidence>=lo/15)&(confidence<(lo+1)/15) if lo<14 else confidence>=14/15
        if ids.any():
            ece += float(ids.mean()*abs(confidence[ids].mean()-correct[ids].mean()))
    return {'loss':nll,'accuracy':float(correct.mean()),'brier':brier,'ece15':ece}


def main():
    lock = json.loads((HERE/'SOURCE_LOCK.json').read_text())
    data = json.loads((HERE/'DATA_PREFLIGHT.json').read_text())
    identities = {**lock['inputs'],**data['data_identities']}
    for path,digest in identities.items():
        if sha(path)!=digest:
            raise ValueError('source/data hash changed: '+path)
    errors,gradient_errors,records = [],[],[]
    for seed in SEEDS:
        folder = HERE/'runs'/f's{seed}'
        run = json.loads((folder/'evaluation.json').read_text())
        config = json.loads((folder/'config.json').read_text())
        if run['status']!='DONE' or run['official_test_reads'] or not run['identities_unchanged']:
            raise ValueError('incomplete or wrong-scope result')
        if sha(folder/'checkpoint.pt')!=run['checkpoint_sha256']:
            raise ValueError('checkpoint differs from scored identity')
        manifest = config['data_manifest']
        for client in run['clients']:
            i = client['client']
            with np.load(folder/f'client{i}.npz') as f:
                z,y,hz,hy = [f[key] for key in ('train_logits','train_labels','holdout_logits','holdout_labels')]
                if not np.array_equal(f['train_ids'],manifest['train_indices'][i]) or not np.array_equal(f['holdout_ids'],manifest['holdout_indices'][i]):
                    raise ValueError('saved membership differs from locked manifest')
                counts = np.bincount(y,minlength=100)
                if not np.array_equal(counts,client['train_counts']):
                    raise ValueError('saved TRAIN counts mismatch')
                mu = np.log(counts+1.)
                mu -= mu.mean()
                for variant in ('raw','fixed_prior','zero','prior'):
                    if variant=='raw':
                        scores = hz.astype(np.float64)
                    else:
                        state = f[variant]
                        if state.shape!=(200,) or state.dtype!=np.float32 or state.nbytes!=800:
                            raise ValueError('wrong persistent state')
                        scores = hz.astype(np.float64)*state[:100].astype(float)+state[100:].astype(float)
                        if variant=='fixed_prior' and not np.array_equal(state,np.r_[np.ones(100),mu].astype(np.float32)):
                            raise ValueError('fixed prior differs from actual local TRAIN count center')
                    recomputed = metric(scores,hy)
                    for key,value in recomputed.items():
                        errors.append(abs(value-client['metrics'][variant][key]))
                    if variant in ('zero','prior'):
                        info = client['fits'][variant]
                        theta = np.asarray(info['parameters_float64'])
                        center = np.r_[np.ones(100),mu if variant=='prior' else np.zeros(100)]
                        s = z.astype(float)*theta[:100]+theta[100:]
                        pp = np.exp(s-s.max(axis=1,keepdims=True))
                        pp /= pp.sum(axis=1,keepdims=True)
                        pp[np.arange(len(y)),y] -= 1.
                        gradient = np.r_[(pp*z).sum(axis=0),pp.sum(axis=0)]+10.*(theta-center)
                        gradient[100:] -= gradient[100:].mean()
                        gradient_errors.append(float(np.max(np.abs(gradient))))
                        if info['design_cache_bytes']!=0 or info['hessian_bytes']!=320000:
                            raise ValueError('structured solver memory declaration differs')
                        if any(b['objective']>a['objective'] for a,b in zip(info['trace'],info['trace'][1:])):
                            raise ValueError('objective monotonicity failure')
            records.append({'seed':seed,'client':i,'variants':4})
        identities[str(folder/'checkpoint.pt')] = run['checkpoint_sha256']
        identities[str(folder/'evaluation.json')] = sha(folder/'evaluation.json')
    if max(errors)>1e-12 or max(gradient_errors)>1e-5:
        raise ValueError('independent metric or KKT replay failed')
    result = {'status':'PASS_INDEPENDENT_SAVED_ARRAY_REPLAY','runs':3,'clients':len(records),
              'arm_client_rows':4*len(records),'metric_max_abs_error':max(errors),
              'fp64_fitted_projected_gradient_max':max(gradient_errors),
              'source_data_checkpoint_sha256':identities,'official_test_reads':0,
              'new_cnn_forwards':0,'new_cnn_training':0,'new_predictor_fits':0,
              'scope':'local independent formula replay, not external peer review or new final TEST'}
    (HERE/'INDEPENDENT_RESULTS_VERIFICATION.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='source_data_checkpoint_sha256'},indent=2))


if __name__=='__main__':
    main()
