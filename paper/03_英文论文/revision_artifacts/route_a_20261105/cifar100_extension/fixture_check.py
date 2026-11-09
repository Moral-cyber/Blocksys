"""Numerical fixtures only: no images, CNN or official TEST reads."""
from pathlib import Path
import importlib.util
import hashlib
import json
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
OLD = Path('/Users/wind/Desktop/Codex/paper/05_核心方法研发/repro/work/prior_vector/solver.py')


def load(name,path):
    spec = importlib.util.spec_from_file_location(name,path)
    obj = importlib.util.module_from_spec(spec)
    sys.modules[name] = obj
    spec.loader.exec_module(obj)
    return obj


def main():
    new,old = load('_c100_structured_fixture',HERE/'solver.py'),load('_c100_old_fixture',OLD)
    rng = np.random.default_rng(20261107)
    equivalence = []
    for name,n,mode in [('balanced',60,'balanced'),('absent',137,'absent'),('batched',529,'balanced')]:
        z = rng.normal(size=(n,10))*2
        y = np.arange(n)%10 if mode=='balanced' else rng.integers(0,3,size=n)
        for kind,center in [('zero',np.zeros(10)),('prior',old.train_prior_anchor(y))]:
            pn,po = new.Problem(z,y,center),old.Problem(z,y,center)
            theta = pn.project(pn.anchor+rng.normal(size=20)*.1)
            nv,ng,nh = pn.objective(theta)
            ov,og,oh = po.objective(theta)
            assert abs(nv-ov) < 1e-10
            assert np.max(np.abs(ng-og)) < 1e-10
            assert np.max(np.abs(nh-oh)) < 1e-10
            sn,ni = new.fit(z,y,center)
            so,oi = old.fit(z,y,center)
            state_error = float(np.max(np.abs(sn.astype(float)-so.astype(float))))
            metric_error = abs(new.stats(new.transform(z,sn),y)['loss']-old.stats(old.transform(z,so),y)['loss'])
            assert state_error <= 2e-6
            assert metric_error <= 1e-7
            assert all(b['objective'] <= a['objective'] for a,b in zip(ni['trace'],ni['trace'][1:]))
            equivalence.append({'fixture':name,'center':kind,'fp32_bitwise_equal':bool(np.array_equal(sn,so)),
                                'fp32_state_max_error':state_error,'nll_error':metric_error,
                                'objective_error':abs(nv-ov),'gradient_error':float(np.max(np.abs(ng-og))),
                                'hessian_error':float(np.max(np.abs(nh-oh)))})
    z = rng.normal(size=(113,100))
    y = rng.integers(0,100,size=113)
    p = new.Problem(z,y,new.prior_anchor(y,100))
    theta = p.project(p.anchor+rng.normal(size=200)*.1)
    value,g,h = p.objective(theta)
    errors = []
    epsilon = 1e-5
    for _ in range(8):
        direction = rng.normal(size=200)
        direction /= np.linalg.norm(direction)
        plus,pg,_ = p.objective(theta+epsilon*direction)
        minus,mg,_ = p.objective(theta-epsilon*direction)
        ge = abs((plus-minus)/(2*epsilon)-float(g@direction))
        he = float(np.max(np.abs((pg-mg)/(2*epsilon)-h@direction)))
        assert ge < 2e-6 and he < 2e-6
        errors.append({'directional_gradient_abs_error':ge,'hessian_vector_max_error':he})
    eigen = float(np.linalg.eigvalsh(h).min())
    assert eigen >= 10.-1e-9
    state,info = new.fit(z,y,p.anchor[100:])
    assert state.shape == (200,) and state.nbytes == 800 and info['design_cache_bytes']==0
    assert abs(float(state[100:].astype(float).sum())) < 1e-5
    assert info['projected_gradient_inf'] <= 1e-5
    uniform = new.stats(np.zeros((100,100)),np.arange(100))
    perfect = new.stats(1000*np.eye(100),np.arange(100))
    assert abs(uniform['accuracy']-.01)<1e-15 and abs(uniform['loss']-np.log(100))<1e-12
    assert abs(uniform['brier']-.99)<1e-12 and uniform['ece15']<1e-12
    assert perfect['accuracy']==1. and perfect['loss']==0. and perfect['brier']==0. and perfect['ece15']==0.
    record = {'status':'PASS_NUMERICAL_FIXTURES','scope':'invented logit/count fixtures, not task results',
              'old10_equivalence':equivalence,'equivalence_bounds':{'state_abs':2e-6,'nll_abs':1e-7},
              'k100_finite_difference':errors,'k100_min_hessian_eigenvalue':eigen,
              'k100_fp32_state_bytes':state.nbytes,'k100_fit':info,
              'k100_metric_uniform_and_perfect_fixtures':True,
              'source_sha256':{str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in (HERE/'solver.py',HERE/'fixture_check.py',OLD,OLD.parent.parent/'calibration_baselines/solver.py')},
              'new_cnn_forwards':0,'new_cnn_training':0,'official_test_reads':0}
    (HERE/'FIXTURES.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({'status':record['status'],'old10_cases':len(equivalence),
                      'old10_bitwise_equal':sum(x['fp32_bitwise_equal'] for x in equivalence),
                      'max_state_error':max(x['fp32_state_max_error'] for x in equivalence),
                      'max_nll_error':max(x['nll_error'] for x in equivalence),
                      'k100_min_eigenvalue':eigen,'k100_state_bytes':state.nbytes},indent=2))


if __name__ == '__main__':
    main()
