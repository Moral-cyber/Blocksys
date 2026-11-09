"""Explicit configurable-anchor wrapper around frozen vector-scaling math."""
from pathlib import Path
import hashlib
import importlib.util
import sys
import time

import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
BASE_PATH=HERE.parent/'calibration_baselines/solver.py'
BASE_SHA256='10b11405e6662cb06d08a20549db92e2948b730c426aa1f0b912b958f1211f1c'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


if sha(BASE_PATH)!=BASE_SHA256:
    raise ValueError('frozen calibration solver identity mismatch')
spec=importlib.util.spec_from_file_location('_prior_vector_frozen_math',BASE_PATH)
base=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=base
spec.loader.exec_module(base)


def train_prior_anchor(labels):
    """Only local TRAIN labels determine the centered additive prior anchor."""
    labels=np.asarray(labels)
    if (labels.ndim!=1 or not len(labels) or labels.dtype.kind not in 'iu'
            or np.any(labels<0) or np.any(labels>=base.CLASSES)):
        raise ValueError('nonempty ten-class integer TRAIN labels required')
    counts=np.bincount(labels,minlength=base.CLASSES)
    anchor=np.log(counts.astype(np.float64)+1.)
    return anchor-anchor.mean()


class Problem(base.Problem):
    """The frozen affine design/objective with an explicit centered b anchor."""
    def __init__(self,logits,labels,bias_anchor):
        super().__init__(logits,labels,'vector_scaling')
        bias_anchor=np.asarray(bias_anchor,dtype=np.float64)
        if bias_anchor.shape!=(base.CLASSES,) or not np.isfinite(bias_anchor).all():
            raise ValueError('finite ten-dimensional bias anchor required')
        self.anchor[10:]=bias_anchor-bias_anchor.mean()


def fit(logits,labels,bias_anchor):
    """Same fixed Newton/Armijo schedule; the supplied anchor is the sole change.

    This independent implementation copies the frozen numerical schedule rather
    than modifying that module's Problem class, globals, source or fitted states.
    """
    began=time.perf_counter()
    problem=Problem(logits,labels,bias_anchor)
    parameters=problem.project(problem.anchor)
    objective,gradient,hessian=problem.objective(parameters)
    trace=[{'step':0,'objective':objective,
        'projected_gradient_inf':float(np.abs(problem.projected_gradient(parameters,gradient)).max())}]
    solves,evaluations,accepted,reason=0,1,0,'maximum_steps'
    for step in range(1,base.MAX_STEPS+1):
        projected_gradient=problem.projected_gradient(parameters,gradient)
        if np.abs(projected_gradient).max()<=base.TOLERANCE:
            reason='kkt_gradient_tolerance'
            break
        free=np.ones(problem.dimension,dtype=bool)
        # Vector scaling has no scalar lower bound; all 20 variables are free.
        direction=np.zeros(problem.dimension)
        direction[free]=-np.linalg.solve(hessian[np.ix_(free,free)],gradient[free])
        solves+=1
        direction[problem.bias_slice]-=direction[problem.bias_slice].mean()
        alpha,found=1.,False
        for backtracks in range(base.MAX_BACKTRACKS+1):
            trial=problem.project(parameters+alpha*direction)
            displacement=trial-parameters
            slope=float(np.dot(gradient,displacement))
            candidate=problem.objective(trial,derivatives=False)
            evaluations+=1
            if slope<0 and candidate<=objective+base.ARMIJO*slope:
                found=True
                break
            alpha*=.5
        if not found:
            direction=-projected_gradient
            alpha=1./max(base.RIDGE,float(np.linalg.norm(hessian,ord=2)))
            for backtracks in range(base.MAX_BACKTRACKS+1):
                trial=problem.project(parameters+alpha*direction)
                displacement=trial-parameters
                slope=float(np.dot(gradient,displacement))
                candidate=problem.objective(trial,derivatives=False)
                evaluations+=1
                if slope<0 and candidate<=objective+base.ARMIJO*slope:
                    found=True
                    break
                alpha*=.5
        if not found:
            if np.abs(projected_gradient).max()<1e-5:
                reason='float64_objective_resolution'
                break
            raise ValueError('fixed projected Armijo vector calibration did not find descent')
        decrease=objective-candidate
        if decrease<0:
            raise ValueError('vector calibration objective increased')
        parameters=trial
        objective,gradient,hessian=problem.objective(parameters)
        evaluations+=1
        accepted+=1
        trace.append({'step':step,'objective':objective,'alpha':alpha,'backtracks':backtracks,
            'projected_gradient_inf':float(np.abs(problem.projected_gradient(parameters,gradient)).max()),
            'scale_at_lower_bound':False})
        if decrease<=np.finfo(np.float64).eps*max(1.,abs(objective)):
            reason='float64_objective_resolution'
            break
    state=parameters.astype(np.float32)
    return state,{'kind':'vector_scaling_configurable_bias_anchor','dimension':problem.dimension,
        'regularization_scale':base.RIDGE,'regularization_bias':base.RIDGE,
        'anchor':problem.anchor.tolist(),'parameters_float64':parameters.tolist(),
        'state_dtype':'float32','state_bytes':int(state.nbytes),'accepted_steps':accepted,
        'newton_linear_solves':solves,'objective_evaluations':evaluations,'stop_reason':reason,
        'initial_objective':trace[0]['objective'],'final_objective':objective,
        'projected_gradient_inf':float(np.abs(problem.projected_gradient(parameters,gradient)).max()),
        'hessian_min_eigenvalue':float(np.linalg.eigvalsh(hessian).min()),
        'constraint':'all scales unrestricted real; centered bias',
        'design_cache_bytes':int(problem.design.nbytes),
        'design_flat_shares_memory':bool(np.shares_memory(problem.design,problem.design_flat)),
        'train_logits_float64_bytes':int(problem.logits.nbytes),
        'hessian_float64_bytes':int(hessian.nbytes),
        'seconds':time.perf_counter()-began,'trace':trace}


def install_data_guard():
    attempts=[]
    def audit(event,arguments):
        if event=='open' and isinstance(arguments[0],(str,bytes)):
            path=str(arguments[0])
            if '/dataset/' in path or '/datasets/' in path or 'test_batch' in path or 't10k' in path:
                attempts.append(path)
                raise ValueError('prior vector permits saved arrays only; dataset/TEST bytes forbidden')
    sys.addaudithook(audit)
    return attempts


transform=lambda logits,parameters:base.transform(logits,'vector_scaling',parameters)
stats=base.stats
summarize=base.summarize
