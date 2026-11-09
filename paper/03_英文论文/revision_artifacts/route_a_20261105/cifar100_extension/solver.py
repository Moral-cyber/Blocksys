"""K-class diagonal affine calibration without an N*K*2K design cache."""
import time
import numpy as np

RIDGE = 10.
MAX_STEPS = 50
ARMIJO = 1e-4
MAX_BACKTRACKS = 40
TOLERANCE = 1e-8
BATCH = 512


def prior_anchor(labels, classes):
    labels = np.asarray(labels)
    if (labels.ndim != 1 or not len(labels) or labels.dtype.kind not in 'iu' or
            np.any(labels < 0) or np.any(labels >= classes)):
        raise ValueError('finite K-class integer TRAIN labels required')
    value = np.log(np.bincount(labels, minlength=classes).astype(np.float64)+1.)
    return value-value.mean()


class Problem:
    def __init__(self, logits, labels, bias_anchor):
        self.logits = np.asarray(logits, dtype=np.float64)
        self.labels = np.asarray(labels)
        if self.logits.ndim != 2 or self.logits.shape[1] < 2 or not len(self.logits):
            raise ValueError('nonempty N by K logits required')
        self.classes = k = self.logits.shape[1]
        if (self.labels.shape != (len(self.logits),) or self.labels.dtype.kind not in 'iu' or
                np.any(self.labels < 0) or np.any(self.labels >= k) or not np.isfinite(self.logits).all()):
            raise ValueError('finite logits and in-range integer labels required')
        center = np.asarray(bias_anchor, dtype=np.float64)
        if center.shape != (k,) or not np.isfinite(center).all():
            raise ValueError('finite K-dimensional anchor required')
        self.dimension = 2*k
        self.anchor = np.concatenate((np.ones(k), center-center.mean()))

    def project(self, theta):
        theta = np.asarray(theta, dtype=np.float64).copy()
        if theta.shape != (self.dimension,) or not np.isfinite(theta).all():
            raise ValueError('invalid affine state')
        theta[self.classes:] -= theta[self.classes:].mean()
        return theta

    def projected_gradient(self, theta, gradient):
        value = np.asarray(gradient).copy()
        value[self.classes:] -= value[self.classes:].mean()
        return value

    def objective(self, theta, derivatives=True):
        theta = np.asarray(theta, dtype=np.float64)
        if theta.shape != (self.dimension,) or not np.isfinite(theta).all():
            raise ValueError('invalid objective state')
        k = self.classes
        a,b = theta[:k],theta[k:]
        delta = theta-self.anchor
        objective = .5*RIDGE*float(delta@delta)
        if derivatives:
            ga,gb = np.zeros(k),np.zeros(k)
            haa,hab,hbb = np.zeros((k,k)),np.zeros((k,k)),np.zeros((k,k))
        for start in range(0,len(self.labels),BATCH):
            z = self.logits[start:start+BATCH]
            y = self.labels[start:start+BATCH]
            s = z*a+b
            maxima = s.max(axis=1)
            ex = np.exp(s-maxima[:,None])
            p = ex/ex.sum(axis=1,keepdims=True)
            objective += float((maxima+np.log(ex.sum(axis=1))-s[np.arange(len(y)),y]).sum())
            if not derivatives:
                continue
            error = p.copy()
            error[np.arange(len(y)),y] -= 1.
            ga += (z*error).sum(axis=0)
            gb += error.sum(axis=0)
            pz = p*z
            haa += np.diag((pz*z).sum(axis=0))-pz.T@pz
            hab += np.diag(pz.sum(axis=0))-pz.T@p
            hbb += np.diag(p.sum(axis=0))-p.T@p
        if not derivatives:
            return objective
        gradient = np.concatenate((ga,gb))+RIDGE*delta
        hessian = np.block([[haa,hab],[hab.T,hbb]])+RIDGE*np.eye(2*k)
        return objective,gradient,.5*(hessian+hessian.T)


def fit(logits, labels, bias_anchor):
    began = time.perf_counter()
    problem = Problem(logits,labels,bias_anchor)
    theta = problem.project(problem.anchor)
    objective,g,h = problem.objective(theta)
    trace = [{'step':0,'objective':objective,
              'projected_gradient_inf':float(np.abs(problem.projected_gradient(theta,g)).max())}]
    solves,evaluations,accepted,reason = 0,1,0,'maximum_steps'
    for step in range(1,MAX_STEPS+1):
        pg = problem.projected_gradient(theta,g)
        if np.abs(pg).max() <= TOLERANCE:
            reason = 'kkt_gradient_tolerance'
            break
        direction = -np.linalg.solve(h,g)
        direction[problem.classes:] -= direction[problem.classes:].mean()
        solves += 1
        alpha,found = 1.,False
        for backtracks in range(MAX_BACKTRACKS+1):
            trial = problem.project(theta+alpha*direction)
            slope = float(g@(trial-theta))
            candidate = problem.objective(trial,derivatives=False)
            evaluations += 1
            if slope < 0 and candidate <= objective+ARMIJO*slope:
                found = True
                break
            alpha *= .5
        if not found:
            direction = -pg
            alpha = 1./max(RIDGE,float(np.linalg.norm(h,ord=2)))
            for backtracks in range(MAX_BACKTRACKS+1):
                trial = problem.project(theta+alpha*direction)
                slope = float(g@(trial-theta))
                candidate = problem.objective(trial,derivatives=False)
                evaluations += 1
                if slope < 0 and candidate <= objective+ARMIJO*slope:
                    found = True
                    break
                alpha *= .5
        if not found:
            if np.abs(pg).max() < 1e-5:
                reason = 'float64_objective_resolution'
                break
            raise ValueError('fixed structured Armijo solver cannot find descent')
        decrease = objective-candidate
        if decrease < 0:
            raise ValueError('objective increased')
        theta = trial
        objective,g,h = problem.objective(theta)
        evaluations += 1
        accepted += 1
        trace.append({'step':step,'objective':objective,'alpha':alpha,'backtracks':backtracks,
                      'projected_gradient_inf':float(np.abs(problem.projected_gradient(theta,g)).max())})
        if decrease <= np.finfo(np.float64).eps*max(1.,abs(objective)):
            reason = 'float64_objective_resolution'
            break
    residual = float(np.abs(problem.projected_gradient(theta,g)).max())
    if residual > 1e-5:
        raise ValueError('structured solver did not converge within declared KKT bound')
    state = theta.astype(np.float32)
    return state,{'kind':'K_class_structured_affine','classes':problem.classes,
        'dimension':problem.dimension,'lambda':RIDGE,'anchor':problem.anchor.tolist(),
        'parameters_float64':theta.tolist(),'state_dtype':'float32','state_bytes':int(state.nbytes),
        'accepted_steps':accepted,'newton_linear_solves':solves,'objective_evaluations':evaluations,
        'stop_reason':reason,'initial_objective':trace[0]['objective'],'final_objective':objective,
        'projected_gradient_inf':residual,'hessian_min_eigenvalue':float(np.linalg.eigvalsh(h).min()),
        'hessian_bytes':int(h.nbytes),'design_cache_bytes':0,
        'train_logits_float64_bytes':int(problem.logits.nbytes),'derivative_batch_rows':BATCH,
        'constraint':'unrestricted scales; centered K biases','seconds':time.perf_counter()-began,'trace':trace}


def transform(logits,state):
    logits = np.asarray(logits,dtype=np.float64)
    k = logits.shape[1]
    state = np.asarray(state,dtype=np.float64)
    if state.shape != (2*k,):
        raise ValueError('wrong K-class state shape')
    return logits*state[:k]+state[k:]


def stats(scores,labels):
    scores = np.asarray(scores,dtype=np.float64)
    labels = np.asarray(labels,dtype=np.int64)
    mx = scores.max(axis=1)
    ex = np.exp(scores-mx[:,None])
    p = ex/ex.sum(axis=1,keepdims=True)
    losses = mx+np.log(ex.sum(axis=1))-scores[np.arange(len(labels)),labels]
    correct = scores.argmax(axis=1)==labels
    brier = (p*p).sum(axis=1)-2*p[np.arange(len(labels)),labels]+1.
    confidence = p.max(axis=1)
    bins = np.minimum((confidence*15).astype(np.int64),14)
    count = np.bincount(bins,minlength=15)
    confidence_sum = np.bincount(bins,weights=confidence,minlength=15)
    correct_sum = np.bincount(bins,weights=correct.astype(float),minlength=15)
    ece = float(np.abs(confidence_sum-correct_sum).sum()/len(labels))
    return {'accuracy':float(correct.mean()),'loss':float(losses.mean()),'samples':len(labels),
            'brier':float(brier.mean()),'ece15':ece,'ece_bin_count':count.tolist(),
            'ece_bin_confidence_sum':confidence_sum.tolist(),'ece_bin_correct_sum':correct_sum.tolist()}


def summarize(clients):
    weights = np.asarray([c['holdout_samples'] for c in clients],dtype=np.float64)
    weights /= weights.sum()
    result = {}
    for variant in clients[0]['metrics']:
        values = {key:np.asarray([c['metrics'][variant][key] for c in clients]) for key in ('accuracy','loss','brier','ece15')}
        bin_confidence = np.sum([c['metrics'][variant]['ece_bin_confidence_sum'] for c in clients],axis=0)
        bin_correct = np.sum([c['metrics'][variant]['ece_bin_correct_sum'] for c in clients],axis=0)
        samples = sum(c['holdout_samples'] for c in clients)
        result[variant] = {**{'macro_'+k:float(v.mean()) for k,v in values.items()},
                           **{'weighted_'+k:float(v@weights) for k,v in values.items()},
                           'client_accuracy_q10':float(np.quantile(values['accuracy'],.1)),
                           'pooled_ece15':float(np.abs(bin_confidence-bin_correct).sum()/samples)}
    return result
