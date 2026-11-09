"""Convex affine-logit calibration, fixed ridge10, NumPy float64 only."""
import time
import numpy as np

CLASSES = 10
RIDGE = 10.
MIN_SCALE = .001
MAX_STEPS = 50
ARMIJO = 1e-4
MAX_BACKTRACKS = 40
TOLERANCE = 1e-8
KINDS = ('zero_anchor_bias', 'temperature_scaling', 'bcts', 'vector_scaling')
DIMENSIONS = {'zero_anchor_bias': 10, 'temperature_scaling': 1, 'bcts': 11, 'vector_scaling': 20}


class Problem:
    def __init__(self, logits, labels, kind):
        self.logits = np.asarray(logits, dtype=np.float64)
        self.labels = np.asarray(labels)
        if (kind not in KINDS or self.logits.ndim != 2 or self.logits.shape[1] != CLASSES
                or not len(self.logits) or self.labels.shape != (len(self.logits),)
                or self.labels.dtype.kind not in 'iu' or np.any(self.labels < 0)
                or np.any(self.labels >= CLASSES) or not np.isfinite(self.logits).all()):
            raise ValueError('finite nonempty ten-class logits and integer labels required')
        self.kind, self.dimension = kind, DIMENSIONS[kind]
        self.anchor = np.zeros(self.dimension, dtype=np.float64)
        identity = np.broadcast_to(np.eye(CLASSES), (len(self.logits), CLASSES, CLASSES))
        if kind == 'zero_anchor_bias':
            self.offset, self.design = self.logits, identity.copy()
            self.bias_slice, self.scale_index = slice(0, 10), None
        elif kind == 'temperature_scaling':
            self.offset, self.design = 0., self.logits[:, :, None].copy()
            self.anchor[0] = 1.
            self.bias_slice, self.scale_index = None, 0
        elif kind == 'bcts':
            self.offset = 0.
            self.design = np.concatenate((self.logits[:, :, None], identity), axis=2)
            self.anchor[0] = 1.
            self.bias_slice, self.scale_index = slice(1, 11), 0
        else:
            self.offset = 0.
            diagonal = self.logits[:, :, None] * identity
            self.design = np.concatenate((diagonal, identity), axis=2)
            self.anchor[:10] = 1.
            self.bias_slice, self.scale_index = slice(10, 20), None
        self.design_flat = self.design.reshape(-1, self.dimension)

    def project(self, parameters):
        parameters = np.asarray(parameters, dtype=np.float64).copy()
        if parameters.shape != (self.dimension,) or not np.isfinite(parameters).all():
            raise ValueError('invalid calibration state')
        if self.scale_index is not None:
            parameters[0] = max(MIN_SCALE, parameters[0])
        if self.bias_slice is not None:
            parameters[self.bias_slice] -= parameters[self.bias_slice].mean()
        return parameters

    def score(self, parameters):
        return self.offset + np.einsum('nci,i->nc', self.design, parameters, optimize=True)

    def objective(self, parameters, derivatives=True):
        parameters = np.asarray(parameters, dtype=np.float64)
        if parameters.shape != (self.dimension,) or not np.isfinite(parameters).all():
            raise ValueError('invalid calibration parameters')
        scores = self.score(parameters)
        maxima = scores.max(axis=1)
        exponentials = np.exp(scores - maxima[:, None])
        probabilities = exponentials / exponentials.sum(axis=1, keepdims=True)
        loss = maxima + np.log(exponentials.sum(axis=1)) - scores[np.arange(len(scores)), self.labels]
        delta = parameters - self.anchor
        objective = float(loss.sum() + .5 * RIDGE * np.dot(delta, delta))
        if not derivatives:
            return objective
        error = probabilities.copy()
        error[np.arange(len(error)), self.labels] -= 1.
        gradient = np.einsum('nci,nc->i', self.design, error, optimize=True) + RIDGE * delta
        mean_design = np.einsum('nci,nc->ni', self.design, probabilities, optimize=True)
        weighted = (self.design * probabilities[:, :, None]).reshape(-1, self.dimension)
        hessian = self.design_flat.T @ weighted - mean_design.T @ mean_design + RIDGE * np.eye(self.dimension)
        return objective, gradient, .5 * (hessian + hessian.T)

    def projected_gradient(self, parameters, gradient):
        result = gradient.copy()
        if self.scale_index is not None and parameters[0] <= MIN_SCALE + 1e-12 and gradient[0] > 0:
            result[0] = 0.
        return result


def transform(logits, kind, parameters):
    logits = np.asarray(logits, dtype=np.float64)
    parameters = np.asarray(parameters, dtype=np.float64)
    if kind == 'zero_anchor_bias':
        return logits + parameters
    if kind == 'temperature_scaling':
        return logits * parameters[0]
    if kind == 'bcts':
        return logits * parameters[0] + parameters[1:]
    if kind == 'vector_scaling':
        return logits * parameters[:10] + parameters[10:]
    raise ValueError('unknown calibration kind')


def fit(logits, labels, kind):
    began = time.perf_counter()
    problem = Problem(logits, labels, kind)
    parameters = problem.project(problem.anchor)
    objective, gradient, hessian = problem.objective(parameters)
    trace = [{'step': 0, 'objective': objective,
        'projected_gradient_inf': float(np.abs(problem.projected_gradient(parameters, gradient)).max())}]
    solves, evaluations, accepted, reason = 0, 1, 0, 'maximum_steps'
    for step in range(1, MAX_STEPS + 1):
        projected_gradient = problem.projected_gradient(parameters, gradient)
        if np.abs(projected_gradient).max() <= TOLERANCE:
            reason = 'kkt_gradient_tolerance'
            break
        free = np.ones(problem.dimension, dtype=bool)
        if problem.scale_index is not None and parameters[0] <= MIN_SCALE + 1e-12 and gradient[0] > 0:
            free[0] = False
        direction = np.zeros(problem.dimension)
        direction[free] = -np.linalg.solve(hessian[np.ix_(free, free)], gradient[free])
        solves += 1
        if problem.bias_slice is not None:
            direction[problem.bias_slice] -= direction[problem.bias_slice].mean()
        alpha, found = 1., False
        for backtracks in range(MAX_BACKTRACKS + 1):
            trial = problem.project(parameters + alpha * direction)
            displacement = trial - parameters
            slope = float(np.dot(gradient, displacement))
            candidate = problem.objective(trial, derivatives=False)
            evaluations += 1
            if slope < 0 and candidate <= objective + ARMIJO * slope:
                found = True
                break
            alpha *= .5
        if not found:
            # Projected Newton can lose descent at the scalar bound. A fixed
            # projected-gradient fallback has no tunable hyperparameter grid.
            direction = -projected_gradient
            alpha = 1. / max(RIDGE, float(np.linalg.norm(hessian, ord=2)))
            for backtracks in range(MAX_BACKTRACKS + 1):
                trial = problem.project(parameters + alpha * direction)
                displacement = trial - parameters
                slope = float(np.dot(gradient, displacement))
                candidate = problem.objective(trial, derivatives=False)
                evaluations += 1
                if slope < 0 and candidate <= objective + ARMIJO * slope:
                    found = True
                    break
                alpha *= .5
        if not found:
            if np.abs(projected_gradient).max() < 1e-5:
                reason = 'float64_objective_resolution'
                break
            raise ValueError('fixed projected Armijo calibration did not find descent')
        decrease = objective - candidate
        if decrease < 0:
            raise ValueError('calibration objective increased')
        parameters = trial
        objective, gradient, hessian = problem.objective(parameters)
        evaluations += 1
        accepted += 1
        trace.append({'step': step, 'objective': objective, 'alpha': alpha,
            'backtracks': backtracks, 'projected_gradient_inf': float(np.abs(problem.projected_gradient(parameters, gradient)).max()),
            'scale_at_lower_bound': bool(problem.scale_index is not None and parameters[0] <= MIN_SCALE + 1e-12)})
        if decrease <= np.finfo(np.float64).eps * max(1., abs(objective)):
            reason = 'float64_objective_resolution'
            break
    state = parameters.astype(np.float32)
    return state, {'kind': kind, 'dimension': problem.dimension,
        'regularization': RIDGE, 'anchor': problem.anchor.tolist(),
        'parameters_float64': parameters.tolist(), 'state_dtype': 'float32',
        'state_bytes': int(state.nbytes), 'accepted_steps': accepted,
        'newton_linear_solves': solves, 'objective_evaluations': evaluations,
        'stop_reason': reason, 'initial_objective': trace[0]['objective'],
        'final_objective': objective,
        'projected_gradient_inf': float(np.abs(problem.projected_gradient(parameters, gradient)).max()),
        'hessian_min_eigenvalue': float(np.linalg.eigvalsh(hessian).min()),
        'constraint': 'a>=0.001' if problem.scale_index is not None else 'unconstrained scale; centered bias',
        'temperature': float(1. / parameters[0]) if kind == 'temperature_scaling' else None,
        'design_cache_bytes': int(problem.design.nbytes),
        'seconds': time.perf_counter() - began, 'trace': trace}


def stats(logits, labels):
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    maxima = logits.max(axis=1)
    losses = maxima + np.log(np.exp(logits - maxima[:, None]).sum(axis=1)) - logits[np.arange(len(logits)), labels]
    return {'accuracy': float(np.mean(logits.argmax(axis=1) == labels)),
            'loss': float(losses.mean()), 'samples': len(labels)}


def summarize(clients):
    weights = np.array([client['holdout_samples'] for client in clients], dtype=np.float64)
    weights /= weights.sum()
    result = {}
    for variant in clients[0]['metrics']:
        values = {key: np.array([client['metrics'][variant][key] for client in clients])
                  for key in ('accuracy', 'loss')}
        result[variant] = {**{'macro_' + key: float(value.mean()) for key, value in values.items()},
            **{'weighted_' + key: float(value @ weights) for key, value in values.items()},
            'client_accuracy_q10': float(np.quantile(values['accuracy'], .1))}
    return result
