"""Fixed NumPy float64, prior-anchored ten-dimensional bias calibration."""
import numpy as np

CLASSES = 10
REGULARIZATION = 10.
MAX_STEPS = 20
ARMIJO = 1e-4
MAX_BACKTRACKS = 40
GRADIENT_TOLERANCE = 1e-8


def center(value):
    value = np.asarray(value, dtype=np.float64)
    return value - value.mean()


def validate(logits, labels, bias, prior):
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels)
    bias, prior = np.asarray(bias, dtype=np.float64), np.asarray(prior, dtype=np.float64)
    if (logits.ndim != 2 or logits.shape[1] != CLASSES or not len(logits)
            or labels.shape != (len(logits),) or labels.dtype.kind not in 'iu'
            or np.any(labels < 0) or np.any(labels >= CLASSES)
            or bias.shape != (CLASSES,) or prior.shape != (CLASSES,)
            or not all(np.isfinite(value).all() for value in (logits, bias, prior))):
        raise ValueError('finite nonempty ten-class logits, integer labels and ten-dimensional bias/prior required')
    return logits, labels, bias, prior


def objective_gradient_hessian(logits, labels, bias, prior):
    logits, labels, bias, prior = validate(logits, labels, bias, prior)
    adjusted = logits + bias
    maxima = adjusted.max(axis=1)
    exponentials = np.exp(adjusted - maxima[:, None])
    probabilities = exponentials / exponentials.sum(axis=1, keepdims=True)
    logsumexp = maxima + np.log(exponentials.sum(axis=1))
    displacement = bias - prior
    objective = float((logsumexp - adjusted[np.arange(len(labels)), labels]).sum()
                      + .5 * REGULARIZATION * np.dot(displacement, displacement))
    gradient = probabilities.sum(axis=0) - np.bincount(labels, minlength=CLASSES)
    gradient += REGULARIZATION * displacement
    hessian = np.diag(probabilities.sum(axis=0)) - probabilities.T @ probabilities
    hessian += REGULARIZATION * np.eye(CLASSES)
    return objective, gradient, hessian


def fit(logits, labels):
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels)
    counts = np.bincount(labels, minlength=CLASSES)
    prior = center(np.log((counts + 1.) / (len(labels) + CLASSES)))
    bias = prior.copy()
    objective, gradient, hessian = objective_gradient_hessian(logits, labels, bias, prior)
    trace = [{'step': 0, 'objective': objective, 'gradient_inf': float(np.abs(gradient).max()),
              'bias_mean': float(bias.mean())}]
    accepted, evaluations, solves = 0, 1, 0
    reason = 'maximum_steps'
    for step in range(1, MAX_STEPS + 1):
        if np.abs(gradient).max() <= GRADIENT_TOLERANCE:
            reason = 'gradient_tolerance'
            break
        direction = center(np.linalg.solve(hessian, gradient))
        solves += 1
        directional_derivative = -float(np.dot(gradient, direction))
        if not np.isfinite(directional_derivative) or directional_derivative >= 0:
            raise ValueError('Newton direction must be a finite descent direction')
        alpha, found = 1., False
        for backtracks in range(MAX_BACKTRACKS + 1):
            trial = center(bias - alpha * direction)
            candidate, candidate_gradient, candidate_hessian = objective_gradient_hessian(logits, labels, trial, prior)
            evaluations += 1
            if candidate <= objective + ARMIJO * alpha * directional_derivative:
                found = True
                break
            alpha *= .5
        if not found:
            if -directional_derivative <= 1e-10 * max(1., abs(objective)):
                reason = 'float64_objective_resolution'
                break
            raise ValueError('fixed Armijo backtracking failed')
        improvement = objective - candidate
        if improvement < 0:
            raise ValueError('accepted Newton update increased objective')
        bias, objective, gradient, hessian = trial, candidate, candidate_gradient, candidate_hessian
        accepted += 1
        trace.append({'step': step, 'objective': objective, 'gradient_inf': float(np.abs(gradient).max()),
            'alpha': alpha, 'backtracks': backtracks, 'directional_derivative': directional_derivative,
            'bias_mean': float(bias.mean())})
        if improvement <= np.finfo(np.float64).eps * max(1., abs(objective)):
            reason = 'float64_objective_resolution'
            break
    state = bias.astype(np.float32)
    return state, {'prior_bias_float64': prior.tolist(), 'fitted_bias_float64': bias.tolist(),
        'train_counts': counts.tolist(), 'accepted_newton_steps': accepted,
        'linear_solves': solves, 'objective_evaluations': evaluations,
        'stop_reason': reason, 'initial_objective': trace[0]['objective'],
        'final_objective': objective, 'final_gradient_inf': float(np.abs(gradient).max()),
        'hessian_min_eigenvalue': float(np.linalg.eigvalsh(hessian).min()),
        'local_state_dtype': 'float32', 'local_state_bytes': int(state.nbytes), 'trace': trace}


def stats(logits, labels):
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    maxima = logits.max(axis=1)
    losses = maxima + np.log(np.exp(logits - maxima[:, None]).sum(axis=1)) - logits[np.arange(len(labels)), labels]
    return {'accuracy': float(np.mean(logits.argmax(axis=1) == labels)),
            'loss': float(losses.mean()), 'samples': len(labels)}


def summarize(clients):
    result = {}
    weights = np.array([client['holdout_samples'] for client in clients], dtype=np.float64)
    weights /= weights.sum()
    for variant in clients[0]['metrics']:
        values = {name: np.array([client['metrics'][variant][name] for client in clients])
                  for name in ('accuracy', 'loss')}
        result[variant] = {**{'macro_' + name: float(value.mean()) for name, value in values.items()},
            **{'weighted_' + name: float(np.dot(value, weights)) for name, value in values.items()},
            'client_accuracy_q10': float(np.quantile(values['accuracy'], .1))}
    return result
