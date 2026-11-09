"""Generic centered PSD bias-penalty fit for a fixed development candidate."""
from pathlib import Path
import importlib.util
import sys
import time

import numpy as np
from scipy.special import polygamma

ROOT = Path('/Users/wind/Desktop/Codex/paper/05_核心方法研发')
SOURCE = ROOT / 'scripts/prior_vector/solver.py'
spec = importlib.util.spec_from_file_location('_geometry_frozen_vector', SOURCE)
frozen = importlib.util.module_from_spec(spec); sys.modules[spec.name] = frozen; spec.loader.exec_module(frozen)
H = np.eye(10) - np.ones((10, 10)) / 10
J = np.ones((10, 10)) / 10


def weighted_precision(weights):
    w = np.asarray(weights, dtype=np.float64)
    if w.shape != (10,) or not np.isfinite(w).all() or np.any(w <= 0):
        raise ValueError('ten finite positive weights required')
    return np.diag(w) - np.outer(w, w) / w.sum()


def geometries(counts):
    counts = np.asarray(counts, dtype=np.float64)
    d = polygamma(1, counts + 1.)
    covariance = H @ np.diag(d) @ H
    q = weighted_precision(1. / d)
    count = weighted_precision(counts + .5)
    q *= 9. / np.trace(q); count *= 9. / np.trace(count)
    q = .5 * (q + q.T); count = .5 * (count + count.T)
    order = np.roll(np.arange(10), 1)
    permuted = q[np.ix_(order, order)]
    return {'geometry': q, 'permuted_geometry': permuted, 'count_ridge': count}, covariance


class Problem(frozen.Problem):
    def __init__(self, logits, labels, anchor, precision, scale_lambda=10., bias_lambda=10.):
        super().__init__(logits, labels, anchor)
        self.precision = np.asarray(precision, dtype=np.float64)
        if self.precision.shape != (10, 10) or not np.isfinite(self.precision).all():
            raise ValueError('finite10x10 PSD precision required')
        if np.max(np.abs(self.precision - self.precision.T)) > 1e-10 or np.max(np.abs(self.precision @ np.ones(10))) > 1e-9:
            raise ValueError('symmetric centered precision required')
        eigenvalues = np.linalg.eigvalsh(self.precision)
        if eigenvalues[0] < -1e-10 or eigenvalues[1] <= 0:
            raise ValueError('positive precision on nine-dimensional centered subspace required')
        self.penalty = np.zeros((20, 20))
        self.penalty[:10, :10] = scale_lambda * np.eye(10)
        self.penalty[10:, 10:] = bias_lambda * self.precision

    def objective(self, parameters, derivatives=True):
        scores = self.score(parameters)
        maximum = scores.max(axis=1)
        exp = np.exp(scores - maximum[:, None]); probabilities = exp / exp.sum(axis=1, keepdims=True)
        ce = maximum + np.log(exp.sum(axis=1)) - scores[np.arange(len(scores)), self.labels]
        residual = parameters - self.anchor
        value = float(ce.sum() + .5 * residual @ self.penalty @ residual)
        if not derivatives: return value
        error = probabilities.copy(); error[np.arange(len(error)), self.labels] -= 1.
        gradient = np.einsum('nci,nc->i', self.design, error, optimize=True) + self.penalty @ residual
        mean_design = np.einsum('nci,nc->ni', self.design, probabilities, optimize=True)
        weighted = (self.design * probabilities[:, :, None]).reshape(-1, self.dimension)
        hessian = self.design_flat.T @ weighted - mean_design.T @ mean_design + self.penalty
        return value, gradient, .5 * (hessian + hessian.T)

    def projected_gradient(self, parameters, gradient):
        result = gradient.copy(); result[10:] -= result[10:].mean(); return result


def fit(logits, labels, anchor, precision, scale_lambda=10., bias_lambda=10.):
    # The exact isotropic limit calls the unchanged original numerical schedule.
    # H and I are identical on centered b; this avoids changing legacy rounding.
    if np.array_equal(precision, H) and scale_lambda == 10. and bias_lambda == 10.:
        state, info = frozen.fit(logits, labels, anchor)
        return state, {**info, 'isotropic_frozen_dispatch': True}
    began = time.perf_counter()
    problem = Problem(logits, labels, anchor, precision, scale_lambda, bias_lambda)
    parameters = problem.project(problem.anchor)
    objective, gradient, hessian = problem.objective(parameters)
    trace = [objective]; reason = 'maximum_steps'; evaluations = 1
    gauge = np.zeros((20, 20)); gauge[10:, 10:] = J
    for step in range(1, frozen.base.MAX_STEPS + 1):
        pg = problem.projected_gradient(parameters, gradient)
        if np.max(np.abs(pg)) <= frozen.base.TOLERANCE:
            reason = 'kkt_gradient_tolerance'; break
        direction = -np.linalg.solve(hessian + gauge, pg)
        direction[10:] -= direction[10:].mean()
        found = False; alpha = 1.
        for _ in range(frozen.base.MAX_BACKTRACKS + 1):
            trial = problem.project(parameters + alpha * direction)
            slope = float(gradient @ (trial - parameters))
            candidate = problem.objective(trial, derivatives=False); evaluations += 1
            if slope < 0 and candidate <= objective + frozen.base.ARMIJO * slope:
                found = True; break
            alpha *= .5
        if not found:
            direction = -pg; alpha = 1. / max(10., float(np.linalg.norm(hessian, ord=2)))
            for _ in range(frozen.base.MAX_BACKTRACKS + 1):
                trial = problem.project(parameters + alpha * direction)
                slope = float(gradient @ (trial - parameters))
                candidate = problem.objective(trial, derivatives=False); evaluations += 1
                if slope < 0 and candidate <= objective + frozen.base.ARMIJO * slope:
                    found = True; break
                alpha *= .5
        if not found:
            if np.max(np.abs(pg)) < 1e-5:
                reason = 'float64_objective_resolution'; break
            raise ValueError('fixed Armijo PSD solve failed')
        decrease = objective - candidate
        if decrease < 0: raise ValueError('objective increased')
        parameters = trial; objective, gradient, hessian = problem.objective(parameters); evaluations += 1
        trace.append(objective)
        if decrease <= np.finfo(np.float64).eps * max(1., abs(objective)):
            reason = 'float64_objective_resolution'; break
    pg = problem.projected_gradient(parameters, gradient)
    if np.max(np.abs(pg)) > 1e-5: raise ValueError('KKT tolerance not satisfied')
    state = parameters.astype(np.float32)
    return state, {'kind': 'centered_psd_bias_penalty', 'isotropic_frozen_dispatch': False,
                   'scale_lambda': scale_lambda, 'bias_lambda': bias_lambda, 'stop_reason': reason,
                   'projected_gradient_inf': float(np.max(np.abs(pg))), 'objective_evaluations': evaluations,
                   'objective_trace': trace, 'parameters_float64': parameters.tolist(),
                   'state_dtype': 'float32', 'state_bytes': state.nbytes,
                   'precision_eigenvalues': np.linalg.eigvalsh(precision).tolist(),
                   'gauge_solver': 'unit curvature along common bias direction; objective unchanged',
                   'seconds': time.perf_counter() - began}
