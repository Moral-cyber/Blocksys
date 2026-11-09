"""Synthetic independent derivatives, monotonicity and gauge checks."""
from pathlib import Path
import hashlib
import json
import numpy as np
from solver import (CLASSES, REGULARIZATION, center, fit,
                    objective_gradient_hessian, stats)

HERE = Path(__file__).resolve().parent
rng = np.random.default_rng(20261231)
z, y = rng.normal(size=(37, 10)), rng.integers(0, 10, size=37)
prior, bias = center(rng.normal(size=10)), center(rng.normal(size=10))
objective, gradient, hessian = objective_gradient_hessian(z, y, bias, prior)
epsilon = 1e-5
numerical_gradient = np.empty(10)
numerical_hessian = np.empty((10, 10))
for column in range(10):
    offset = np.eye(10)[column] * epsilon
    plus = objective_gradient_hessian(z, y, bias + offset, prior)
    minus = objective_gradient_hessian(z, y, bias - offset, prior)
    numerical_gradient[column] = (plus[0] - minus[0]) / (2 * epsilon)
    numerical_hessian[:, column] = (plus[1] - minus[1]) / (2 * epsilon)
gradient_error = float(np.abs(gradient - numerical_gradient).max())
hessian_error = float(np.abs(hessian - numerical_hessian).max())
assert gradient_error < 1e-7 and hessian_error < 1e-7
assert np.max(np.abs(hessian - hessian.T)) < 1e-12
assert np.linalg.eigvalsh(hessian).min() >= REGULARIZATION - 1e-10
assert np.max(np.abs(hessian @ np.ones(10) - REGULARIZATION)) < 1e-12
# CE invariance to per-row logit constants; quadratic gauge relative to prior.
offsets = rng.normal(size=(37, 1)) * 100
gauge_objective = objective_gradient_hessian(z + offsets, y, bias, prior)
assert abs(gauge_objective[0] - objective) < 1e-10
assert np.max(np.abs(gauge_objective[1] - gradient)) < 1e-10
assert np.max(np.abs(gauge_objective[2] - hessian)) < 1e-10
both_shifted = objective_gradient_hessian(z, y, bias + 17, prior + 17)
assert abs(both_shifted[0] - objective) < 1e-10
cases = []
for name, logits, labels in (
        ('imbalanced', rng.normal(size=(103, 10)), rng.choice([0, 2, 7], size=103)),
        ('single_class', rng.normal(size=(21, 10)) * 20, np.full(21, 3)),
        ('one_example', np.full((1, 10), 1000.), np.array([9])),
        ('uniform_zero_logits', np.zeros((100, 10)), np.tile(np.arange(10), 10))):
    state, result = fit(logits, labels)
    shifted_state, shifted_result = fit(logits + 20., labels)
    assert state.dtype == np.float32 and state.nbytes == 40
    assert np.isfinite(state).all() and abs(state.mean()) < 1e-6
    assert result['final_objective'] <= result['initial_objective']
    assert all(left['objective'] >= right['objective'] for left, right in zip(result['trace'], result['trace'][1:]))
    assert result['accepted_newton_steps'] <= 20 and result['hessian_min_eigenvalue'] >= 10 - 1e-9
    assert np.max(np.abs(state - shifted_state)) < 1e-6
    assert np.isfinite(stats(logits + state, labels)['loss'])
    cases.append({'case': name, 'accepted_steps': result['accepted_newton_steps'],
        'initial_objective': result['initial_objective'], 'final_objective': result['final_objective'],
        'gradient_inf': result['final_gradient_inf'], 'stop_reason': result['stop_reason'],
        'mean_fp32_bias': float(state.mean()), 'state_bytes': state.nbytes,
        'logit_constant_shift_solution_max_error': float(np.max(np.abs(state - shifted_state)))})
report = {'status': 'PASS', 'seed': 20261231, 'scope': 'NumPy synthetic logits only; no CNN/data access',
    'analytic_gradient_fd_error': gradient_error, 'analytic_hessian_fd_error': hessian_error,
    'hessian_positive_definite_with_minimum_10': True,
    'hessian_data_gauge_null_direction': True, 'objective_gauge_checks': True,
    'fixed_armijo_descent_cases': cases,
    'source_sha256': {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
                      for name in ('solver.py', 'PLAN.md', 'local_fixture_check.py')}}
(HERE / 'LOCAL_FIXTURES.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
