"""Synthetic finite differences, convexity, gauge and known calibration cases."""
from pathlib import Path
import hashlib
import json
import numpy as np
from solver import KINDS, RIDGE, MIN_SCALE, Problem, fit, transform

HERE = Path(__file__).resolve().parent
rng = np.random.default_rng(20261251)
logits, labels = rng.normal(size=(23, 10)), rng.integers(0, 10, size=23)
derivatives, known = [], []
for kind in KINDS:
    problem = Problem(logits, labels, kind)
    theta = problem.project(problem.anchor + rng.normal(size=problem.dimension) * .1)
    objective, gradient, hessian = problem.objective(theta)
    eps, numerical_gradient, numerical_hessian = 1e-5, [], []
    for column in range(problem.dimension):
        offset = np.eye(problem.dimension)[column] * eps
        plus, minus = problem.objective(theta + offset), problem.objective(theta - offset)
        numerical_gradient.append((plus[0] - minus[0]) / (2 * eps))
        numerical_hessian.append((plus[1] - minus[1]) / (2 * eps))
    gerror = float(np.max(np.abs(gradient - numerical_gradient)))
    herror = float(np.max(np.abs(hessian - np.array(numerical_hessian).T)))
    assert gerror < 1e-7 and herror < 1e-7
    assert np.linalg.eigvalsh(hessian).min() >= RIDGE - 1e-9
    assert np.max(np.abs(problem.score(theta) - transform(logits, kind, theta))) < 1e-12
    other = problem.project(problem.anchor + rng.normal(size=problem.dimension) * .2)
    assert problem.objective((theta + other) / 2, False) <= (problem.objective(theta, False) + problem.objective(other, False)) / 2 + 1e-12
    if problem.bias_slice is not None:
        gauge = np.zeros(problem.dimension)
        gauge[problem.bias_slice] = 1.
        shifted = theta + .7 * gauge
        # CE is invariant; ridge adds exactly lambda/2 * C * .7^2.
        assert abs(problem.objective(shifted, False) - objective - .5 * RIDGE * 10 * .7**2) < 1e-10
        assert np.max(np.abs(hessian @ gauge - RIDGE * gauge)) < 1e-10
    derivatives.append({'kind': kind, 'dimension': problem.dimension,
        'gradient_fd_max_error': gerror, 'hessian_fd_max_error': herror,
        'hessian_min_eigenvalue': float(np.linalg.eigvalsh(hessian).min()),
        'convex_jensen_and_bias_gauge_check': True})
    state, result = fit(np.zeros((100, 10)), np.tile(np.arange(10), 10), kind)
    assert np.array_equal(state, Problem(np.zeros((100, 10)), np.tile(np.arange(10), 10), kind).anchor.astype(np.float32))
    assert result['accepted_steps'] == 0 and state.nbytes == problem.dimension * 4
    known.append({'kind': kind, 'uniform_zero_logits_returns_anchor': True, 'state_bytes': state.nbytes})

fits = []
for name, z, y in [('imbalanced', rng.normal(size=(107, 10)), rng.choice([0, 3, 5], 107)),
                   ('single_class', rng.normal(size=(19, 10)) * 20, np.repeat(4, 19)),
                   ('large_logits', rng.normal(size=(17, 10)) * 100, rng.integers(0, 10, 17))]:
    for kind in KINDS:
        state, result = fit(z, y, kind)
        assert np.isfinite(state).all() and result['accepted_steps'] <= 50
        assert result['final_objective'] <= result['initial_objective']
        assert all(a['objective'] >= b['objective'] for a, b in zip(result['trace'], result['trace'][1:]))
        assert result['hessian_min_eigenvalue'] >= RIDGE - 1e-7
        if kind in ('temperature_scaling', 'bcts'):
            assert state[0] >= np.float32(MIN_SCALE)
        if kind != 'temperature_scaling':
            bias = state if kind == 'zero_anchor_bias' else state[-10:]
            assert abs(float(bias.mean())) < 1e-6
        fits.append({'case': name, 'kind': kind, 'steps': result['accepted_steps'],
            'stop_reason': result['stop_reason'], 'initial_objective': result['initial_objective'],
            'final_objective': result['final_objective'], 'kkt_gradient_inf': result['projected_gradient_inf'],
            'state_bytes': state.nbytes})
# A scalar anti-predictive score has its constrained optimum at a=0.001.
anti = np.full((30, 10), 4.)
anti[:, 0] = -4.
state, result = fit(anti, np.zeros(30, dtype=np.int64), 'temperature_scaling')
assert state[0] == np.float32(MIN_SCALE)
assert result['projected_gradient_inf'] < 1e-8
report = {'status': 'PASS', 'seed': 20261251, 'scope': 'synthetic NumPy arrays only',
    'analytic_derivatives_and_convexity': derivatives, 'known_anchor_cases': known,
    'descent_finite_cases': fits,
    'scalar_positive_boundary': {'a': float(state[0]), 'constraint': 'a>=0.001',
        'kkt_gradient_inf': result['projected_gradient_inf']},
    'source_sha256': {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
                      for name in ('PLAN.md', 'solver.py', 'local_fixture_check.py')}}
(HERE / 'LOCAL_FIXTURES.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({'status': report['status'], 'derivative_checks': derivatives,
                  'finite_descent_cases': len(fits), 'positive_boundary': report['scalar_positive_boundary']}, indent=2))
