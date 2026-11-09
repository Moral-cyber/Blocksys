"""Synthetic-only derivatives, geometry and frozen-isotropic regression checks."""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
from scipy.special import polygamma

import solver as s

HERE = Path(__file__).resolve().parent
attempts = []
def audit(event, args):
    if event == 'open' and isinstance(args[0], (str, bytes)):
        p = str(args[0])
        if any(v in p for v in ('predictive_final_test', 'test_batch', 't10k', '/dataset/', '/datasets/', 'checkpoint.pt')):
            attempts.append(p); raise ValueError('fixtures synthetic only')
sys.addaudithook(audit)
rng = np.random.default_rng(20261105)
z = rng.normal(size=(43, 10)); labels = rng.integers(0, 10, size=43)
anchor = s.frozen.train_prior_anchor(labels)
original, _ = s.frozen.fit(z, labels, anchor)
isotropic, _ = s.fit(z, labels, anchor, s.H)
if not np.array_equal(original, isotropic): raise ValueError('isotropic frozen FP32 regression failed')

counts = np.array([0, 0, 1, 2, 5, 9, 20, 49, 100, 1000])
geometries, covariance = s.geometries(counts)
d = polygamma(1, counts + 1.)
unscaled = s.weighted_precision(1. / d)
pinv_error = float(np.max(np.abs(np.linalg.pinv(covariance, hermitian=True) - unscaled)))
projection_error = float(np.max(np.abs(covariance @ unscaled - s.H)))
if pinv_error > 1e-7 or projection_error > 1e-10: raise ValueError('Q closed-form identity failed')

problem = s.Problem(z, labels, anchor, geometries['geometry'])
p = problem.project(problem.anchor + rng.normal(scale=.1, size=20))
iso_generic = s.Problem(z, labels, anchor, s.H)
iso_original = s.frozen.Problem(z, labels, anchor)
iv, ig, ih = iso_generic.objective(p); ov, og, oh = iso_original.objective(p)
projector = np.eye(20); projector[10:, 10:] = s.H
isotropic_objective_error = abs(iv - ov)
isotropic_gradient_error = float(np.max(np.abs(projector @ (ig - og))))
isotropic_hessian_error = float(np.max(np.abs(projector @ (ih - oh) @ projector)))
if max(isotropic_objective_error, isotropic_gradient_error, isotropic_hessian_error) > 1e-12:
    raise ValueError('generic isotropic objective differs on centered subspace')
value, gradient, hessian = problem.objective(p)
eps = 1e-5
numeric_gradient = np.zeros(20); numeric_hessian = np.zeros((20, 20))
for i in range(20):
    direction = np.eye(20)[i]
    numeric_gradient[i] = (problem.objective(p + eps * direction, False) - problem.objective(p - eps * direction, False)) / (2 * eps)
    gp = problem.objective(p + eps * direction)[1]; gm = problem.objective(p - eps * direction)[1]
    numeric_hessian[:, i] = (gp - gm) / (2 * eps)
gradient_error = float(np.max(np.abs(gradient - numeric_gradient)))
hessian_error = float(np.max(np.abs(hessian - numeric_hessian)))
if gradient_error > 2e-8 or hessian_error > 2e-8: raise ValueError('generic penalty derivative fixture failed')
if np.max(np.abs(geometries['geometry'] @ np.ones(10))) > 1e-10: raise ValueError('Q gauge failed')
if np.linalg.eigvalsh(geometries['geometry'])[1] <= 0: raise ValueError('centered PSD curvature failed')
state, fit = s.fit(z, labels, anchor, geometries['geometry'])
pg = fit['projected_gradient_inf']
if pg > 1e-5 or any(a < b for a, b in zip(fit['objective_trace'], fit['objective_trace'][1:])):
    raise ValueError('KKT/monotonicity fixture failed')
metric_uniform_nll = float(np.log(np.exp(np.zeros((10, 10))).sum(axis=1)).mean())
if abs(metric_uniform_nll - np.log(10)) > 1e-14: raise ValueError('known metric fixture failed')
result = {'status': 'PASS_SYNTHETIC_FIXTURES', 'isotropic_original_fp32_bitwise_equal': True,
          'Q_pseudoinverse_closed_form_max_error': pinv_error, 'Q_covariance_projection_max_error': projection_error,
          'finite_difference_gradient_max_error': gradient_error, 'finite_difference_hessian_max_error': hessian_error,
          'generic_penalty_projected_gradient_inf': pg, 'objective_monotone': True,
          'generic_isotropic_objective_error': isotropic_objective_error,
          'generic_isotropic_projected_gradient_error': isotropic_gradient_error,
          'generic_isotropic_projected_hessian_error': isotropic_hessian_error,
          'positive_on_nine_centered_directions': True, 'metric_uniform_nll': metric_uniform_nll,
          'official_test_reads': 0, 'forbidden_open_attempts': attempts,
          'sources': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__), HERE / 'solver.py')}}
(HERE / 'FIXTURES.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
