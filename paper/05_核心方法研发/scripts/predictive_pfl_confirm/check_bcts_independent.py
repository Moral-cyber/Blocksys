"""Independent BCTS saved-array audit; author code is used only on small CPU toys."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / 'work/predictive_pfl_confirm'
VECTOR = ROOT / 'work/prior_vector'
RAW_ATTEMPTS = []


def guard(event, arguments):
    if event == 'open' and isinstance(arguments[0], (str, bytes)):
        path = str(arguments[0])
        if '/dataset/' in path or '/datasets/' in path or '/rawdata/' in path or 'test_batch' in path or 't10k' in path:
            RAW_ATTEMPTS.append(path)
            raise RuntimeError('BCTS independent audit permits saved arrays and CPU toys only')


sys.addaudithook(guard)
import numpy as np

PROTECTED = {}
CHECKS = []
ERRORS = {}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def protect(path):
    path = Path(path).resolve()
    digest = sha(path)
    if str(path) in PROTECTED and PROTECTED[str(path)] != digest:
        raise AssertionError('source/input changed during checks: ' + str(path))
    PROTECTED[str(path)] = digest
    return digest


def read_json(path):
    protect(path)
    return json.loads(Path(path).read_text())


def arrays(path):
    protect(path)
    with np.load(path, allow_pickle=False) as source:
        return {name: source[name].copy() for name in source.files}


def check(value, label):
    if not bool(value):
        raise AssertionError(label)
    CHECKS.append(label)


def close(actual, expected, label, tolerance=3e-12):
    error = float(np.max(np.abs(np.asarray(actual, dtype=float) - np.asarray(expected, dtype=float))))
    ERRORS[label] = max(ERRORS.get(label, 0.), error)
    check(error <= tolerance, label)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def objective(z, y, parameters, anchor, derivatives=True):
    """Direct11D scalar-affine objective, gradient and covariance Hessian."""
    a, bias = parameters[0], parameters[1:]
    score = a * z + bias
    lse = np.logaddexp.reduce(score, axis=1)
    probability = np.exp(score - lse[:, None])
    difference = parameters - anchor
    value = float(np.sum(lse - score[np.arange(len(y)), y]) + 5 * difference @ difference)
    if not derivatives:
        return value
    error = probability.copy()
    error[np.arange(len(y)), y] -= 1
    gradient = np.r_[np.sum(error * z), np.sum(error, axis=0)] + 10 * difference
    expected_z = np.sum(probability * z, axis=1)
    hessian = np.empty((11, 11))
    hessian[0, 0] = np.sum(np.sum(probability * z * z, axis=1) - expected_z * expected_z) + 10
    hessian[0, 1:] = np.sum(probability * (z - expected_z[:, None]), axis=0)
    hessian[1:, 0] = hessian[0, 1:]
    hessian[1:, 1:] = np.diag(probability.sum(axis=0)) - probability.T @ probability + 10 * np.eye(10)
    return value, gradient, (hessian + hessian.T) / 2


def kkt(parameters, gradient):
    residual = gradient.copy()
    if parameters[0] <= .001 + 1e-12 and gradient[0] > 0:
        residual[0] = 0.
    return float(np.abs(residual).max())


def refit(z, y, anchor):
    """Independent10D centered-bias reduced Newton with the same fixed a bound."""
    contrast = np.eye(10)[:, :9] - np.eye(10)[:, [9]]
    q, _ = np.linalg.qr(contrast)
    basis = np.zeros((11, 10))
    basis[0, 0] = 1.
    basis[1:, 1:] = q
    reduced = basis.T @ anchor
    trace = []
    for step in range(100):
        full = basis @ reduced
        value, gradient, hessian = objective(z, y, full, anchor)
        g, h = basis.T @ gradient, basis.T @ hessian @ basis
        free = np.ones(10, dtype=bool)
        if reduced[0] <= .001 + 1e-12 and g[0] > 0:
            free[0] = False
        residual = g.copy()
        residual[~free] = 0
        trace.append({'step': step, 'objective': value, 'reduced_kkt_inf': float(np.abs(residual).max())})
        if np.abs(residual).max() <= 5e-10:
            break
        factor = np.linalg.cholesky(h[np.ix_(free, free)])
        direction = np.zeros(10)
        direction[free] = np.linalg.solve(factor.T, np.linalg.solve(factor, -g[free]))
        rate = 1.
        for _ in range(60):
            proposal = reduced + rate * direction
            proposal[0] = max(.001, proposal[0])
            slope = g @ (proposal - reduced)
            trial = objective(z, y, basis @ proposal, anchor, False)
            if trial <= value + 1e-4 * slope and slope < 0:
                break
            rate /= 2
        else:
            break
        displacement = proposal - reduced
        reduced = proposal
        if np.linalg.norm(displacement, ord=np.inf) < 2e-13:
            break
    full = basis @ reduced
    value, gradient, _ = objective(z, y, full, anchor)
    return full, {'objective': value, 'kkt_inf': kkt(full, gradient), 'steps': len(trace) - 1, 'trace': trace}


def toys():
    # Import authored code only for synthetic equivalence/derivative fixtures.
    solver = load('_independent_bcts_author_vector', VECTOR / 'solver.py')
    previous = sys.modules.get('solver')
    sys.modules['solver'] = solver
    try:
        authored = load('_independent_bcts_author_control', VECTOR / 'bcts_control.py')
    finally:
        if previous is None:
            sys.modules.pop('solver', None)
        else:
            sys.modules['solver'] = previous
    original_factory = solver.base.fit.__globals__['Problem']
    rng = np.random.default_rng(20261371)
    z = rng.normal(size=(31, 10))
    y = rng.choice([1, 4, 8], size=31)
    anchor_bias = np.log(np.bincount(y, minlength=10) + 1.)
    anchor_bias -= anchor_bias.mean()
    anchor = np.r_[1., anchor_bias]
    parameters = anchor + rng.normal(size=11) * .08
    parameters[1:] -= parameters[1:].mean()
    current = objective(z, y, parameters, anchor)
    author_problem = authored.Problem(z, y, anchor_bias)
    author = author_problem.objective(parameters)
    for value, expected, name in zip(current, author, ('toy objective', 'toy gradient', 'toy Hessian')):
        close(value, expected, name, 2e-11)
    eps = 1e-5
    gradient_fd, hessian_fd = [], []
    for column in range(11):
        offset = np.eye(11)[column] * eps
        plus, minus = objective(z, y, parameters + offset, anchor), objective(z, y, parameters - offset, anchor)
        gradient_fd.append((plus[0] - minus[0]) / (2 * eps))
        hessian_fd.append((plus[1] - minus[1]) / (2 * eps))
    grad_error = float(np.max(np.abs(current[1] - gradient_fd)))
    hess_error = float(np.max(np.abs(current[2] - np.asarray(hessian_fd).T)))
    check(grad_error < 1e-7 and hess_error < 1e-7, 'independent scalar-BCTS finite difference gradient/Hessian')
    check(np.linalg.eigvalsh(current[2]).min() >= 10 - 1e-10, 'independent Hessian convex with ridge10')
    zero_anchor = np.r_[1., np.zeros(10)]
    removed = objective(z, y, parameters, zero_anchor)
    close(current[2], removed[2], 'prior anchor changes no Hessian', 0.)
    close(current[1] - removed[1], -10 * (anchor - zero_anchor), 'anchor changes gradient by ridge shift only')
    inversion_y = np.tile(np.arange(10), 10)
    cases = [('imbalance', z, y), ('single_class', rng.normal(size=(23, 10)) * 20, np.full(23, 3)),
        ('one_row', np.full((1, 10), 1000.), np.array([9])),
        ('balanced_zero', np.zeros((50, 10)), np.tile(np.arange(10), 5)),
        ('active_bound', -10 * np.eye(10)[inversion_y], inversion_y)]
    result = []
    for name, case_z, case_y in cases:
        removed_state, removed_info = authored.fit(case_z, case_y, np.zeros(10))
        frozen_state, frozen_info = solver.base.fit(case_z, case_y, 'bcts')
        check(np.array_equal(removed_state, frozen_state)
              and removed_info['parameters_float64'] == frozen_info['parameters_float64']
              and removed_info['trace'] == frozen_info['trace'], 'anchor deletion exactly equals frozen BCTS: ' + name)
        bias_anchor = np.log(np.bincount(case_y, minlength=10) + 1.)
        bias_anchor -= bias_anchor.mean()
        state, info = authored.fit(case_z, case_y, bias_anchor)
        theta = np.asarray(info['parameters_float64'])
        full_anchor = np.r_[1., bias_anchor]
        value, gradient, hessian = objective(case_z, case_y, theta, full_anchor)
        close(value, info['final_objective'], 'synthetic author optimum TRAIN objective', 2e-10)
        close(kkt(theta, gradient), info['projected_gradient_inf'], 'synthetic bound KKT direct reconstruction', 2e-9)
        check(state.dtype == np.float32 and state.shape == (11,) and state.nbytes == 44 and state[0] >= .001,
              'synthetic BCTS state is one positive scale and ten biases,44B')
        if name == 'active_bound':
            independent, diagnostic = refit(case_z, case_y, full_anchor)
            check(abs(theta[0] - .001) < 1e-14 and abs(independent[0] - .001) < 1e-14,
                  'authored and independent solver both reach positive scalar boundary')
            check(diagnostic['kkt_inf'] < 1e-8, 'independent active-bound KKT checked')
        result.append({'name': name, 'anchor_deletion_state_trace_exact': True,
                       'scale': float(theta[0]), 'kkt_inf': kkt(theta, gradient)})
    check(solver.base.fit.__globals__['Problem'] is original_factory, 'original frozen optimizer globals never patched')
    return {'gradient_fd_max_error': grad_error, 'hessian_fd_max_error': hess_error,
            'anchor_deletion_cases': result, 'author_import_scope': 'synthetic arrays only'}


def main(task):
    began = time.perf_counter()
    independent = load('_independent_bcts_frozen_metrics', PROTOCOL / 'check_next.py')
    seed = independent.PROFILES[task]['seeds'][0]
    independent.configure(task, seed)
    partition, rows = independent.PARTITION, independent.ROWS
    out = PROTOCOL / f'independent_bcts_{task}_s{seed}'
    bcts_root = PROTOCOL / f'bcts_{task}_p{partition}_s{seed}'
    for name, expected in independent.FIRST_SNAPSHOT.items():
        check(protect(PROTOCOL / 'independent_cifar_s20261231' / name) == expected, 'first independent CIFAR files unchanged')
    fixture = read_json(VECTOR / 'BCTS_CONTROL_FIXTURES.json')
    for path, expected in fixture['source_identities_unchanged'].items():
        check(protect(path) == expected, 'BCTS source identity matches pre-real-array fixture')
    for path in (PROTOCOL / 'check_next.py', Path(__file__), VECTOR / 'bcts_control_fixture.py'):
        protect(path)
    check(protect(VECTOR / 'BCTS_CONTROL_PLAN.md') == '9ea93ea8496f33f4f7330a1ea7a3764e3aa101432c04c0bb916903370116d1cc',
          'mechanism addendum fixed identity')
    lock = read_json(PROTOCOL / 'ASSESSMENT_LOCK.json')
    registration = read_json(PROTOCOL / 'PREREGISTRATION.json')
    for relative, expected in {**lock['sources'], **registration['frozen_sources']}.items():
        check(protect(ROOT / relative) == expected, 'main protocol/registration/assess sources unchanged')
    original = read_json(independent.EVALUATION / 'result.json')
    vector = read_json(independent.ADAPTATION / 'result.json')
    assessment = read_json(independent.ASSESSMENT / 'result.json')
    control = read_json(bcts_root / 'result.json')
    check(all(report['status'] == 'DONE' for report in (original, vector, assessment, control)), 'all paired source/control results complete')
    check((original['training_seed'], original['partition_seed'], original['training_rounds']) == (seed, partition, 20),
          'first paired seed/partition/final checkpoint correct')
    check(Path(control['input_root']).resolve() == independent.EVALUATION.resolve()
          and Path(control['vector_root']).resolve() == independent.ADAPTATION.resolve()
          and control['protocol_plan_sha256'] == independent.PLAN_SHA,
          'BCTS binds exact first-seed evaluation/vector states/prospective PLAN')
    bound_count = 0
    for mapping in (control['input_sha256_before_after_equal'], control['source_identities'],
                    vector['input_sha256_before_after_equal'], vector['source_identities'], assessment['source_identities']):
        for path, expected in mapping.items():
            check(protect(path) == expected, 'all recorded BCTS/vector/source hashes match current files')
            bound_count += 1
    check('decision' not in control and 'gates' not in control
          and control['totals']['new_bcts_fits'] == 60 and control['totals']['new_vector_fits'] == 0,
          'mechanism control adds60 BCTS fits with no vector refit/candidate/gate')
    check(original['official_test_byte_reads'] is False and original['official_test_inference'] is False
          and control['forbidden_dataset_open_attempts'] == [] and control['new_dataset_bytes'] == 0,
          'authors report saved-array-only BCTS and no TEST inference')
    toy_result = toys()
    independent.VARIANTS = (*independent.VARIANTS, 'prioranchored_bcts')
    summaries, saved, optimizer_rows = {}, {}, []
    for method in independent.METHODS:
        clients = []
        check(len(control['results'][method]['clients']) == 20, 'all20 BCTS clients present')
        for client in range(20):
            raw = arrays(independent.EVALUATION / f'{method}_client{client}.npz')
            paired = arrays(independent.ADAPTATION / f'{method}_client{client}.npz')
            state_file = arrays(bcts_root / f'{method}_client{client}.npz')
            report = control['results'][method]['clients'][client]
            check(report['client'] == client, 'BCTS report preserves client original position')
            for key in ('train_ids', 'holdout_ids'):
                check(np.array_equal(state_file[key], raw[key]) and np.array_equal(state_file[key], paired[key]),
                      'all BCTS/vector/source original IDs exactly equal')
            for key in ('prioranchored_vector', 'zero_anchor_vector', 'prioranchor_bias'):
                check(np.array_equal(state_file[key], paired[key]), 'cached vector/ten-bias state reused byte-for-byte')
            counts = np.bincount(raw['train_labels'], minlength=10)
            bias_anchor = np.log(counts + 1.)
            bias_anchor -= bias_anchor.mean()
            full_anchor = np.r_[1., bias_anchor]
            close(state_file['train_bias_anchor'], bias_anchor, 'BCTS prior anchor derived only from TRAIN', 0.)
            diagnostic = report['optimization']
            close(diagnostic['anchor'], full_anchor, '11D optimization anchor correct', 2e-15)
            state = state_file['prioranchored_bcts']
            parameters = np.asarray(diagnostic['parameters_float64'])
            check(state.dtype == np.float32 and state.shape == (11,) and state.nbytes == 44
                  and state[0] >= .001 and report['incremental_state_bytes'] == 44,
                  'every deployed BCTS is11D/44B with one common positive scalar')
            check(np.array_equal(state, parameters.astype(np.float32)), 'all BCTS states exact FP32 rounding of optimizer theta')
            check(diagnostic['dimension'] == 11 and diagnostic['regularization'] == 10
                  and diagnostic['regularization_scale'] == 10 and diagnostic['regularization_bias'] == 10,
                  'BCTS matches frozen ridge10 coefficient and11D problem')
            check(abs(parameters[1:].sum()) < 2e-13 and parameters[0] >= .001 - 1e-14,
                  'FP64 optimized bias centered and scalar bound satisfied')
            value, gradient, hessian = objective(raw['train_logits'].astype(float), raw['train_labels'], parameters, full_anchor)
            close(value, diagnostic['final_objective'], 'all60 TRAIN objective independent scalar formula', 2e-9)
            projected = kkt(parameters, gradient)
            close(projected, diagnostic['projected_gradient_inf'], 'all60 KKT direct reconstruction', 3e-9)
            check(np.linalg.eigvalsh(hessian).min() >= 10 - 1e-8, 'all60 Hessians positive definite ridge10')
            check(diagnostic['accepted_steps'] <= 50
                  and all(b['objective'] <= a['objective'] for a, b in zip(diagnostic['trace'], diagnostic['trace'][1:])),
                  'all60 fixed schedule max50 and monotone accepted objective')
            optimizer_rows.append({'method': method, 'client': client, 'scale': float(parameters[0]),
                'projected_gradient_inf': projected, 'stop_reason': diagnostic['stop_reason']})
            z = raw['raw'].astype(float)
            learned = raw['learned'].astype(float)
            scores = {'raw': z, 'analytic_prior': z + raw['prior_shift'],
                'prioranchor_bias': z + raw['fitted_bias'].astype(float),
                'learned_head': learned, 'learned_head_plus_prior': learned + raw['prior_shift'],
                'prioranchored_bcts': z * float(state[0]) + state[1:].astype(float)}
            close(scores['prioranchored_bcts'], state_file['prioranchored_bcts_holdout_logits'],
                  'every BCTS holdout score exact deployed-state application', 0.)
            for kind in ('zero_anchor_vector', 'prioranchored_vector'):
                theta = state_file[kind].astype(float)
                scores[kind] = z * theta[:10] + theta[10:]
            measurements = {name: independent.metrics(score, raw['labels']) for name, score in scores.items()}
            for kind in independent.VARIANTS:
                current, old = measurements[kind], report['metrics'][kind]
                for metric in ('accuracy', 'nll', 'brier', 'ece15'):
                    close(current[metric], old[metric], 'all480 client-variant ' + metric)
                    if kind != 'prioranchored_bcts':
                        close(current[metric], vector['results'][method]['clients'][client]['metrics'][kind][metric],
                              'all seven original baselines unchanged ' + metric)
                for actual, expected in zip(current['bins'], old['ece15_bins']):
                    check(actual['count'] == expected['count'] and actual['correct_sum'] == expected['correct_sum'],
                          'all480 client-variant ECE bins/counts/correct exact')
                    close(actual['confidence_sum'], expected['confidence_sum'], 'all480 ECE bin confidence sum', 2e-10)
            clients.append({'client': client, 'holdout_samples': len(raw['labels']), 'metrics': measurements})
            saved[(method, client)] = raw
        summaries[method] = independent.summarize(clients)
        for kind, summary in summaries[method].items():
            for metric, value in summary.items():
                close(value, control['results'][method]['summaries'][kind][metric], 'all BCTS/control summaries ' + metric)
    refits = []
    for client in (0, 7, 19):
        data = saved[('fedlc_uniform3', client)]
        bias_anchor = np.log(np.bincount(data['train_labels'], minlength=10) + 1.)
        bias_anchor -= bias_anchor.mean()
        theta, info = refit(data['train_logits'].astype(float), data['train_labels'], np.r_[1., bias_anchor])
        old = control['results']['fedlc_uniform3']['clients'][client]['optimization']
        parameter_error = float(np.max(np.abs(theta - old['parameters_float64'])))
        objective_gap = abs(info['objective'] - old['final_objective'])
        check(parameter_error < 2e-6 and objective_gap < 2e-8 and info['kkt_inf'] < 2e-6,
              'preselected0/7/19 independent reduced-BCTS optimum agrees')
        refits.append({'client': client, 'train_rows': len(data['train_labels']), 'parameter_max_error': parameter_error,
                       'objective_gap': objective_gap, **info})
    differences, gates, thresholds = independent.independent_gates(summaries)
    check(gates == assessment['first_seed_conditions'], 'original10 stage gates exactly unchanged by BCTS control')
    for name, values in differences.items():
        for key, value in values.items():
            close(value, assessment['differences'][name][key], 'original stage-gate difference unchanged')
    comparisons = {}
    for method, result in summaries.items():
        scalar, vector_summary = result['prioranchored_bcts'], result['prioranchored_vector']
        comparisons[method] = {'bcts_minus_vector_accuracy_pp': 100 * (scalar['macro_accuracy'] - vector_summary['macro_accuracy']),
            **{'bcts_minus_vector_' + metric: scalar['macro_' + metric] - vector_summary['macro_' + metric]
               for metric in ('nll', 'brier', 'ece15')}}
        for key, value in comparisons[method].items():
            close(value, control['comparisons'][method][key], 'mechanism comparisons independently recomputed')
    for path, expected in PROTECTED.items():
        check(sha(path) == expected, 'all BCTS source/input/main-result/assessment hashes unchanged')
    check(not RAW_ATTEMPTS, 'zero raw/TEST open attempts')
    stops = dict(Counter(row['stop_reason'] for row in optimizer_rows))
    maximum_kkt = max(row['projected_gradient_inf'] for row in optimizer_rows)
    tolerance_count = sum(row['projected_gradient_inf'] <= 1e-8 for row in optimizer_rows)
    selected = comparisons['fedlc_uniform3']
    report = {'status': 'PASS', 'task': task, 'seed': seed, 'partition_seed': partition,
        'checks_passed': len(CHECKS), 'client_variant_metrics_recomputed': 480,
        'bcts_states_checked': 60, 'frozen_vector_states_reused_without_refit': 120,
        'toy_independent_math_and_deletion_equivalence': toy_result, 'all_summaries': summaries,
        'maximum_errors': ERRORS, 'independent_real_TRAIN_refits': refits,
        'optimization_summary': {'states_at_1e8_tolerance': tolerance_count, 'states': 60,
            'max_projected_gradient_inf': maximum_kkt, 'stop_reasons': stops, 'per_state': optimizer_rows},
        'comparisons': comparisons, 'unchanged_stage_conditions': gates,
        'unchanged_main_decision': assessment['decision'], 'no_new_gate_or_candidate': True,
        'input_binding_checks': bound_count, 'protected_files_unchanged': len(PROTECTED),
        'source_input_result_sha256_before_after_equal': PROTECTED,
        'interpretation': 'single-scale BCTS accounts for much of performance;44B vs80B. One vs ten scale dimensions and total ridge penalties differ, so class-specific-scale necessity or causal attribution is not established.',
        'raw_dataset_or_TEST_open_attempts': [], 'new_cnn_forwards': 0, 'new_cnn_backwards': 0,
        'new_ML_training_runs': 0, 'author_code_usage': 'small synthetic equivalence only; real-state objective/metrics/refits independent NumPy',
        'external_cross_model_review': 'REVIEW_UNAVAILABLE', 'verification': 'LOCAL_INDEPENDENT_NUMPY_CHECK_ONLY',
        'seconds': time.perf_counter() - began, 'script_sha256': sha(__file__)}
    out.mkdir(parents=True, exist_ok=True)
    (out / 'RESULT.json').write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    text = f'''# {task} 首seed BCTS独立核查：{len(CHECKS)} PASS

机制control的11维a*z+b使用共同正尺度a≥.001、中心化TRAIN log(count+1)锚、ridge10；与原BCTS相同bound/KKT/Newton/Armijo调度。独立直接11维梯度/Hessian经有限差分，误差{toy_result['gradient_fd_max_error']:.3g}/{toy_result['hessian_fd_max_error']:.3g}；凸性通过。5类CPU toy删除anchor时部署态、FP64参数、trace与冻结BCTS逐位一致，含active-bound KKT，原solver globals未改。

全部60个真实BCTS态核TRAIN目标/梯度/Hessian/KKT、正尺度、44B FP32舍入和FP64应用；只读TRAIN logits/GT生成锚，cached20维状态逐字节复用、0新vectorfit。KKT≤1e−8为{tolerance_count}/60，最大{maximum_kkt:.3g}；stop={stops}，float64_resolution不等于全都超阈值。预定client0/7/19另用10维对比基独立优化，参数max差{max(item['parameter_max_error'] for item in refits):.3g}、目标差{max(item['objective_gap'] for item in refits):.3g}。

3模型×20clients×8变体=480组acc/NLL/Brier/ECE及宏/weighted/pooled ECE/q10重算一致。FedLC-uniform3下，20维相对11维accuracy {(-selected['bcts_minus_vector_accuracy_pp']):+.8f}pp、NLL低{selected['bcts_minus_vector_nll']:.8f}、Brier低{selected['bcts_minus_vector_brier']:.8f}；11维减20维宏ECE={selected['bcts_minus_vector_ece15']:+.8f}。完整三模型有利/不利差值见JSON；单尺度已解释大部分收益，不能称class A必要或大幅提升。

两者state44B/80B、scale维数1/10，总罚项/自由度不同；同ridge系数不等于统计复杂度相同。原主10gates和decisions未变，没有新门槛/选method。{len(PROTECTED)}保护文件不变、{bound_count}绑定匹配。原图身份继承，不读原图/TEST、不CNN/训练；仅独立CPU数值，墙钟非独占速度证据。REVIEW_UNAVAILABLE。
'''
    if len(text) > 1500:
        raise AssertionError('report exceeds1500 characters')
    (out / 'REPORT.md').write_text(text)
    print(json.dumps({'status': 'PASS', 'task': task, 'checks_passed': len(CHECKS),
        'protected_files': len(PROTECTED), 'bindings': bound_count, 'max_kkt': maximum_kkt,
        'states_at_tolerance': tolerance_count, 'selected_comparison': selected,
        'refit_max_parameter_error': max(item['parameter_max_error'] for item in refits),
        'report_characters': len(text)}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--task', choices=('cifar', 'fashion'), required=True)
    main(parser.parse_args().task)
