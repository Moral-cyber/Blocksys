"""Parameterized saved-array audit; numerical functions exactly frozen from CIFAR seed1."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / 'work/predictive_pfl_confirm'
PROFILES = {
    'cifar': {'partition': 20261217, 'seeds': (20261231, 20261232, 20261233),
        'rows': 50000, 'task_name': 'CIFAR10',
        'labels_sha256': 'f5cfe00b0f00968c0cc5ff3b1d2de51b10e33efa277c2986f4e0fa63e58c9f4f'},
    'fashion': {'partition': 20261219, 'seeds': (20261241, 20261242, 20261243),
        'rows': 60000, 'task_name': 'FashionMNIST',
        'labels_sha256': 'e3245b63f7c40d1c8b652835f19744d970c1613b40ddac6bcfc87b9c46e16a0b'}}
FIRST_SNAPSHOT = {
    'check.py': '78a22ea9bfe29e7ec7d4a4bbfda4a976ea352fcb144da70190ba4dd04615b291',
    'REPORT.md': '1b7ae2173cbf13b314710c6ebca11d8d1600b74bf837e568811818b248f7274d',
    'RESULT.json': 'a0028d199dd588f567642a49046b216d85241b358df8b03c8ab93ea0f5d48b17'}
MATH_NAMES = ('metrics', 'summarize', 'objective', 'refit', 'independent_gates')


def configure(task, seed):
    global TASK, SEED, PROFILE, PARTITION, ROWS, HERE, EVALUATION, ADAPTATION, ASSESSMENT, TRAINING
    if task not in PROFILES or seed not in PROFILES[task]['seeds']:
        raise ValueError('only preregistered task/seed combinations are allowed')
    TASK, SEED, PROFILE = task, seed, PROFILES[task]
    PARTITION, ROWS = PROFILE['partition'], PROFILE['rows']
    suffix = f'{TASK}_p{PARTITION}_s{SEED}'
    HERE = PROTOCOL / f'independent_{TASK}_s{SEED}'
    EVALUATION = PROTOCOL / ('evaluation_' + suffix)
    ADAPTATION = PROTOCOL / ('adaptation_' + suffix)
    ASSESSMENT = PROTOCOL / ('assessment_' + suffix)
    TRAINING = PROTOCOL / ('development_' + suffix)


METHODS = ('ce_uniform3', 'fedlc_uniform3', 'fedlc_dense')
VARIANTS = ('raw', 'analytic_prior', 'prioranchor_bias', 'zero_anchor_vector',
            'prioranchored_vector', 'learned_head', 'learned_head_plus_prior')
PLAN_SHA = '703d5956c9edd24d5a03501a75f678a43222d19bd3dd1ede515ed77545b0830b'
ASSESS_SHA = 'bdb0bac2d4809d100e3112e1516e14d8f295c8dbe77fa4da34ce0543706b7b17'
RAW_ATTEMPTS = []


def guard(event, arguments):
    if event == 'open' and isinstance(arguments[0], (str, bytes)):
        path = str(arguments[0])
        if '/dataset/' in path or '/datasets/' in path or '/rawdata/' in path or 'test_batch' in path or 't10k' in path:
            RAW_ATTEMPTS.append(path)
            raise RuntimeError('raw dataset/TEST bytes forbidden in independent saved-array audit')


sys.addaudithook(guard)
import numpy as np

CHECKS = []
ERRORS = {}
PROTECTED = {}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def protect(path):
    path = Path(path).resolve()
    digest = sha(path)
    if str(path) in PROTECTED and PROTECTED[str(path)] != digest:
        raise AssertionError('protected source/input changed: ' + str(path))
    PROTECTED[str(path)] = digest
    return digest


def load_json(path):
    protect(path)
    return json.loads(Path(path).read_text())


def load_arrays(path):
    protect(path)
    with np.load(path, allow_pickle=False) as handle:
        return {name: handle[name].copy() for name in handle.files}


def check(condition, name):
    if not bool(condition):
        raise AssertionError(name)
    CHECKS.append(name)


def close(actual, expected, key, tolerance=2e-12):
    difference = float(np.max(np.abs(np.asarray(actual, dtype=float) - np.asarray(expected, dtype=float))))
    ERRORS[key] = max(ERRORS.get(key, 0.), difference)
    check(difference <= tolerance, key)


def metrics(logits, labels):
    """Different algebra from author: logaddexp, expanded Brier, per-bin masks."""
    z = np.asarray(logits, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    normalizer = np.logaddexp.reduce(z, axis=1)
    probability = np.exp(z - normalizer[:, None])
    truth_probability = probability[np.arange(len(y)), y]
    correct = z.argmax(axis=1) == y
    confidence = probability.max(axis=1)
    bins = []
    for k in range(15):
        member = (confidence >= k / 15) & ((confidence < (k + 1) / 15) if k < 14 else (confidence <= 1 + 2e-14))
        bins.append({'count': int(member.sum()), 'confidence_sum': float(confidence[member].sum()),
                     'correct_sum': int(correct[member].sum())})
    ece = sum(abs(item['correct_sum'] - item['confidence_sum']) for item in bins) / len(y)
    check(sum(item['count'] for item in bins) == len(y), 'independent confidence bins cover every row')
    return {'accuracy': float(correct.mean()),
        'nll': float((normalizer - z[np.arange(len(y)), y]).mean()),
        'brier': float((np.sum(probability * probability, axis=1) - 2 * truth_probability + 1).mean()),
        'ece15': float(ece), 'samples': len(y), 'bins': bins}


def summarize(clients):
    result = {}
    counts = np.asarray([row['holdout_samples'] for row in clients], dtype=float)
    for name in VARIANTS:
        output = {}
        for metric in ('accuracy', 'nll', 'brier', 'ece15'):
            values = np.asarray([row['metrics'][name][metric] for row in clients])
            output['macro_' + metric] = float(np.sum(values) / len(values))
            output['weighted_' + metric] = float(np.sum(values * counts) / np.sum(counts))
        output['macro_loss'], output['weighted_loss'] = output['macro_nll'], output['weighted_nll']
        acc = sorted(row['metrics'][name]['accuracy'] for row in clients)
        index = (len(acc) - 1) * .1
        low = int(np.floor(index))
        output['client_accuracy_q10'] = acc[low] + (index - low) * (acc[low + 1] - acc[low])
        correct = np.sum([[item['correct_sum'] for item in row['metrics'][name]['bins']] for row in clients], axis=0)
        confidence = np.sum([[item['confidence_sum'] for item in row['metrics'][name]['bins']] for row in clients], axis=0)
        output['pooled_ece15'] = float(np.abs(correct - confidence).sum() / counts.sum())
        result[name] = output
    return result


def objective(z, y, parameters, anchor, derivatives=True):
    """Direct diagonal-affine derivatives; no author's n*10*20 design tensor."""
    a, b = parameters[:10], parameters[10:]
    score = z * a + b
    lse = np.logaddexp.reduce(score, axis=1)
    p = np.exp(score - lse[:, None])
    diff = parameters - anchor
    value = float(np.sum(lse - score[np.arange(len(y)), y]) + 5 * (diff @ diff))
    if not derivatives:
        return value
    error = p.copy()
    error[np.arange(len(y)), y] -= 1
    gradient = np.r_[np.sum(error * z, axis=0), np.sum(error, axis=0)] + 10 * diff
    pz = p * z
    haa = np.diag(np.sum(p * z * z, axis=0)) - pz.T @ pz + 10 * np.eye(10)
    hab = np.diag(np.sum(pz, axis=0)) - pz.T @ p
    hbb = np.diag(np.sum(p, axis=0)) - p.T @ p + 10 * np.eye(10)
    hessian = np.block([[haa, hab], [hab.T, hbb]])
    return value, gradient, (hessian + hessian.T) / 2


def refit(z, y, anchor):
    """Independent 19D centered-bias Newton fit, fixed ridge10, TRAIN only."""
    contrast = np.eye(10)[:, :9] - np.eye(10)[:, [9]]
    q, _ = np.linalg.qr(contrast)
    basis = np.zeros((20, 19))
    basis[:10, :10] = np.eye(10)
    basis[10:, 10:] = q
    reduced = basis.T @ anchor
    trace = []
    for step in range(100):
        full = basis @ reduced
        value, gradient, hessian = objective(z, y, full, anchor)
        g = basis.T @ gradient
        trace.append({'step': step, 'objective': value, 'gradient_inf': float(np.abs(g).max())})
        if np.abs(g).max() <= 5e-10:
            break
        h = basis.T @ hessian @ basis
        factor = np.linalg.cholesky(h)
        direction = np.linalg.solve(factor.T, np.linalg.solve(factor, -g))
        step_size = 1.
        for _ in range(60):
            proposal = reduced + step_size * direction
            proposed_value = objective(z, y, basis @ proposal, anchor, False)
            if proposed_value <= value + 1e-4 * step_size * (g @ direction):
                break
            step_size /= 2
        else:
            break
        reduced = proposal
        if np.linalg.norm(step_size * direction, ord=np.inf) < 2e-13:
            break
    full = basis @ reduced
    value, gradient, _ = objective(z, y, full, anchor)
    return full, {'objective': value, 'reduced_gradient_inf': float(np.abs(basis.T @ gradient).max()),
                  'steps': len(trace) - 1, 'trace': trace}


def independent_gates(summaries):
    candidate = summaries['fedlc_uniform3']['prioranchored_vector']
    comparisons = [('zero_vector', 'fedlc_uniform3', 'zero_anchor_vector', .01, -.5),
        ('prior', 'fedlc_uniform3', 'analytic_prior', .01, -.2),
        ('ten_bias', 'fedlc_uniform3', 'prioranchor_bias', -.005, -.5),
        ('ce_vector', 'ce_uniform3', 'prioranchored_vector', -.01, -.5),
        ('dense_vector', 'fedlc_dense', 'prioranchored_vector', -.01, -.5)]
    names = [('zero_vector_nll_reduction_at_least_001', 'zero_vector_accuracy_within_half_pp'),
        ('prior_nll_reduction_at_least_001', 'prior_accuracy_within_02pp'),
        ('ten_bias_nll_within_0005', 'ten_bias_accuracy_within_half_pp'),
        ('ce_vector_nll_within_001', 'ce_vector_accuracy_within_half_pp'),
        ('dense_vector_nll_within_001', 'dense_vector_accuracy_within_half_pp')]
    differences, gates, thresholds = {}, {}, {}
    for (name, method, variant, minimum_nll, minimum_accuracy), (nll_name, acc_name) in zip(comparisons, names):
        reference = summaries[method][variant]
        reduction = reference['macro_nll'] - candidate['macro_nll']
        accuracy_pp = 100 * (candidate['macro_accuracy'] - reference['macro_accuracy'])
        differences[name] = {'nll_reduction': reduction, 'delta_accuracy_pp': accuracy_pp}
        gates[nll_name], gates[acc_name] = bool(reduction >= minimum_nll), bool(accuracy_pp >= minimum_accuracy)
        thresholds[name] = {'reference_method': method, 'reference_variant': variant,
            'minimum_reference_minus_candidate_nll': minimum_nll,
            'minimum_candidate_minus_reference_accuracy_pp': minimum_accuracy}
    return differences, gates, thresholds


def main():
    began = time.perf_counter()
    HERE.mkdir(parents=True, exist_ok=True)
    first_root = PROTOCOL / 'independent_cifar_s20261231'
    for name, expected in FIRST_SNAPSHOT.items():
        check(protect(first_root / name) == expected, 'first CIFAR audit source/report/result unchanged')
    def function_texts(path):
        text = Path(path).read_text()
        tree = ast.parse(text)
        return {node.name: ast.get_source_segment(text, node) for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name in MATH_NAMES}
    first_functions, current_functions = function_texts(first_root / 'check.py'), function_texts(__file__)
    math_hashes = {}
    for name in MATH_NAMES:
        check(current_functions[name] == first_functions[name], 'numerical/gate function source exactly unchanged: ' + name)
        math_hashes[name] = hashlib.sha256(current_functions[name].encode()).hexdigest()
    plan = load_json(PROTOCOL / 'PREREGISTRATION.json')
    lock = load_json(PROTOCOL / 'ASSESSMENT_LOCK.json')
    check(protect(PROTOCOL / 'PLAN.md') == PLAN_SHA, 'prospective PLAN hash matches preregistered protocol')
    check(protect(PROTOCOL / 'assess.py') == ASSESS_SHA, 'assessment code hash matches first-evaluation lock')
    for relative, expected in {**plan['frozen_sources'], **lock['sources']}.items():
        check(protect(ROOT / relative) == expected, 'preregistered/assessment-locked source identity')
    for relative in ('work/prior_vector/evaluate_saved.py', 'work/calibration_baselines/solver.py',
                     'work/calibration_baselines/probe.py', 'work/prior_vector/PLAN.md',
                     'work/prior_bias/solver.py', 'work/calibrated_lowbit_confirm/evaluate.py'):
        protect(ROOT / relative)
    old_paths = [ROOT / 'work/prior_bias/diagnostic_v1/result.json', ROOT / 'work/prior_vector/diagnostic_v1/result.json']
    old_decisions = {str(path): load_json(path)['decision'] for path in old_paths}
    check(set(old_decisions.values()) == {'STOP'}, 'both historical accuracy decisions remain STOP')
    evaluation = load_json(EVALUATION / 'result.json')
    adapted = load_json(ADAPTATION / 'result.json')
    assessment = load_json(ASSESSMENT / 'result.json')
    for report in (evaluation, adapted, assessment):
        check(report['status'] == 'DONE', 'all author results complete before independent checks')
    check((evaluation['training_seed'], evaluation['partition_seed'], evaluation['training_rounds'])
          == (SEED, PARTITION, 20), 'fresh paired task/seed/partition/final-round identity')
    check(set(evaluation['results']) == set(METHODS) and set(adapted['results']) == set(METHODS),
          'all three prescribed methods present')
    check(evaluation['official_test_byte_reads'] is False and evaluation['official_test_inference'] is False,
          'author evaluation explicitly forbids official TEST')
    check(adapted['protocol_plan_sha256'] == PLAN_SHA
          and Path(adapted['protocol_registration']).resolve() == (PROTOCOL / 'PREREGISTRATION.json').resolve()
          and Path(adapted['input_root']).resolve() == EVALUATION.resolve(),
          'adaptation binds prospective protocol and exact evaluation root')
    input_binding_checks = 0
    for mapping in (adapted['input_sha256_before_after_equal'], adapted['source_identities'],
                    assessment['source_identities'], evaluation['source_identities'], evaluation['frozen_model_inputs']):
        for path, expected in mapping.items():
            check(protect(path) == expected, 'all source/model/saved-array hashes match author binding')
            input_binding_checks += 1
    check(adapted['forbidden_dataset_open_attempts'] == [] and adapted['new_dataset_bytes'] == 0,
          'adaptation saved-array-only guard reports zero forbidden reads')
    check(assessment['original_evaluation_accuracy_decision_unchanged'] == evaluation['decision'],
          'new assessment preserves original new-split ten-dimensional decision')
    check(assessment['external_cross_model_review'] == 'REVIEW_UNAVAILABLE', 'external review remains unavailable')

    configs = {}
    manifest = None
    for method in METHODS:
        config = load_json(TRAINING / method / 'config.json')
        check(config['signature'] == canonical({k: v for k, v in config.items() if k != 'signature'}),
              'training configuration signature rederived independently')
        check((config['method'], config['seed'], config['partition_seed'], config['rounds'])
              == (method, SEED, PARTITION, 20), 'training method/seed/partition/rounds match')
        check(config['official_test_byte_read'] is False and config['official_test_inference'] is False,
              'training TEST guard metadata false')
        result = load_json(TRAINING / method / 'result.json')
        check(result['status'] == 'DONE' and len(result['rows']) == 20
              and [row['round'] for row in result['rows']] == list(range(1, 21)),
              'training completed exact final20 rounds')
        check(result['signature'] == config['signature'], 'training result binds signed config')
        if manifest is None:
            manifest = config['data_manifest']
        check(config['data_manifest'] == manifest and config['data_identities'] == evaluation['data_identities'],
              'all training methods share exact partition and recorded TRAIN input identities')
        configs[method] = config
    check(manifest['test_used_for_selection'] is False, 'partition declares no official TEST selection')
    check(manifest['source_train_samples'] == ROWS
          and manifest['official_train_labels_sha256'] == PROFILE['labels_sha256'],
          'task-specific TRAIN row count and known label SHA match signed source manifest')
    if TASK == 'fashion':
        check(manifest['task'] == 'FashionMNIST' and len(manifest['source_gzip_files']) == 2,
              'Fashion uses60000-row TRAIN namespace and two strict TRAIN gzip identities')
        check({Path(path).name for path in evaluation['data_identities']} ==
              {'train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz'},
              'Fashion recorded data identities contain only two official TRAIN gzip files')
    saved_labels = np.full(ROWS, -1, dtype=np.int64)
    saved_memberships = []
    recomputed, saved_data, optimization_checks = {}, {}, []
    metric_rows = 0
    for method in METHODS:
        clients = []
        for client in range(20):
            original = load_arrays(EVALUATION / f'{method}_client{client}.npz')
            calibration = load_arrays(ADAPTATION / f'{method}_client{client}.npz')
            old = adapted['results'][method]['clients'][client]
            check(old['client'] == client, 'client order agrees with original IDs')
            for name, expected in (('train_ids', manifest['train_indices'][client]),
                                   ('holdout_ids', manifest['holdout_indices'][client])):
                check(np.array_equal(original[name], expected), 'saved membership exactly equals training manifest')
                check(np.array_equal(calibration[name], original[name]), 'adaptation preserves exact TRAIN/holdout ID order')
            check(np.intersect1d(original['train_ids'], original['holdout_ids']).size == 0,
                  'TRAIN and holdout IDs disjoint for every client')
            counts = np.bincount(original['train_labels'], minlength=10)
            check(np.array_equal(counts, manifest['client_label_counts'][client]), 'TRAIN labels reproduce manifest class counts')
            expected_prior = np.log((counts + 1.) / (len(original['train_labels']) + 10)).astype(np.float32)
            check(np.array_equal(expected_prior, original['prior_shift']), 'fixed prior is TRAIN-only exact FP32 log-frequency')
            anchor_bias = np.log(counts + 1.)
            anchor_bias -= anchor_bias.mean()
            close(calibration['train_bias_anchor'], anchor_bias, 'saved anchor derived from TRAIN counts', 0.)
            if method == METHODS[0]:
                for ids_name, labels_name in (('train_ids', 'train_labels'), ('holdout_ids', 'labels')):
                    ids = original[ids_name].astype(np.int64)
                    check(np.all((ids >= 0) & (ids < ROWS)) and np.all(saved_labels[ids] == -1),
                          'each official TRAIN row ID appears once')
                    saved_labels[ids] = original[labels_name]
                    saved_memberships.append(ids)
            else:
                reference = saved_data[(METHODS[0], client)]
                for name in ('train_ids', 'train_labels', 'holdout_ids', 'labels', 'prior_shift'):
                    check(np.array_equal(original[name], reference[name]), 'paired models share saved labels/IDs/prior exactly')
            raw = original['raw'].astype(np.float64)
            learned = original['learned'].astype(np.float64)
            scores = {'raw': raw, 'analytic_prior': raw + expected_prior,
                'prioranchor_bias': raw + original['fitted_bias'].astype(np.float64),
                'learned_head': learned, 'learned_head_plus_prior': learned + expected_prior}
            for variant, state_key, diagnostic_key, expected_bias in (
                ('prioranchored_vector', 'prioranchored_vector', 'prioranchored_vector_optimization', anchor_bias),
                ('zero_anchor_vector', 'zero_anchor_vector', 'zero_anchor_vector_optimization', np.zeros(10))):
                state = calibration[state_key]
                diag = old[diagnostic_key]
                check(state.dtype == np.float32 and state.shape == (20,) and state.nbytes == 80,
                      'both vector controls deploy exactly20 FP32/80B parameters')
                parameters = np.asarray(diag['parameters_float64'], dtype=float)
                expected_anchor = np.r_[np.ones(10), expected_bias]
                close(diag['anchor'], expected_anchor, 'optimizer anchor uses only TRAIN labels', 2e-15)
                check(np.array_equal(state, parameters.astype(np.float32)), 'deployed state is exact FP64-to-FP32 rounding')
                scores[variant] = raw * state[:10].astype(float) + state[10:].astype(float)
                close(calibration[state_key + '_holdout_logits'], scores[variant], 'saved vector holdout scores reapply deployed state', 0.)
                train_z, train_y = original['train_logits'].astype(float), original['train_labels']
                value, gradient, _ = objective(train_z, train_y, parameters, expected_anchor)
                close(value, diag['final_objective'], 'TRAIN objective matches independent direct formula', 2e-9)
                close(float(np.abs(gradient).max()), diag['projected_gradient_inf'], 'TRAIN gradient matches independent direct formula', 2e-9)
                check(abs(parameters[10:].sum()) <= 2e-13, 'optimized FP64 bias is centered')
                check(diag['accepted_steps'] <= 50, 'fixed max50 optimization budget respected')
                optimization_checks.append({'method': method, 'client': client, 'variant': variant,
                    'gradient_inf': float(np.abs(gradient).max()), 'objective': value})
            measurements = {variant: metrics(scores[variant], original['labels']) for variant in VARIANTS}
            for variant, current in measurements.items():
                previous = old['metrics'][variant]
                for metric in ('accuracy', 'nll', 'brier', 'ece15'):
                    close(current[metric], previous[metric], 'all-client ' + metric)
                close(current['nll'], previous['loss'], 'NLL and loss alias')
                check(current['samples'] == previous['samples'], 'all-client metric sample counts match')
                for actual_bin, expected_bin in zip(current['bins'], previous['ece15_bins']):
                    check(actual_bin['count'] == expected_bin['count'] and actual_bin['correct_sum'] == expected_bin['correct_sum'],
                          'all-client ECE bins reproduce counts/correctness')
                    close(actual_bin['confidence_sum'], expected_bin['confidence_sum'], 'ECE confidence sums', 2e-10)
                metric_rows += 1
            original_names = {'raw': 'raw', 'analytic_prior': 'analytic_prior', 'prioranchor_bias': 'prioranchor_bias',
                'learned_head': 'matched_learned_head', 'learned_head_plus_prior': 'matched_learned_head_plus_prior'}
            for key, original_key in original_names.items():
                prior_metrics = evaluation['results'][method]['clients'][client]['metrics'][original_key]
                close(measurements[key]['accuracy'], prior_metrics['accuracy'], 'original saved evaluation accuracy')
                close(measurements[key]['nll'], prior_metrics['loss'], 'original saved evaluation NLL', 2e-12)
            clients.append({'client': client, 'holdout_samples': len(original['labels']), 'metrics': measurements})
            saved_data[(method, client)] = original
        summary = summarize(clients)
        for variant in VARIANTS:
            for key, value in summary[variant].items():
                close(value, adapted['results'][method]['summaries'][variant][key], 'all summaries ' + key)
        recomputed[method] = {'clients': clients, 'summaries': summary}
    check(np.array_equal(np.sort(np.concatenate(saved_memberships)), np.arange(ROWS)),
          'saved TRAIN/holdout IDs cover official TRAIN namespace exactly once')
    check(np.array_equal(np.bincount(saved_labels, minlength=10), np.full(10, ROWS // 10)),
          'reconstructed official TRAIN label counts match task profile')
    check(hashlib.sha256(saved_labels.astype('<i8').tobytes()).hexdigest() == manifest['official_train_labels_sha256'],
          'saved labels in original ID order match frozen official TRAIN label SHA256')

    refits = []
    for client in (0, 7, 19):
        original = saved_data[('fedlc_uniform3', client)]
        anchor = np.log(np.bincount(original['train_labels'], minlength=10) + 1.)
        anchor -= anchor.mean()
        full_anchor = np.r_[np.ones(10), anchor]
        fitted, diagnostics = refit(original['train_logits'].astype(float), original['train_labels'], full_anchor)
        author = adapted['results']['fedlc_uniform3']['clients'][client]['prioranchored_vector_optimization']
        author_params = np.asarray(author['parameters_float64'])
        parameter_error = float(np.abs(fitted - author_params).max())
        objective_gap = abs(diagnostics['objective'] - author['final_objective'])
        check(parameter_error < 2e-6, 'three preselected clients independent19D refit parameters agree')
        check(objective_gap < 2e-8, 'three preselected clients independent19D optimum objective agrees')
        check(diagnostics['reduced_gradient_inf'] < 2e-6, 'three preselected clients independent reduced KKT small')
        refits.append({'client': client, 'method': 'fedlc_uniform3', 'train_rows': len(original['train_labels']),
            'parameter_max_error': parameter_error, 'objective_gap': objective_gap,
            'solver': 'independent NumPy19D contrast-basis Cholesky Newton', **diagnostics})
    summaries = {method: data['summaries'] for method, data in recomputed.items()}
    differences, gates, thresholds = independent_gates(summaries)
    for name, values in differences.items():
        for key, value in values.items():
            close(value, assessment['differences'][name][key], 'prospective gate differences', 2e-12)
    check(gates == assessment['first_seed_conditions'], 'all10 independently recomputed prospective conditions match')
    decision = 'FIRST_SEED_FOLLOW_UP' if all(gates.values()) else 'FIRST_SEED_STOP'
    if SEED != PROFILE['seeds'][0]:
        decision = 'PLANNED_REPETITION_RECORDED; task-level aggregation required'
    check(decision == assessment['decision'], 'independent seed-specific decision equals locked helper')
    check(not RAW_ATTEMPTS, 'independent audit attempted zero raw/TEST file reads')
    for path, expected in PROTECTED.items():
        check(sha(path) == expected, 'all protected author sources/inputs/results unchanged after audit')
    static_limitations = [
        'assess.py checks this exact evaluation result hash, root and protocol identity, but does not itself rehash every saved NPZ or verify ASSESSMENT_LOCK. This independent audit verifies the full current identity chain.',
        'registration is linked by path in assess.py; its actual digest is checked here against ASSESSMENT_LOCK and adaptation input hashes.',
        'underlying image byte identities are inherited recorded metadata; no image file bytes or CNN predictions were independently regenerated.',
        'prior_vector evaluator has seven variants; the full predeclared strong controls and FedRoD/TopK are separate remaining evidence before any final paper claim.',
        'current first-seed continuation does not establish a three-seed task claim or overturn historical accuracy STOPs.']
    report = {'status': 'PASS', 'task': TASK, 'seed': SEED, 'partition_seed': PARTITION, 'numerical_function_sha256': math_hashes, 'first_CIFAR_audit_snapshot': FIRST_SNAPSHOT, 'checks_passed': len(CHECKS), 'metric_client_variant_rows': metric_rows,
        'all_summaries': summaries, 'maximum_errors': ERRORS, 'independent_refits': refits,
        'vector_optimization_states_checked': len(optimization_checks), 'optimization_checks': optimization_checks,
        'decision': decision, 'conditions': gates, 'thresholds_and_units': thresholds, 'differences': differences,
        'old_accuracy_decisions': old_decisions, 'new_split_ten_bias_decision_unchanged': evaluation['decision'],
        'input_binding_hash_checks': input_binding_checks, 'protected_files_unchanged': len(PROTECTED),
        'source_input_result_sha256_before_after_equal': PROTECTED,
        'static_limitations': static_limitations, 'label_and_id_scope': {'task': TASK, 'seed': SEED, 'partition_seed': PARTITION, 'rows': ROWS, 'train_rows': evaluation['training_samples_actual'], 'holdout_rows': evaluation['holdout_samples_actual'], 'labels_sha256': PROFILE['labels_sha256'], 'scope': 'all saved official TRAIN IDs/labels reconstructed; disjoint exact manifest and label SHA'},
        'raw_or_test_open_attempts': [], 'new_cnn_forwards': 0, 'new_cnn_backwards': 0,
        'new_training_runs': 0, 'source_modules_imported': False, 'external_cross_model_review': 'REVIEW_UNAVAILABLE',
        'verification': 'LOCAL_INDEPENDENT_NUMPY_CHECK_ONLY', 'seconds': time.perf_counter() - began,
        'numpy': np.__version__, 'script_sha256': sha(__file__)}
    (HERE / 'RESULT.json').write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    candidate = summaries['fedlc_uniform3']['prioranchored_vector']
    maximum_refit = max(item['parameter_max_error'] for item in refits)
    maximum_gap = max(item['objective_gap'] for item in refits)
    text = f'''# {PROFILE['task_name']} seed{SEED}独立核查：{len(CHECKS)} PASS

新问题为p{PARTITION}/seed{SEED}的宏NLL与accuracy保留约束；旧10维与20维accuracy STOP及首CIFAR核查三文件哈希不变。PLAN、PREREG、assess锁SHA bdb0bac…与来源/结果绑定均核对；本seed决定为{decision}，条件{sum(gates.values())}/10，全部重算一致。首seed依预注册决定后续，重复seed仍须任务级汇总，不能视为最终成功。

未导入作者solver/evaluator：用独立NumPy的logaddexp NLL、展开式Brier、逐bin ECE与原始argmax，重算3模型×20clients×7变体={metric_rows}组；逐client accuracy/NLL/Brier/ECE、15bin计数、宏/加权/pooled ECE与q10全部一致。最大NLL误差{ERRORS['all-client nll']:.3g}，Brier {ERRORS['all-client brier']:.3g}，ECE {ERRORS['all-client ece15']:.3g}。

候选宏accuracy {100*candidate['macro_accuracy']:.8f}%，NLL {candidate['macro_nll']:.10f}，Brier {candidate['macro_brier']:.10f}，ECE {candidate['macro_ece15']:.10f}。对zero-vector：accuracy {differences['zero_vector']['delta_accuracy_pp']:+.8f}pp，NLL降低{differences['zero_vector']['nll_reduction']:.8f}；对fixed prior：{differences['prior']['delta_accuracy_pp']:+.8f}pp、降低{differences['prior']['nll_reduction']:.8f}。五组门槛NLL用绝对自然对数损失，accuracy先乘100再以pp比较，方向/阈值符合PLAN；完整差值见JSON。

120个vector态均核TRAIN目标/梯度、anchor、80B FP32舍入及部署评分。预定client0/7/19另用19维中心化对比基独立优化，参数最大差{maximum_refit:.3g}，目标差{maximum_gap:.3g}。三模型ID/标签严格同序，{evaluation['training_samples_actual']} TRAIN与{evaluation['holdout_samples_actual']} holdout不重叠，重建{ROWS}条TRAIN标签SHA与冻结task profile/manifest相同。{len(PROTECTED)}个作者源/输入/结果前后不变，{input_binding_checks}项历史绑定匹配。

边界：assess自身只绑定evaluation结果hash/路径及protocol，不逐包重验全部NPZ或ASSESSMENT_LOCK；本次额外核全链。原图字节身份继承既有记录，未重做CNN。完整强对照、FedRoD/TopK、三seed任务级汇总及最终TEST仍属后续证据。0原始数据/TEST打开、0CNN、0训练；墙钟有并行竞争不作独占速度证据。REVIEW_UNAVAILABLE；仅本地独立数值核查。
'''
    if len(text) > 1500:
        raise AssertionError('human report must stay within1500 characters')
    (HERE / 'REPORT.md').write_text(text)
    print(json.dumps({'status': 'PASS', 'checks_passed': len(CHECKS), 'decision': decision,
        'metric_rows': metric_rows, 'protected_files': len(PROTECTED),
        'candidate': candidate, 'differences': differences, 'max_refit_parameter_error': maximum_refit,
        'max_refit_objective_gap': maximum_gap, 'report_characters': len(text)}, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--task', choices=PROFILES, required=True)
    parser.add_argument('--seed', type=int, required=True)
    arguments = parser.parse_args()
    configure(arguments.task, arguments.seed)
    main()
