"""Export frozen TRAIN-fitted deployment states before any official TEST access.

Reconstructs only the original matched one-epoch last-Linear fit. Existing
calibrators are copied, never refitted. CLI exports all three methods for one
fixed task/seed. No TEST loader, TEST lock, new FL round, or new seed exists here.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import sys
import time

sys.dont_write_bytecode = True
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
METHODS = ('ce_uniform3', 'fedlc_uniform3', 'fedlc_dense')
TASKS = {'cifar': (20261217, (20261231, 20261232, 20261233)),
         'fashion': (20261219, (20261241, 20261242, 20261243))}
STATE_DIMENSIONS = {'analytic_prior': 10, 'prioranchor_bias': 10, 'zero_anchor_bias': 10,
                    'temperature_scaling': 1, 'bcts': 11, 'zero_anchor_vector': 20,
                    'prioranchored_vector': 20, 'prioranchored_bcts': 11}
TOLERANCE = 1e-6
PINS = {
    'work/calibrated_lowbit_confirm/common.py': '3f2e4693488d7e5a35a182e5340edb9d73f5cafc45b7c0e61b1e2cc392a22d9a',
    'work/calibrated_lowbit_confirm/evaluate.py': 'afcb8a47dc9c303b3e7d558e4e91b50b560cd10dd0287956969d61ab27e13250',
    'work/calibrated_fashion/adapter.py': '624109d019433ed1983bdc810063cdd5fe46116f0896c9dd8636214d553915d2',
    'work/predictive_pfl_confirm/PLAN.md': '703d5956c9edd24d5a03501a75f678a43222d19bd3dd1ede515ed77545b0830b',
    'work/predictive_pfl_confirm/PREREGISTRATION.json': '1eb7d284c43992a3cca58e31448a4b595b88e67abdf167c24a605342bf2f0568'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_sha(value):
    a = np.ascontiguousarray(value)
    return hashlib.sha256(str(a.dtype).encode() + repr(a.shape).encode() + a.tobytes()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def install_no_test_guard():
    attempts = []
    def audit(event, args):
        if event == 'open' and isinstance(args[0], (str, bytes)):
            path = str(args[0])
            if 'test_batch' in path or 't10k' in path:
                attempts.append(path)
                raise ValueError('TRAIN state exporter forbids official TEST byte access')
    sys.addaudithook(audit)
    return attempts


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def runtime(task):
    for rel, expected in PINS.items():
        if sha(ROOT / rel) != expected:
            raise ValueError('frozen source changed: ' + rel)
    if task == 'fashion':
        adapter = load('_train_state_fashion_adapter', ROOT / 'work/calibrated_fashion/adapter.py')
        return adapter.common, adapter.evaluation
    common = load('_train_state_cifar_common', ROOT / 'work/calibrated_lowbit_confirm/common.py')
    previous = sys.modules.get('common')
    sys.modules['common'] = common
    try:
        evaluation = load('_train_state_frozen_evaluation', ROOT / 'work/calibrated_lowbit_confirm/evaluate.py')
    finally:
        if previous is None:
            del sys.modules['common']
        else:
            sys.modules['common'] = previous
    return common, evaluation


def snapshot_report(path, inputs):
    path = Path(path).resolve()
    inputs[str(path)] = sha(path)
    report = json.loads(path.read_text())
    if report.get('status') != 'DONE' or set(report.get('results', {})) != set(METHODS):
        raise ValueError('complete DONE three-method input required: ' + str(path))
    for method in METHODS:
        if [r.get('client') for r in report['results'][method]['clients']] != list(range(20)):
            raise ValueError('all20 ordered clients required: ' + str(path))
    for field in ('source_identities', 'data_identities', 'frozen_model_inputs', 'input_sha256_before_after_equal'):
        for filename, expected in report.get(field, {}).items():
            if sha(filename) != expected:
                raise ValueError('input report provenance changed: ' + filename)
            inputs[filename] = expected
    return report


def arrays(path, inputs=None):
    path = Path(path).resolve()
    if inputs is not None:
        inputs[str(path)] = sha(path)
    with np.load(path, allow_pickle=False) as source:
        return {name: source[name].copy() for name in source.files}


def collect_states(base, vector, calibration, bcts):
    """Exact scorer-compatible state dictionary plus original TRAIN metadata."""
    y = np.asarray(base['train_labels'])
    if y.dtype.kind not in 'iu' or y.ndim != 1 or not len(y) or np.any(y < 0) or np.any(y >= 10):
        raise ValueError('nonempty integer local TRAIN labels required')
    counts = np.bincount(y, minlength=10).astype(np.int64)
    if (base['train_counts'].dtype.kind not in 'iu' or base['train_counts'].shape != (10,)
            or not np.array_equal(counts, base['train_counts'])):
        raise ValueError('saved TRAIN classcounts differ from labels')
    for name in ('train_ids', 'holdout_ids'):
        ids = base[name]
        if ids.dtype.kind not in 'iu' or ids.ndim != 1 or len(np.unique(ids)) != len(ids):
            raise ValueError('unique integer original IDs required')
        for other in (vector, calibration, bcts):
            if not np.array_equal(ids, other[name]):
                raise ValueError('calibration TRAIN/holdout original membership differs')
    if len(base['train_ids']) != len(y) or np.intersect1d(base['train_ids'], base['holdout_ids']).size:
        raise ValueError('TRAIN/holdout membership mismatch or overlap')
    for name, size in (('train_logits', len(y)), ('raw', len(base['holdout_ids'])), ('learned', len(base['holdout_ids']))):
        value = base[name]
        if value.dtype != np.float32 or value.shape != (size, 10) or not np.isfinite(value).all():
            raise ValueError('finite saved FP32 logits required: ' + name)
    states = {'analytic_prior': base['prior_shift'], 'prioranchor_bias': base['fitted_bias'],
              'zero_anchor_bias': calibration['zero_anchor_bias'],
              'temperature_scaling': calibration['temperature_scaling'], 'bcts': calibration['bcts'],
              'zero_anchor_vector': vector['zero_anchor_vector'],
              'prioranchored_vector': vector['prioranchored_vector'],
              'prioranchored_bcts': bcts['prioranchored_bcts']}
    for name, dimension in STATE_DIMENSIONS.items():
        value = states[name]
        if value.dtype != np.float32 or value.shape != (dimension,) or not np.isfinite(value).all():
            raise ValueError('exact finite FP32 calibration state required: ' + name)
        if name in ('temperature_scaling', 'bcts', 'prioranchored_bcts') and value[0] < np.float32(.001):
            raise ValueError('positive scalar below fixed bound')
    expected = np.log((counts + 1.) / (len(y) + 10)).astype(np.float32)
    if not np.array_equal(states['analytic_prior'], expected):
        raise ValueError('analytic prior differs from smoothed TRAIN counts')
    for other in (vector, calibration, bcts):
        if not np.array_equal(other['prioranchor_bias'], states['prioranchor_bias']):
            raise ValueError('saved original prior-anchor bias differs')
    if not np.array_equal(calibration['vector_scaling'], states['zero_anchor_vector']):
        raise ValueError('frozen zero-vector fits from paired roots differ')
    for key in ('zero_anchor_vector', 'prioranchored_vector'):
        if not np.array_equal(bcts[key], states[key]):
            raise ValueError('BCTS cached paired vector state differs')
    return {key: value.copy() for key, value in states.items()}, counts


def rebuild_head(evaluation, model, training, holdout, seed, expected_raw, expected_learned, expected_labels):
    """Execute unchanged matched_head, observe actual forwards/head backwards."""
    counters = {'train_forward_samples': 0, 'train_forward_batches': 0,
                'verification_forward_samples': 0, 'verification_forward_batches': 0,
                'head_backward_calls': 0}
    phase = ['verification']
    def forward_hook(module, args, value):
        key = 'train' if phase[0] == 'train' else 'verification'
        counters[key + '_forward_samples'] += len(args[0])
        counters[key + '_forward_batches'] += 1
    def backward_hook(module, grad_input, grad_output):
        counters['head_backward_calls'] += 1
    handles = [model.register_forward_hook(forward_hook),
               model.classifier[-1].register_full_backward_hook(backward_hook)]
    before = evaluation.model_hash(model)
    started = time.perf_counter()
    try:
        raw, labels, raw_cost = evaluation.forward(model, holdout)
        phase[0] = 'train'
        adapted, original_cost = evaluation.matched_head(model, training, seed)
        phase[0] = 'verification'
        learned, learned_labels, learned_cost = evaluation.forward(adapted, holdout)
    finally:
        for handle in handles:
            handle.remove()
    if evaluation.model_hash(model) != before:
        raise ValueError('matched-head reconstruction changed global model')
    if not np.array_equal(labels, expected_labels) or not np.array_equal(learned_labels, expected_labels):
        raise ValueError('reconstructed holdout original order/GT differs')
    raw_error = float(np.max(np.abs(raw.astype(np.float64) - expected_raw)))
    head_error = float(np.max(np.abs(learned.astype(np.float64) - expected_learned)))
    if raw_error > TOLERANCE or head_error > TOLERANCE:
        raise ValueError(f'holdout reproduction exceeds fixed1e-6: raw={raw_error}, learned={head_error}')
    if any(parameter.grad is not None for parameter in list(adapted.parameters())[:-2]):
        raise ValueError('reconstructed frozen body received gradients')
    if (counters['head_backward_calls'] != original_cost['gradient_evaluations']
            or counters['train_forward_samples'] != len(training)):
        raise ValueError('observed head-training cost differs from original evaluator')
    head = adapted.classifier[-1]
    weight, bias = head.weight.detach().cpu().numpy().copy(), head.bias.detach().cpu().numpy().copy()
    if weight.dtype != np.float32 or bias.dtype != np.float32 or weight.shape != (10, 64) or bias.shape != (10,):
        raise ValueError('original650-parameter FP32 last-Linear state required')
    return weight, bias, {'seed': seed, 'holdout_raw_max_abs_error': raw_error,
                         'holdout_learned_max_abs_error': head_error, 'tolerance': TOLERANCE,
                         'head_state_bytes': weight.nbytes + bias.nbytes,
                         'observed': counters, 'original_matched_head_cost': original_cost,
                         'raw_verification_cost': raw_cost, 'head_verification_cost': learned_cost,
                         'elapsed_seconds': time.perf_counter() - started,
                         'body_backward_calls': 0, 'new_calibration_fits': 0, 'new_FL_rounds': 0,
                         'head_gradient_scalars': 650 * counters['head_backward_calls']}


def run(task, model_root, evaluation_root, adaptation_root, calibration_root, bcts_root, out):
    if task not in TASKS:
        raise ValueError('fixed cifar/fashion task required')
    model_root, evaluation_root, adaptation_root, calibration_root, bcts_root, out = [Path(p).resolve() for p in (
        model_root, evaluation_root, adaptation_root, calibration_root, bcts_root, out)]
    if out.exists():
        raise ValueError('fresh TRAIN-state output required')
    attempts = install_no_test_guard()
    torch.set_num_threads(1)
    common, evaluation = runtime(task)
    inputs = {}
    reports = [snapshot_report(p / 'result.json', inputs) for p in (
        evaluation_root, adaptation_root, calibration_root, bcts_root)]
    base_report, vector_report, calibration_report, bcts_report = reports
    for report in reports[1:]:
        if Path(report.get('input_root', '')).resolve() != evaluation_root:
            raise ValueError('all calibration reports must bind exact base evaluation')
        if report.get('input_sha256_before_after_equal', {}).get(str(evaluation_root / 'result.json')) != inputs[str(evaluation_root / 'result.json')]:
            raise ValueError('calibration report does not bind unchanged source result')
    if Path(bcts_report.get('vector_root', '')).resolve() != adaptation_root:
        raise ValueError('BCTS report must bind exact vector adaptation root')
    sources = common.source_identities()
    sources.update({str(ROOT / rel): expected for rel, expected in PINS.items()})
    sources[str(Path(__file__).resolve())] = sha(__file__)
    configs, models = {}, {}
    for method in METHODS:
        folder = model_root / method
        for name in ('config.json', 'checkpoint.pt', 'result.json'):
            inputs[str(folder / name)] = sha(folder / name)
        config = json.loads((folder / 'config.json').read_text())
        unsigned = {k: v for k, v in config.items() if k != 'signature'}
        if config.get('signature') != common.signature(unsigned):
            raise ValueError('full training source/config signature mismatch')
        if (config['method'] != method or config['partition_seed'] != TASKS[task][0]
                or config['seed'] not in TASKS[task][1] or config['rounds'] != 20
                or config['parameters'] != 136874 or config['batch'] != 64
                or config['lr'] != .03 or config['local_epochs'] != 1):
            raise ValueError('fixed main three-method task/seed/training protocol required')
        if (config['source_identities'] != common.source_identities()
                or config['source_identities'] != base_report['source_identities']
                or config['data_identities'] != base_report['data_identities']):
            raise ValueError('checkpoint source/data differ from original frozen evaluation')
        for name in ('config.json', 'checkpoint.pt'):
            if base_report['frozen_model_inputs'].get(str(folder / name)) != inputs[str(folder / name)]:
                raise ValueError('base evaluation does not bind this final checkpoint/config')
        finished = json.loads((folder / 'result.json').read_text())
        if finished.get('status') != 'DONE' or len(finished.get('rows', [])) != 20:
            raise ValueError('only complete final20 checkpoint may export')
        checkpoint = torch.load(folder / 'checkpoint.pt', map_location='cpu', weights_only=False)
        models[method] = common.validate_checkpoint(checkpoint, config).eval()
        if len(checkpoint['rows']) != 20:
            raise ValueError('incomplete final checkpoint')
        configs[method] = config
    first = configs[METHODS[0]]
    seed = first['seed']
    if (len({c['seed'] for c in configs.values()}) != 1 or base_report['training_seed'] != seed
            or base_report['partition_seed'] != TASKS[task][0] or base_report['training_rounds'] != 20):
        raise ValueError('all model/evaluation task seeds must match')
    manifest = first['data_manifest']
    if (task == 'fashion') != (manifest.get('task') == 'FashionMNIST'):
        raise ValueError('explicit task differs from checkpoint task')
    if any(c['data_manifest'] != manifest for c in configs.values()):
        raise ValueError('paired model memberships differ')
    cached = {}
    for method in METHODS:
        for client in range(20):
            base, vector, calibration, bcts = [arrays(p / f'{method}_client{client}.npz', inputs)
                                                for p in (evaluation_root, adaptation_root, calibration_root, bcts_root)]
            states, counts = collect_states(base, vector, calibration, bcts)
            if (not np.array_equal(base['train_ids'], manifest['train_indices'][client])
                    or not np.array_equal(base['holdout_ids'], manifest['holdout_indices'][client])
                    or not np.array_equal(counts, manifest['client_label_counts'][client])):
                raise ValueError('saved states do not match exact original TRAIN manifest')
            cached[method, client] = base, states, counts
    training, _, actual_manifest, data_ids, images, labels = common.prepare(first['partition_seed'])
    if actual_manifest != manifest or data_ids != first['data_identities']:
        raise ValueError('reconstructed strict TRAIN partition/source differs')
    holdouts = [common.partitions.IndexedCifar10(images, labels, ids) for ids in manifest['holdout_indices']]
    for client in range(20):
        for method in METHODS:
            if not np.array_equal(cached[method, client][0]['train_labels'], labels[manifest['train_indices'][client]]):
                raise ValueError('saved TRAIN GT differs from strict original rows')
    out.mkdir(parents=True)
    atomic_json(out / 'input_snapshot.json', {**sources, **inputs})
    atomic_json(out / 'manifest.json', manifest)
    train_counts = np.asarray(manifest['client_label_counts'], dtype=np.int64)
    np.save(out / 'train_counts.npy', train_counts, allow_pickle=False)
    results, files = {}, []
    started = time.perf_counter()
    try:
        for method in METHODS:
            clients = []
            for client in range(20):
                base, states, counts = cached[method, client]
                weight, bias, cost = rebuild_head(evaluation, models[method], training[client], holdouts[client],
                    seed + 100000 + client, base['raw'], base['learned'], base['labels'])
                target = out / f'{method}_client{client}.npz'
                np.savez_compressed(target, head_weight=weight, head_bias=bias,
                    train_ids=base['train_ids'], holdout_ids=base['holdout_ids'], train_counts=counts,
                    **{'state_' + name: value for name, value in states.items()})
                exported = {'path': target.name, 'sha256': sha(target), 'bytes': target.stat().st_size}
                files.append(exported)
                clients.append({'client': client, 'export': exported, 'cost_and_verification': cost,
                    'array_hashes': {name: array_sha(value) for name, value in arrays(target).items()}})
                print(json.dumps({'task': task, 'seed': seed, 'method': method, 'client': client,
                    'status': 'EXPORTED', 'head_holdout_max_abs_error': cost['holdout_learned_max_abs_error']}), flush=True)
            results[method] = {'clients': clients, 'global_model_tensor_sha256': evaluation.model_hash(models[method]),
                               'checkpoint_sha256': inputs[str(model_root / method / 'checkpoint.pt')],
                               'config_signature': configs[method]['signature']}
        if any(sha(p) != digest for p, digest in {**sources, **inputs}.items()) or attempts:
            raise ValueError('source/model/array changed or forbidden TEST access attempted')
        report = {'status': 'DONE', 'schema': 'train_deployment_states_v1', 'task': task,
            'training_seed': seed, 'partition_seed': TASKS[task][0], 'rounds': 20,
            'methods': list(METHODS), 'model_root': str(model_root), 'evaluation_root': str(evaluation_root),
            'adaptation_root': str(adaptation_root), 'calibration_root': str(calibration_root), 'bcts_root': str(bcts_root),
            'results': results, 'source_identities': sources, 'input_snapshot_sha256_before_after_equal': inputs,
            'train_counts_file': {'path': 'train_counts.npy', 'sha256': sha(out / 'train_counts.npy'),
                                  'array_sha256': array_sha(train_counts)},
            'train_manifest_file': {'path': 'manifest.json', 'sha256': sha(out / 'manifest.json')},
            'state_dimensions': STATE_DIMENSIONS, 'npz_state_prefix': 'state_',
            'head_layout': {'head_weight': [10, 64], 'head_bias': [10], 'dtype': 'float32', 'bytes': 2600},
            'head_reconstruction': 'unchanged frozen evaluate.matched_head, seed+100000+client, one TRAIN epoch',
            'max_head_holdout_abs_error': max(c['cost_and_verification']['holdout_learned_max_abs_error']
                                             for r in results.values() for c in r['clients']),
            'fixed_reproduction_tolerance': TOLERANCE, 'forbidden_official_test_open_attempts': attempts,
            'new_FL_rounds': 0, 'new_calibration_fits': 0, 'official_test_bytes': 0,
            'cost_scope': 'actual reconstructed TRAIN head forward/backward plus two development holdout verification forwards; inherited original costs are not rerun calibration',
            'exports': files, 'elapsed_seconds': time.perf_counter() - started,
            'scientific_scope': 'pre-TEST deployment-state export, no research gate or seed selection',
            'runtime': {'numpy': np.__version__, 'torch': torch.__version__, 'python': sys.version}}
        atomic_json(out / 'result.json', report)
        return report
    except Exception as error:
        atomic_json(out / 'failure.json', {'status': 'FAILED', 'task': task, 'seed': seed,
            'error': repr(error), 'completed_exports': files, 'official_test_open_attempts': attempts})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--task', choices=TASKS, required=True)
    for name in ('model-root', 'evaluation-root', 'adaptation-root', 'calibration-root', 'bcts-root', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    report = run(**vars(args))
    print(json.dumps({'status': report['status'], 'task': report['task'],
                      'seed': report['training_seed'], 'exports': len(report['exports'])}), flush=True)


if __name__ == '__main__':
    main()
