"""Explicit current-path adapter and independent replay of fixed saved inference.

Default is preflight (no TEST bytes or model forward). --execute replays exactly
the six main-method checkpoints on CPU and writes only one fresh result.json.
Historical drivers, locks, models, TRAIN states and TEST exports stay unchanged.
"""
from pathlib import Path
import argparse
import ast
from datetime import datetime
import hashlib
import importlib.util
import json
import platform
import sys
import time

sys.dont_write_bytecode = True
import numpy as np
import torch
from torch import nn

METHOD = Path(__file__).resolve().parents[1]
TASKS = {'cifar': (20261217, (20261231, 20261232, 20261233)),
         'fashion': (20261219, (20261241, 20261242, 20261243))}
METHOD_NAME = 'fedlc_uniform3'
# Fixed before replay; no tolerance is selected from observed discrepancies.
TOLERANCE = {'raw_atol': 2e-6, 'raw_rtol': 2e-6, 'score_atol': 2e-5,
             'score_rtol': 2e-6, 'metric_atol': 2e-6, 'argmax_disagreements': 0}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


class Inputs:
    def __init__(self):
        self.files = {}

    def pin(self, path, expected=None):
        path = Path(path).resolve()
        actual = sha(path)
        require(expected is None or actual == expected, 'source SHA mismatch: ' + str(path))
        require(self.files.get(str(path), actual) == actual, 'changed input: ' + str(path))
        self.files[str(path)] = actual
        return path

    def verify(self):
        require(all(sha(path) == expected for path, expected in self.files.items()), 'input changed during replay')


def expected_suffix(mapping, suffix):
    selected = [value for key, value in mapping.items() if key.endswith('/' + suffix)]
    require(len(selected) == 1, 'one historical SHA identity required for ' + suffix)
    return selected[0]


def load_data_helper(path):
    spec = importlib.util.spec_from_file_location('_replay_frozen_test_data', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def model_class(path):
    # Execute only the recorded class, never the legacy dataset/experiment entry.
    tree = ast.parse(path.read_text())
    nodes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'SmallCNN']
    require(len(nodes) == 1, 'one original SmallCNN class required')
    namespace = {'torch': torch, 'nn': nn, '__name__': '_replay_original_smallcnn'}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['SmallCNN']


def preflight(args):
    inputs = Inputs()
    inputs.pin(__file__)
    helper = args.method_root / 'repro/work/predictive_final_test'
    source = args.project_root / 'tools/fedavg_update_diagnostic.py'
    bundles = []
    for task, (partition, seeds) in TASKS.items():
        final = args.method_root / f'results/predictive_final_test/official_{task}'
        inputs.pin(final / 'result.json')
        complete = read_json(final / 'result.json')
        require(complete['status'] == 'DONE' and complete['fixed_three_seeds'] == list(seeds), 'fixed final task not complete')
        inputs.pin(final / 'assignment.json')
        assignment = read_json(final / 'assignment.json')
        lock = args.method_root / f'checks/predictive_final_test/{task.upper()}_ROOT_LOCK.json'
        task_bundle = {'task': task, 'partition': partition, 'assignment': assignment, 'models': [], 'lock': lock}
        for seed in seeds:
            folder = final / f'{METHOD_NAME}_s{seed}'
            report_path = inputs.pin(folder / 'result.json')
            report = read_json(report_path)
            require((report['status'], report['task'], report['seed'], report['method'], report['rounds'])
                    == ('DONE', task, seed, METHOD_NAME, 20), 'saved fixed model identity mismatch')
            require(report['assignment_hash'] == complete['assignment_hash'] == assignment['assignment_hash'], 'assignment identity mismatch')
            original = report['input_snapshot_sha256_before_after_equal']
            inputs.pin(lock, report['authorization']['root_lock_sha256'])
            for name in ('infer.py', 'data.py', 'scorer.py', 'FINAL_TEST_PLAN.md'):
                inputs.pin(helper / name, expected_suffix(original, 'predictive_final_test/' + name))
            inputs.pin(source, expected_suffix(original, 'tinyproto_diag/tools/fedavg_update_diagnostic.py'))
            tag = f'{task}_p{partition}_s{seed}'
            model_root = args.method_root / f'results/predictive_pfl_confirm/development_{tag}/{METHOD_NAME}'
            state_root = args.method_root / f'results/predictive_final_test/train_states_{tag}'
            checkpoint = inputs.pin(model_root / 'checkpoint.pt', expected_suffix(original, f'development_{tag}/{METHOD_NAME}/checkpoint.pt'))
            inputs.pin(model_root / 'config.json', expected_suffix(original, f'development_{tag}/{METHOD_NAME}/config.json'))
            for name in ('manifest.json', 'train_counts.npy', 'result.json'):
                inputs.pin(state_root / name, expected_suffix(original, f'train_states_{tag}/' + name))
            for client in range(20):
                inputs.pin(state_root / f'{METHOD_NAME}_client{client}.npz',
                           expected_suffix(original, f'train_states_{tag}/{METHOD_NAME}_client{client}.npz'))
            for item in report['exports']:
                inputs.pin(folder / item['path'], item['sha256'])
            require(len(report['exports']) == 20, 'twenty saved client exports required')
            task_bundle['models'].append({'seed': seed, 'checkpoint': checkpoint, 'model_root': model_root,
                                         'state_root': state_root, 'folder': folder, 'report': report})
        if task == 'cifar':
            task_bundle['raw'] = args.project_root / 'dataset/Cifar10/rawdata'
            raw_files = [task_bundle['raw'] / 'cifar-10-batches-py/test_batch']
        else:
            task_bundle['raw'] = args.fashion_raw
            raw_files = [args.fashion_raw / name for name in ('t10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz')]
        require(all(path.is_file() for path in raw_files), 'missing explicit official TEST source')
        task_bundle['raw_files'] = raw_files
        bundles.append(task_bundle)
    return inputs, helper, source, bundles


def normalize(images, task):
    # Independently state the fixed normalization instead of reusing the driver.
    if task == 'cifar':
        x = np.transpose(images, (0, 3, 1, 2)).astype(np.float32)
    else:
        x = np.pad(images[:, None, :, :], ((0, 0), (0, 0), (2, 2), (2, 2)))
        x = np.repeat(x, 3, axis=1).astype(np.float32)
    return (x / np.float32(255.) - np.float32(.5)) / np.float32(.5)


def forward(model, images, task):
    result = []
    with torch.inference_mode():
        for begin in range(0, len(images), 256):
            x = torch.from_numpy(normalize(images[begin:begin + 256], task))
            result.append(model(x).numpy().copy())
    logits = np.concatenate(result)
    require(logits.shape == (10000, 10) and logits.dtype == np.float32 and np.isfinite(logits).all(), 'invalid raw network output')
    return logits


def metrics(z, y):
    z = np.asarray(z, dtype=np.float64)
    shifted = z - z.max(axis=1, keepdims=True)
    ex = np.exp(shifted)
    p = ex / ex.sum(axis=1, keepdims=True)
    correct = z.argmax(axis=1) == y
    confidence = p.max(axis=1)
    residual = p - np.eye(10)[y]
    hits, conf = [], []
    for i in range(15):
        mask = (confidence >= i / 15) & (confidence < (i + 1) / 15 if i < 14 else confidence <= 1)
        hits.append(float(correct[mask].sum()))
        conf.append(float(confidence[mask].sum()))
    return {'accuracy': float(correct.mean()),
            'nll': float(np.mean(np.log(ex.sum(axis=1)) - shifted[np.arange(len(y)), y])),
            'brier': float(np.mean(np.sum(residual ** 2, axis=1))),
            'ece15': float(np.abs(np.array(hits) - conf).sum() / len(y)),
            'hits': hits, 'confidence': conf, 'samples': len(y)}


def summary(rows):
    result = {}
    weights = np.array([row['samples'] for row in rows], dtype=np.float64)
    for name in ('accuracy', 'nll', 'brier', 'ece15'):
        values = np.array([row[name] for row in rows])
        result['macro_' + name] = float(values.mean())
        result['weighted_' + name] = float(np.average(values, weights=weights))
    result['macro_loss'] = result['macro_nll']
    result['weighted_loss'] = result['weighted_nll']
    result['client_accuracy_q10'] = float(np.quantile([row['accuracy'] for row in rows], .1))
    result['pooled_ece15'] = float(np.abs(np.sum([row['hits'] for row in rows], axis=0)
                                              - np.sum([row['confidence'] for row in rows], axis=0)).sum() / weights.sum())
    return result


def execute(args, inputs, helper, source, bundles):
    require(not args.out.exists(), 'fresh result path required; never overwrite any result')
    began = time.perf_counter()
    torch.set_num_threads(1)
    data = load_data_helper(helper / 'data.py')
    cls = model_class(source)
    runs, failures, raw_bytes = [], [], 0
    for task_bundle in bundles:
        task = task_bundle['task']
        expected_files = task_bundle['models'][0]['report']['test_provenance']['files']
        for path in task_bundle['raw_files']:
            expected = next(item['sha256'] for item in expected_files if Path(item['path']).name == path.name)
            inputs.pin(path, expected)
            raw_bytes += path.stat().st_size
        loader = data.load_cifar_test if task == 'cifar' else data.load_fashion_test
        images, labels, provenance = loader(task_bundle['raw'], root_lock=task_bundle['lock'], execute=True)
        counts = np.asarray(task_bundle['assignment']['train_counts'], dtype=np.int64)
        groups, assignment = data.assign_clients(labels, counts, task=task,
            train_manifest_sha256=task_bundle['assignment']['train_manifest_sha256'])
        require(assignment['assignment_hash'] == task_bundle['assignment']['assignment_hash'], 'replayed assignment differs')
        for bundle in task_bundle['models']:
            started = time.perf_counter()
            config = read_json(bundle['model_root'] / 'config.json')
            checkpoint = torch.load(bundle['checkpoint'], map_location='cpu', weights_only=True)
            require(checkpoint['config'] == config and checkpoint['signature'] == config['signature'], 'checkpoint/config mismatch')
            require(config['seed'] == bundle['seed'] and config['partition_seed'] == task_bundle['partition']
                    and config['rounds'] == len(checkpoint['rows']) == 20 and config['method'] == METHOD_NAME, 'wrong checkpoint identity')
            model = cls().cpu().eval().requires_grad_(False)
            expected_state = model.state_dict()
            require(set(checkpoint['model']) == set(expected_state), 'checkpoint state keys differ')
            require(all(isinstance(value, torch.Tensor) and value.device.type == 'cpu'
                        and value.dtype == torch.float32 and value.shape == expected_state[key].shape
                        and bool(torch.isfinite(value).all()) for key, value in checkpoint['model'].items()), 'invalid model tensor')
            model.load_state_dict(checkpoint['model'], strict=True)
            require(sum(p.numel() for p in model.parameters()) == 136874, 'model parameter count changed')
            logits = forward(model, images, task)
            clients, observed = [], []
            for client, ids in enumerate(groups):
                old_path = bundle['folder'] / f'client{client}.npz'
                with np.load(old_path, allow_pickle=False) as saved:
                    require(np.array_equal(saved['test_ids'], ids) and np.array_equal(saved['test_labels'], labels[ids]), 'saved row/GT mismatch')
                    old_raw = saved['raw_logits'].copy()
                    state = saved['state_prioranchored_vector'].copy()
                    old_score = saved['score_prioranchored_vector'].copy()
                with np.load(bundle['state_root'] / f'{METHOD_NAME}_client{client}.npz', allow_pickle=False) as train:
                    require(np.array_equal(state, train['state_prioranchored_vector']), 'deployed state differs from locked TRAIN state')
                raw = logits[ids]
                score = raw.astype(np.float64) * state[:10].astype(np.float64) + state[10:].astype(np.float64)
                current = metrics(score, labels[ids])
                recorded = bundle['report']['clients'][client]['metrics']['prioranchored_vector']
                delta = {name: current[name] - recorded[name] for name in ('accuracy', 'nll', 'brier', 'ece15')}
                raw_difference = int(np.sum(raw.argmax(axis=1) != old_raw.argmax(axis=1)))
                final_difference = int(np.sum(score.argmax(axis=1) != old_score.argmax(axis=1)))
                passed = (np.allclose(raw, old_raw, atol=TOLERANCE['raw_atol'], rtol=TOLERANCE['raw_rtol'])
                          and np.allclose(score, old_score, atol=TOLERANCE['score_atol'], rtol=TOLERANCE['score_rtol'])
                          and raw_difference == final_difference == 0
                          and max(abs(value) for value in delta.values()) <= TOLERANCE['metric_atol'])
                clients.append({'client': client, 'test_rows': len(ids), 'status': 'PASS' if passed else 'FAIL',
                    'raw_max_abs_error': float(np.max(np.abs(raw.astype(np.float64) - old_raw))),
                    'raw_bitwise_equal': bool(np.array_equal(raw.view(np.uint32), old_raw.view(np.uint32))),
                    'raw_argmax_disagreements': raw_difference, 'final_score_max_abs_error': float(np.max(np.abs(score - old_score))),
                    'final_argmax_disagreements': final_difference, 'final_metric_differences': delta})
                observed.append(current)
                if not passed:
                    failures.append({'task': task, 'seed': bundle['seed'], 'client': client})
            current_summary = summary(observed)
            old_summary = bundle['report']['summaries']['prioranchored_vector']
            differences = {key: value - old_summary[key] for key, value in current_summary.items()}
            if max(abs(value) for value in differences.values()) > TOLERANCE['metric_atol']:
                failures.append({'task': task, 'seed': bundle['seed'], 'error': 'summary tolerance exceeded'})
            run = {'task': task, 'seed': bundle['seed'], 'method': METHOD_NAME,
                   'checkpoint_sha256': inputs.files[str(bundle['checkpoint'])], 'original_inference_backend': bundle['report']['cost']['inference_driver_cost']['backend'],
                   'replay_backend': 'CPU', 'assignment_hash': assignment['assignment_hash'],
                   'raw_logit_shape': list(logits.shape), 'clients': clients,
                   'replayed_main_summary': current_summary, 'summary_differences': differences,
                   'seconds': time.perf_counter() - started}
            runs.append(run)
            print(json.dumps({'task': task, 'seed': bundle['seed'], 'clients': 20,
                  'raw_max_abs_error': max(row['raw_max_abs_error'] for row in clients),
                  'summary_max_abs_error': max(abs(value) for value in differences.values())}), file=sys.stderr, flush=True)
    inputs.verify()
    all_clients = [row for run in runs for row in run['clients']]
    result = {'status': 'PASS' if not failures else 'FAIL', 'created_at': datetime.now().astimezone().isoformat(),
              'scope': 'Independent current-path CPU image -> original checkpoint -> raw logits -> locked prior-vector TEST metrics replay; no model selection',
              'path_adapter': {'project_root': str(args.project_root), 'method_root': str(args.method_root), 'fashion_raw': str(args.fashion_raw),
                               'historical_driver_sources_modified': False, 'historical_locks_or_results_modified': False},
              'tolerances_declared_before_replay': TOLERANCE, 'runs': runs, 'failures': failures,
              'counts': {'models': len(runs), 'client_seed_combinations': len(all_clients), 'cnn_forward_images': len(runs) * 10000,
                         'cnn_forward_batches': len(runs) * 40, 'official_test_raw_source_bytes': raw_bytes,
                         'new_training_rounds': 0, 'new_training_seeds': 0, 'new_fit_calls': 0, 'new_backward_calls': 0},
              'maximum_raw_abs_error': max(row['raw_max_abs_error'] for row in all_clients),
              'all_raw_logits_bitwise_equal': all(row['raw_bitwise_equal'] for row in all_clients),
              'raw_argmax_disagreements': sum(row['raw_argmax_disagreements'] for row in all_clients),
              'final_argmax_disagreements': sum(row['final_argmax_disagreements'] for row in all_clients),
              'maximum_main_summary_abs_error': max(abs(value) for run in runs for value in run['summary_differences'].values()),
              'source_and_input_sha256_before_after_equal': inputs.files,
              'checkpoint_load': 'torch.load(weights_only=True, map_location=cpu); only original AST SmallCNN class executed',
              'runtime': {'python': sys.version, 'numpy': np.__version__, 'torch': torch.__version__, 'platform': platform.platform(), 'cpu_threads': 1},
              'limitations': ['Replays the fixed main fedlc_uniform3 raw network and prior-vector predictor only; reference methods and adapted Linear heads are not rerun.',
                              'Original TEST network inference used CPU even though shared model training used MPS; this replay is not a new training trajectory.',
                              'This is a local verification, not external cross-model review or real network/ledger validation.'],
              'elapsed_seconds': time.perf_counter() - began}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as handle:
        handle.write(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    print(json.dumps({'status': result['status'], 'output': str(args.out), 'models': len(runs),
                      'clients': len(all_clients), 'max_raw_error': result['maximum_raw_abs_error'],
                      'max_summary_error': result['maximum_main_summary_abs_error']}))
    return 0 if not failures else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--project-root', type=Path, default=METHOD.parent / '02_实验工程')
    parser.add_argument('--method-root', type=Path, default=METHOD)
    parser.add_argument('--fashion-raw', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    args.project_root, args.method_root = args.project_root.resolve(), args.method_root.resolve()
    args.fashion_raw = (args.fashion_raw or args.method_root / 'scripts/datasets/FashionMNIST/raw').resolve()
    args.out = (args.out or args.method_root / 'results/replay_saved_inference/result.json').resolve()
    inputs, helper, source, bundles = preflight(args)
    if not args.execute:
        checkpoints = [model['checkpoint'] for task in bundles for model in task['models']]
        print(json.dumps({'status': 'PREFLIGHT_PASS', 'inputs_checked': len(inputs.files),
              'models': len(checkpoints), 'checkpoint_bytes': sum(path.stat().st_size for path in checkpoints),
              'official_test_files': [{'path': str(path), 'bytes': path.stat().st_size} for task in bundles for path in task['raw_files']],
              'test_bytes_read': 0, 'model_forward_calls': 0, 'files_written': 0,
              'explicit_paths': {'project_root': str(args.project_root), 'method_root': str(args.method_root), 'fashion_raw': str(args.fashion_raw)}}))
        return 0
    return execute(args, inputs, helper, source, bundles)


if __name__ == '__main__':
    raise SystemExit(main())
