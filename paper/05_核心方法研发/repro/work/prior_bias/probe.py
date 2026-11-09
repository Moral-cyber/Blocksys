"""One frozen-model TRAIN forward per sample, then fixed NumPy bias fitting."""
from pathlib import Path
import argparse
import hashlib
import json
import pickle
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from solver import fit, stats, summarize

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE.parent / 'method_followup'))
import run_followup as r
import run_partitions as data

METHODS = ('ce_uniform3', 'fedlc_uniform3', 'balanced_uniform3', 'balanced_dense')
MODEL_ROOT = HERE.parent / 'balanced_quant/development_p20261193_s20261201'
BASELINE_ROOT = HERE.parent / 'prior_personalization/diagnostic_v1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def load_train_only(raw, manifest, inputs):
    """Read only the five TRAIN pickle files; preserve existing row/normalization."""
    images, labels = [], []
    expected = {row['name']: row['sha256'] for row in manifest['official_train_batch_files']}
    for number in range(1, 6):
        path = raw / 'cifar-10-batches-py' / f'data_batch_{number}'
        digest = sha(path)
        if digest != expected[path.name]:
            raise ValueError('official TRAIN source differs from frozen data manifest')
        inputs[str(path)] = digest
        with path.open('rb') as stream:
            batch = pickle.load(stream, encoding='latin1')
        images.append(np.asarray(batch['data'], dtype=np.uint8))
        labels.extend(batch['labels'])
    images = np.concatenate(images).reshape(50000, 3, 32, 32).transpose(0, 2, 3, 1)
    labels = np.asarray(labels, dtype=np.int64)
    if (images.shape != (50000, 32, 32, 3) or labels.shape != (50000,)
            or not np.array_equal(np.bincount(labels, minlength=10), np.full(10, 5000))
            or hashlib.sha256(labels.astype('<i8').tobytes()).hexdigest() != manifest['official_train_labels_sha256']):
        raise ValueError('official TRAIN row order or labels mismatch')
    return images, labels


def model_identity(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--out', type=Path, default=HERE / 'diagnostic_v1')
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists():
        raise ValueError('fresh diagnostic out required')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    opened_test = []

    def audit(event, arguments):
        if event == 'open' and isinstance(arguments[0], (str, bytes)):
            path = str(arguments[0])
            if 'test_batch' in path:
                opened_test.append(path)
                raise ValueError('official TEST byte read forbidden in this new diagnostic')

    sys.addaudithook(audit)
    sources = {str(HERE / name): sha(HERE / name) for name in ('PLAN.md', 'solver.py', 'probe.py')}
    helper_sources = {str(path): sha(path) for path in
        (Path(r.__file__), Path(data.__file__), Path(sys.modules[r.SmallCNN.__module__].__file__))}
    inputs = {str(BASELINE_ROOT / 'result.json'): sha(BASELINE_ROOT / 'result.json')}
    previous = json.loads((BASELINE_ROOT / 'result.json').read_text())
    if previous['status'] != 'DONE':
        raise ValueError('prior baseline diagnostics must be complete')
    configs, checkpoints = {}, {}
    for method in METHODS:
        folder = MODEL_ROOT / method
        for name in ('config.json', 'checkpoint.pt'):
            path = folder / name
            inputs[str(path)] = sha(path)
            if inputs[str(path)] != previous['input_sha256'][str(path)]:
                raise ValueError('model input differs from existing PFL probe')
        config = json.loads((folder / 'config.json').read_text())
        checkpoint = torch.load(folder / 'checkpoint.pt', weights_only=False)
        unsigned = {key: value for key, value in config.items() if key != 'signature'}
        expected_signature = hashlib.sha256(json.dumps(unsigned, sort_keys=True).encode()).hexdigest()
        if (checkpoint['signature'] != config['signature'] or config['signature'] != expected_signature
                or len(checkpoint['rows']) != 20 or config['rounds'] != 20):
            raise ValueError('incomplete or incompatible frozen model')
        configs[method], checkpoints[method] = config, checkpoint
    common = configs[METHODS[0]]['data_manifest']
    if sum(map(len, common['train_indices'])) != 37502 or sum(map(len, common['holdout_indices'])) != 12498:
        raise ValueError('unexpected actual sample counts')
    for config in configs.values():
        if config['data_manifest'] != common:
            raise ValueError('frozen models do not share the same partition')
    trainids = np.concatenate(common['train_indices'])
    holdoutids = np.concatenate(common['holdout_indices'])
    if len(np.unique(np.concatenate((trainids, holdoutids)))) != 50000:
        raise ValueError('TRAIN/clientholdout membership overlap or omissions')
    images, labels = load_train_only(data.RAW, common, inputs)
    out.mkdir(parents=True)
    atomic_json(out / 'inputs_before.json', {**sources, **helper_sources, **inputs})
    results, inference_seconds, optimization_seconds = {}, 0., 0.
    overall_start = time.perf_counter()
    all_forwards, all_batches, all_newton_steps = 0, 0, 0
    for method in METHODS:
        model = r.SmallCNN()
        model.load_state_dict(checkpoints[method]['model'])
        model.eval()
        model.requires_grad_(False)
        identity = model_identity(model)
        clients = []
        for client, (training_ids, evaluation_ids) in enumerate(zip(common['train_indices'], common['holdout_indices'])):
            training_ids = np.asarray(training_ids, dtype=np.int64)
            path = BASELINE_ROOT / f'{method}_client{client}.npz'
            inputs[str(path)] = sha(path)
            with np.load(path, allow_pickle=False) as saved:
                holdout = {name: saved[name].copy() for name in saved.files}
            counts = np.bincount(labels[training_ids], minlength=10)
            if (not np.array_equal(holdout['holdout_ids'], evaluation_ids)
                    or not np.array_equal(holdout['labels'], labels[evaluation_ids])
                    or not np.array_equal(holdout['train_counts'], counts)
                    or counts.tolist() != common['client_label_counts'][client]):
                raise ValueError('existing holdout or TRAIN counts do not match frozen manifest')
            expected_prior = np.log((counts + 1) / (len(training_ids) + 10)).astype(np.float32)
            if not np.array_equal(holdout['prior_shift'], expected_prior):
                raise ValueError('existing prior shift is not the fixed local TRAIN prior')
            logits, observed_labels, forward_batches = [], [], 0
            started = time.perf_counter()
            dataset = data.IndexedCifar10(images, labels, training_ids)
            with torch.inference_mode():
                for x, y in DataLoader(dataset, batch_size=256, shuffle=False):
                    logits.append(model(x).numpy().copy())
                    observed_labels.append(y.numpy().copy())
                    forward_batches += 1
            forward_seconds = time.perf_counter() - started
            train_logits, train_labels = np.concatenate(logits), np.concatenate(observed_labels)
            if not np.array_equal(train_labels, labels[training_ids]) or model_identity(model) != identity:
                raise ValueError('forward membership/order or frozen model changed')
            started = time.perf_counter()
            bias, optimization = fit(train_logits, train_labels)
            fit_seconds = time.perf_counter() - started
            raw = holdout['raw'].astype(np.float64)
            learned = holdout['learned'].astype(np.float64)
            prior = holdout['prior_shift'].astype(np.float64)
            scores = {'raw': raw, 'analytic_prior': raw + prior,
                'prior_only': np.broadcast_to(prior, raw.shape),
                'learned_head': learned, 'learned_head_plus_prior': learned + prior,
                'fitted_bias': raw + bias.astype(np.float64)}
            metrics = {name: stats(value, holdout['labels']) for name, value in scores.items()}
            baseline_metrics = previous['results'][method]['clients'][client]['metrics']
            for name in baseline_metrics:
                if (abs(metrics[name]['accuracy'] - baseline_metrics[name]['accuracy']) > 1e-12
                        or abs(metrics[name]['loss'] - baseline_metrics[name]['loss']) > 3e-6):
                    raise ValueError('saved baseline metrics do not reproduce')
            np.savez_compressed(out / f'{method}_client{client}.npz',
                train_logits=train_logits, train_labels=train_labels,
                train_ids=training_ids.astype(np.uint32), holdout_raw=holdout['raw'],
                holdout_learned=holdout['learned'], holdout_labels=holdout['labels'],
                holdout_ids=holdout['holdout_ids'], prior_shift=holdout['prior_shift'],
                fitted_bias=bias, prior_bias_float64=np.array(optimization['prior_bias_float64']))
            clients.append({'client': client, 'train_samples': len(training_ids),
                'holdout_samples': len(holdout['labels']), 'metrics': metrics,
                'extra_train_image_forwards': len(training_ids),
                'extra_cnn_forward_batches': forward_batches, 'extra_cnn_backward': 0,
                'extra_learned_head_training': False, 'extra_uplink_bytes': 0,
                'local_state_bytes': int(bias.nbytes), 'forward_seconds': forward_seconds,
                'numpy_fit_seconds': fit_seconds, 'optimization': optimization})
            all_forwards += len(training_ids)
            all_batches += forward_batches
            all_newton_steps += optimization['accepted_newton_steps']
            inference_seconds += forward_seconds
            optimization_seconds += fit_seconds
        results[method] = {'clients': clients, 'summaries': summarize(clients),
            'model_before_after_same': True, 'train_samples_forwarded': sum(client['train_samples'] for client in clients),
            'evaluation': 'existing official TRAIN clientholdout, includes earlier globaldev'}
        print(json.dumps({'method': method, 'summaries': results[method]['summaries'],
            'train_forwards': results[method]['train_samples_forwarded']}), flush=True)
    candidate = results['balanced_uniform3']['summaries']['fitted_bias']
    same_prior = results['balanced_uniform3']['summaries']['analytic_prior']
    best_prior_method = max(METHODS, key=lambda method: results[method]['summaries']['analytic_prior']['macro_accuracy'])
    best_prior = results[best_prior_method]['summaries']['analytic_prior']
    learned_method, learned_variant = max(((method, variant) for method in METHODS
        for variant in ('learned_head', 'learned_head_plus_prior')),
        key=lambda pair: results[pair[0]]['summaries'][pair[1]]['macro_accuracy'])
    best_learned = results[learned_method]['summaries'][learned_variant]
    gates = {'vs_same_model_prior': {'accuracy_plus_half_pp': candidate['macro_accuracy'] >= same_prior['macro_accuracy'] + .005,
                    'lower_ce': candidate['macro_loss'] < same_prior['macro_loss']},
        'vs_best_existing_prior': {'method': best_prior_method,
                    'accuracy_plus_half_pp': candidate['macro_accuracy'] >= best_prior['macro_accuracy'] + .005,
                    'lower_ce': candidate['macro_loss'] < best_prior['macro_loss']},
        'vs_best_learned': {'method': learned_method, 'variant': learned_variant,
                    'within_half_pp': candidate['macro_accuracy'] >= best_learned['macro_accuracy'] - .005}}
    passed = (all(gates['vs_same_model_prior'].values())
        and gates['vs_best_existing_prior']['accuracy_plus_half_pp'] and gates['vs_best_existing_prior']['lower_ce']
        and gates['vs_best_learned']['within_half_pp'])
    protected = {**sources, **helper_sources, **inputs}
    for path, expected in protected.items():
        if sha(path) != expected:
            raise ValueError('frozen source/input changed: ' + path)
    if all_forwards != 4 * 37502 or opened_test:
        raise ValueError('forward count or official TEST byte policy failure')
    report = {'status': 'DONE', 'candidate_model_fixed_before_results': 'balanced_uniform3',
        'decision': 'FOLLOW_UP' if passed else 'STOP', 'gates': gates, 'results': results,
        'input_sha256_before_after_equal': protected, 'source_sha256': sources,
        'official_test_byte_read_attempts': opened_test, 'official_test_inference': False,
        'train_source_loader': 'only five fixed data_batch files, same row order and IndexedCifar10 normalization',
        'historical_integrity_scope': 'older torchvision train=True constructors may MD5-read test_batch; this is not test-record inference or test-metric model selection',
        'fitting_data': 'local TRAIN labels/logits only; saved clientholdout arrays evaluated only after fitting',
        'hyperparameter_selection': 'C=10; 20 maximum Newton steps; no holdout selection or grid',
        'total_extra_train_image_forwards': all_forwards, 'total_extra_cnn_forward_batches': all_batches,
        'total_extra_cnn_backward': 0, 'total_newton_steps': all_newton_steps,
        'total_extra_uplink_bytes': 0, 'local_bias_state_bytes_per_client': 40,
        'local_bias_state_bytes_per_model_20_clients': 800,
        'forward_seconds': inference_seconds, 'numpy_fit_seconds': optimization_seconds,
        'elapsed_compute_seconds': time.perf_counter() - overall_start,
        'communication_scope': 'incremental local computation/state; no end-to-end communication or network measurement',
        'environment': {'python': sys.version, 'torch': str(torch.__version__), 'numpy': np.__version__,
                        'backend': 'cpu', 'threads': torch.get_num_threads()}}
    atomic_json(out / 'inputs_before.json', protected)
    atomic_json(out / 'result.json', report)
    print(json.dumps({'decision': report['decision'], 'gates': gates,
                      'extra_train_forwards': all_forwards}), flush=True)


if __name__ == '__main__':
    main()
