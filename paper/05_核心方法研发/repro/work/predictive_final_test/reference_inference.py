"""Freeze, then optionally infer the fixed CIFAR TopK/FedRoD references.

Default: checkpoint/state/source preflight only, with no image forward or TEST
file read. --execute additionally requires a root-lock binding the preflight
snapshot and this source. There is no training, fitting, or model selection.
"""
from pathlib import Path
import argparse
import ast
import io
import json
import sys
import time

import numpy as np
import torch
from torch import nn

import data as d
import scorer as s

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
WORK = HERE.parent
SEEDS = (20261231, 20261232, 20261233)
PARTITION = 20261217
TOPK = ('ce_topk', 'fedlc_topk')
FEDROD = 'fedrod_linear_uniform3'
TOPK_VARIANTS = ('raw', 'prior_only', 'analytic_prior', 'prioranchor_bias',
                 'zero_anchor_vector', 'prioranchored_vector')
FEDROD_VARIANTS = ('generic_raw', 'personal_residual', 'generic_plus_local_residual')
STATE_DIMS = {'analytic_prior': 10, 'prioranchor_bias': 10,
              'zero_anchor_vector': 20, 'prioranchored_vector': 20}
MODEL_SOURCE = WORK / 'tinyproto_diag/tools/fedavg_update_diagnostic.py'


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temporary.replace(path)


def signed(value):
    # Existing training signatures use ordinary json.dumps separators.
    return d.sha_bytes(json.dumps(value, sort_keys=True, allow_nan=False).encode())


class Inputs:
    """Hash the exact bytes used and reject inconsistent rereads/pinned sources."""
    def __init__(self):
        self.files = {}

    def read(self, path, expected=None):
        path = Path(path)
        path = (ROOT / path).resolve() if not path.is_absolute() else path.resolve()
        if path.name == 'test_batch' or path.name.startswith('t10k-'):
            raise ValueError('official TEST cannot be a preflight input')
        payload = path.read_bytes()
        digest = d.sha_bytes(payload)
        if (expected is not None and digest != expected) or self.files.get(str(path), digest) != digest:
            raise ValueError('changed/mismatched frozen input: ' + str(path))
        self.files[str(path)] = digest
        return payload

    def json(self, path):
        return json.loads(self.read(path))

    def pins(self, mapping):
        if not isinstance(mapping, dict) or not mapping:
            raise ValueError('nonempty original source/input identities required')
        for path, expected in mapping.items():
            self.read(path, expected)

    def verify(self):
        if any(d.file_sha(path) != expected for path, expected in self.files.items()):
            raise ValueError('input changed during reference inference/preflight')


def model_class(inputs):
    # Compile only the original class; do not import its legacy data entry points.
    tree = ast.parse(inputs.read(MODEL_SOURCE).decode())
    nodes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'SmallCNN']
    if len(nodes) != 1:
        raise ValueError('one original SmallCNN class required')
    namespace = {'torch': torch, 'nn': nn, '__name__': __name__}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(MODEL_SOURCE), 'exec'), namespace)
    return namespace['SmallCNN']


def checked_state(state, template):
    expected = template.state_dict()
    if not isinstance(state, dict) or set(state) != set(expected):
        raise ValueError('exact model/head state layout required')
    for key, value in state.items():
        if (not isinstance(value, torch.Tensor) or value.shape != expected[key].shape
                or value.dtype != torch.float32 or value.device.type != 'cpu'
                or not torch.isfinite(value).all()):
            raise ValueError('finite FP32 CPU model/head tensors required: ' + key)
    template.load_state_dict(state, strict=True)
    template.eval().requires_grad_(False)
    return template


def npz(inputs, path, names):
    with np.load(io.BytesIO(inputs.read(path)), allow_pickle=False) as archive:
        if not set(names).issubset(archive.files):
            raise ValueError('missing already-fitted state in ' + str(path))
        return {name: archive[name].copy() for name in names}


def load_model(inputs, cls, folder, seed, method):
    config = inputs.json(folder / 'config.json')
    if config.get('signature') != signed({key: value for key, value in config.items() if key != 'signature'}):
        raise ValueError('invalid training config signature')
    base = config['base'] if method == FEDROD else config
    if (base.get('seed'), base.get('partition_seed'), base.get('rounds'), base.get('method')) != (seed, PARTITION, 20, method):
        raise ValueError('only fixed final20/partition/seed/method checkpoints allowed')
    inputs.pins(base['source_identities'])
    if method == FEDROD:
        inputs.pins(config['train_adapter_identities'])
    history = inputs.json(folder / 'result.json')
    checkpoint = torch.load(io.BytesIO(inputs.read(folder / 'checkpoint.pt')), map_location='cpu', weights_only=True)
    rows = checkpoint.get('rows')
    if (checkpoint.get('version') != 1 or checkpoint.get('config') != config
            or checkpoint.get('signature') != config['signature'] or history.get('status') != 'DONE'
            or history.get('signature') != config['signature'] or history.get('rows') != rows
            or not isinstance(rows, list) or len(rows) != 20
            or [row.get('round') for row in rows] != list(range(1, 21))):
        raise ValueError('complete matching final checkpoint/history required')
    with torch.random.fork_rng(devices=[]):
        model = checked_state(checkpoint['model'], cls())
    if sum(parameter.numel() for parameter in model.parameters()) != 136874:
        raise ValueError('fixed SmallCNN136874 required')
    counts = d.validate_train_counts(config['data_manifest']['client_label_counts'])
    if not np.array_equal(counts.sum(axis=1), np.asarray(config['data_manifest']['client_sizes'])):
        raise ValueError('TRAIN counts and manifest client sizes differ')
    if not np.allclose(config['weights'], counts.sum(axis=1) / counts.sum(), rtol=0, atol=1e-15):
        raise ValueError('TRAIN count aggregation weights differ')
    heads = []
    if method == FEDROD:
        if not isinstance(checkpoint.get('heads'), list) or len(checkpoint['heads']) != 20:
            raise ValueError('twenty persistent FedRoD heads required')
        for state in checkpoint['heads']:
            with torch.random.fork_rng(devices=[]):
                heads.append(checked_state(state, nn.Linear(64, 10)))
    return {'method': method, 'seed': seed, 'model': model, 'heads': heads,
            'manifest': config['data_manifest'], 'counts': counts, 'states': []}


def preflight():
    inputs = Inputs()
    for name in ('reference_inference.py', 'data.py', 'scorer.py', 'FINAL_TEST_PLAN.md'):
        inputs.read(HERE / name)
    cls = model_class(inputs)
    bundles, frozen = [], {}
    for seed in SEEDS:
        root = WORK / 'calibrated_topk_reference'
        evaluation = root / f'evaluation_p{PARTITION}_s{seed}'
        adaptation = root / f'adaptation_p{PARTITION}_s{seed}'
        er, ar = inputs.json(evaluation / 'result.json'), inputs.json(adaptation / 'result.json')
        if any(report.get('status') != 'DONE' or report.get('training_seed') != seed
               or report.get('partition_seed') != PARTITION or set(report.get('results', {})) != set(TOPK)
               for report in (er, ar)):
            raise ValueError('complete paired TopK evaluation/adaptation required')
        for report, fields in ((er, ('source_identities', 'frozen_model_inputs')),
                               (ar, ('source_identities', 'input_sha256_before_after_equal'))):
            for field in fields:
                inputs.pins(report[field])
        for method in TOPK:
            bundle = load_model(inputs, cls, root / f'development_p{PARTITION}_s{seed}' / method, seed, method)
            for report in (er, ar):
                if [row['client'] for row in report['results'][method]['clients']] != list(range(20)):
                    raise ValueError('all20 ordered fitted-state clients required')
            for client in range(20):
                old = npz(inputs, evaluation / f'{method}_client{client}.npz',
                          ('train_ids', 'train_counts', 'prior_shift', 'fitted_bias'))
                new = npz(inputs, adaptation / f'{method}_client{client}.npz',
                          ('train_ids', 'prioranchored_vector', 'zero_anchor_vector', 'prioranchor_bias'))
                if (not np.array_equal(old['train_counts'], bundle['counts'][client])
                        or not np.array_equal(old['train_ids'], new['train_ids'])
                        or not np.array_equal(old['train_ids'], bundle['manifest']['train_indices'][client])
                        or not np.array_equal(old['fitted_bias'], new['prioranchor_bias'])):
                    raise ValueError('fitted states do not bind the exact local TRAIN client')
                state = {name: new[name] for name in STATE_DIMS if name != 'analytic_prior'}
                state['analytic_prior'] = old['prior_shift']
                expected = np.log((bundle['counts'][client] + 1.) / (bundle['counts'][client].sum() + 10.)).astype(np.float32)
                for name, dimension in STATE_DIMS.items():
                    value = state[name]
                    if value.dtype != np.float32 or value.shape != (dimension,) or not np.isfinite(value).all():
                        raise ValueError('exact finite already-fitted FP32 state required: ' + name)
                if not np.array_equal(expected, state['analytic_prior']):
                    raise ValueError('frozen prior differs from local TRAIN counts')
                bundle['states'].append(state)
            for name in STATE_DIMS:
                frozen[f'{method}_s{seed}_{name}'] = np.stack([state[name] for state in bundle['states']])
            bundles.append(bundle)
        folder = WORK / 'fedrod_linear_reference' / f'development_p{PARTITION}_s{seed}'
        bundle = load_model(inputs, cls, folder, seed, FEDROD)
        frozen[f'{FEDROD}_s{seed}_weight'] = np.stack([head.weight.numpy().copy() for head in bundle['heads']])
        frozen[f'{FEDROD}_s{seed}_bias'] = np.stack([head.bias.numpy().copy() for head in bundle['heads']])
        bundles.append(bundle)
    manifest = bundles[0]['manifest']
    if any(bundle['manifest'] != manifest or not np.array_equal(bundle['counts'], bundles[0]['counts']) for bundle in bundles):
        raise ValueError('every reference method/seed must share the exact TRAIN partition/counts')
    frozen['train_counts'] = bundles[0]['counts']
    inputs.verify()
    return inputs, bundles, frozen, d.canonical_sha(manifest)


def prepare_snapshot(out, *, create):
    inputs, bundles, frozen, manifest_sha = preflight()
    out = Path(out).resolve()
    snapshot_path, state_path = out / 'PRETEST_SNAPSHOT.json', out / 'FROZEN_STATES.npz'
    if out.exists() and set(path.name for path in out.iterdir()) - {snapshot_path.name, state_path.name}:
        raise ValueError('fresh output or unchanged preflight-only output required; no overwrite/resume')
    if not snapshot_path.exists():
        if not create or state_path.exists():
            raise ValueError('pre-existing complete preflight snapshot required before --execute')
        out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(state_path, **frozen)
    with np.load(state_path, allow_pickle=False) as saved:
        if set(saved.files) != set(frozen) or any(not np.array_equal(saved[key], value)
                or saved[key].dtype != value.dtype for key, value in frozen.items()):
            raise ValueError('frozen exported state differs from original fitted state')
    snapshot = {'schema': 'fixed_cifar_references_pretest_v1', 'task': 'cifar', 'partition_seed': PARTITION,
        'seeds': list(SEEDS), 'methods': [*TOPK, FEDROD], 'train_manifest_sha256': manifest_sha,
        'train_counts_int64le_sha256': d.array_sha(frozen['train_counts'], '<i8'),
        'input_sha256': inputs.files, 'frozen_states_file': str(state_path),
        'frozen_states_sha256': d.file_sha(state_path),
        'state_arrays': {key: {'shape': list(value.shape), 'dtype': str(value.dtype),
            'sha256': d.array_sha(value, '<i8' if key == 'train_counts' else '<f4')} for key, value in frozen.items()},
        'official_test_byte_reads': False, 'image_forwards': 0, 'fit_calls': 0, 'backward_calls': 0}
    if snapshot_path.exists():
        if json.loads(snapshot_path.read_text()) != snapshot:
            raise ValueError('preflight snapshot differs from current models/states/sources')
    else:
        write_json(snapshot_path, snapshot)
    inputs.verify()
    return inputs, bundles, snapshot, d.file_sha(snapshot_path)


def forward_cpu(model, images, batch_size=256):
    raw, hidden = [], []
    started = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, len(images), batch_size):
            x = torch.from_numpy(d.normalize_batch(images[start:start + batch_size], 'cifar'))
            h = model.classifier[:-1](model.features(x))
            g = model.classifier[-1](h)
            hidden.append(h.numpy().copy()); raw.append(g.numpy().copy())
    z, h = np.concatenate(raw), np.concatenate(hidden)
    if z.shape != (len(images), 10) or h.shape != (len(images), 64) or not np.isfinite(z).all() or not np.isfinite(h).all():
        raise ValueError('finite original SmallCNN raw/64d feature output required')
    return z, h, {'feature_forward_images': len(images), 'generic_forward_images': len(images),
        'feature_forward_batches': len(raw), 'generic_forward_batches': len(raw),
        'cpu_forward_seconds': time.perf_counter() - started, 'batch_size': batch_size}


def topk_scores(raw, state):
    z = np.asarray(raw, dtype=np.float64)
    a = {name: value.astype(np.float64) for name, value in state.items()}
    return {'raw': z, 'prior_only': np.broadcast_to(a['analytic_prior'], z.shape),
        'analytic_prior': z + a['analytic_prior'], 'prioranchor_bias': z + a['prioranchor_bias'],
        'zero_anchor_vector': z * a['zero_anchor_vector'][:10] + a['zero_anchor_vector'][10:],
        'prioranchored_vector': z * a['prioranchored_vector'][:10] + a['prioranchored_vector'][10:]}


def summarize(clients, variants):
    if len(clients) != 20 or [row['client'] for row in clients] != list(range(20)):
        raise ValueError('all20 ordered clients required')
    weights = np.array([row['test_rows'] for row in clients], dtype=np.float64)
    if np.any(weights <= 0) or weights.sum() != 10000:
        raise ValueError('exact10000 TEST rows and twenty nonempty clients required')
    result = {}
    for variant in variants:
        rows = [row['metrics'][variant] for row in clients]
        result[variant] = {prefix + name: float(np.mean(values) if prefix == 'macro_' else np.average(values, weights=weights))
            for name in s.METRICS for values in [[row[name] for row in rows]] for prefix in ('macro_', 'weighted_')}
        result[variant]['client_accuracy_q10'] = float(np.quantile([row['accuracy'] for row in rows], .1))
        confidence = np.sum([row['ece_bins']['confidence_sum'] for row in rows], axis=0)
        correct = np.sum([row['ece_bins']['correct_sum'] for row in rows], axis=0)
        result[variant]['pooled_ece15'] = float(np.abs(correct - confidence).sum() / 10000)
    return result


def export_bundle(bundle, out, images, labels, groups, assignment_hash):
    method, seed = bundle['method'], bundle['seed']
    folder = out / method / f's{seed}'
    folder.mkdir(parents=True, exist_ok=False)
    z, features, cost = forward_cpu(bundle['model'], images)
    clients, files = [], []
    transform_seconds = metric_seconds = export_seconds = 0.
    variants = FEDROD_VARIANTS if method == FEDROD else TOPK_VARIANTS
    for client, ids in enumerate(groups):
        started = time.perf_counter()
        arrays = {'test_ids': ids, 'test_labels': labels[ids], 'raw_logits': z[ids]}
        if method == FEDROD:
            with torch.inference_mode():
                local = bundle['heads'][client](torch.from_numpy(features[ids])).numpy().copy()
            combined = z[ids] + local  # FP32 addition matches the original frozen predictor.
            scores = {'generic_raw': z[ids].astype(np.float64), 'personal_residual': local.astype(np.float64),
                      'generic_plus_local_residual': combined.astype(np.float64)}
            arrays.update(personal_residual_logits=local, generic_plus_local_residual_logits=combined,
                persistent_head_weight=bundle['heads'][client].weight.numpy(),
                persistent_head_bias=bundle['heads'][client].bias.numpy())
        else:
            scores = topk_scores(z[ids], bundle['states'][client])
            arrays.update({'state_' + key: value for key, value in bundle['states'][client].items()})
        transform_seconds += time.perf_counter() - started
        started = time.perf_counter()
        observed = {name: s.metrics(value, labels[ids]) for name, value in scores.items()}
        metric_seconds += time.perf_counter() - started
        started = time.perf_counter()
        target = folder / f'client{client}.npz'
        np.savez_compressed(target, **arrays, **{'score_' + key: value for key, value in scores.items()})
        files.append({'path': str(target.relative_to(out)), 'sha256': d.file_sha(target), 'bytes': target.stat().st_size})
        export_seconds += time.perf_counter() - started
        clients.append({'client': client, 'test_rows': len(ids), 'metrics': observed})
    cost.update(new_fit_calls=0, new_cnn_backwards=0, new_training_rounds=0,
        residual_forward_images=10000 if method == FEDROD else 0,
        transform_seconds=transform_seconds, metric_seconds=metric_seconds, export_seconds=export_seconds,
        npz_bytes=sum(row['bytes'] for row in files))
    return {'status': 'DONE', 'task': 'cifar', 'method': method, 'seed': seed, 'partition_seed': PARTITION,
        'rounds': 20, 'assignment_hash': assignment_hash, 'fixed_variants': list(variants),
        'summaries': summarize(clients, variants), 'clients': clients, 'exports': files, 'cost': cost,
        'scope': 'locked final TEST references; personal_residual alone is a diagnostic, not FedRoD final prediction'}


def three_seed_summary(reports, method):
    selected = sorted((row for row in reports if row['method'] == method), key=lambda row: row['seed'])
    if ([row['seed'] for row in selected] != list(SEEDS)
            or len({row['assignment_hash'] for row in selected}) != 1):
        raise ValueError('three fixed seeds with identical assignment required')
    return {variant: {name: {'mean': float(np.mean(values)), 'sample_sd': float(np.std(values, ddof=1))}
        for name in selected[0]['summaries'][variant]
        for values in [[row['summaries'][variant][name] for row in selected]]}
        for variant in selected[0]['fixed_variants']}


def execute(out, raw, root_lock):
    out = Path(out).resolve()
    inputs, bundles, snapshot, snapshot_sha = prepare_snapshot(out, create=False)
    lock = inputs.json(root_lock)
    authorization = d.validate_root_lock(root_lock, True, 'cifar')
    if (lock.get('reference_input_snapshot_sha256') != snapshot_sha
            or lock.get('source_sha256', {}).get('reference_inference.py') != d.file_sha(__file__)
            or authorization['root_lock_sha256'] != inputs.files[str(Path(root_lock).resolve())]):
        raise ValueError('root-lock must bind this reference driver and exact preflight snapshot')
    inputs.read(out / 'PRETEST_SNAPSHOT.json', snapshot_sha)
    inputs.read(out / 'FROZEN_STATES.npz', snapshot['frozen_states_sha256'])
    inputs.verify()
    started = time.perf_counter()
    images, labels, provenance = d.load_cifar_test(raw, root_lock=root_lock, execute=True)
    loading_seconds = time.perf_counter() - started
    started = time.perf_counter()
    groups, assignment = d.assign_clients(labels, bundles[0]['counts'], task='cifar',
        train_manifest_sha256=snapshot['train_manifest_sha256'])
    assignment_seconds = time.perf_counter() - started
    write_json(out / 'assignment.json', assignment)
    write_json(out / 'test_provenance.json', provenance)
    reports = []
    for bundle in bundles:
        before = {key: d.array_sha(value.numpy(), '<f4') for key, value in bundle['model'].state_dict().items()}
        report = export_bundle(bundle, out, images, labels, groups, assignment['assignment_hash'])
        if before != {key: d.array_sha(value.numpy(), '<f4') for key, value in bundle['model'].state_dict().items()}:
            raise ValueError('inference modified frozen shared model')
        inputs.verify()
        write_json(out / bundle['method'] / f"s{bundle['seed']}" / 'report.json', report)
        reports.append(report)
    inputs.verify()
    result = {'status': 'DONE', 'scope': 'LOCKED_OFFICIAL_CIFAR_REFERENCE_FINAL_TEST',
        'seeds': list(SEEDS), 'methods': [*TOPK, FEDROD], 'partition_seed': PARTITION,
        'assignment_hash': assignment['assignment_hash'], 'assignment_manifest_hash': assignment['manifest_hash'],
        'pretest_snapshot_sha256': snapshot_sha, 'input_sha256_before_after_equal': inputs.files,
        'test_provenance': provenance, 'authorization': authorization,
        'reports': reports, 'summaries': {method: three_seed_summary(reports, method) for method in (*TOPK, FEDROD)},
        'sd_definition': 'sample SD, ddof=1, exactly three seeds', 'new_fit_calls': 0, 'new_cnn_backwards': 0,
        'test_loading_seconds': loading_seconds, 'assignment_seconds': assignment_seconds,
        'score_arithmetic': 'frozen FP32 TopK states promoted to FP64; FedRoD FP32 g+P promoted after addition',
        'judgment': 'descriptive fixed references; no new gate, selection, training or original-decision change',
        'test_protocol_scope': 'TRAIN-count-defined label-skew TEST assignment; not official benchmark client partition',
        'review': 'REVIEW_UNAVAILABLE'}
    write_json(out / 'result.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--raw', type=Path)
    parser.add_argument('--root-lock', type=Path)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.execute:
        if args.raw is None or args.root_lock is None:
            parser.error('--execute requires --raw and --root-lock')
        report = execute(args.out, args.raw, args.root_lock)
        print(json.dumps({'status': report['status'], 'assignment_hash': report['assignment_hash']}))
    else:
        _, _, snapshot, digest = prepare_snapshot(args.out, create=True)
        print(json.dumps({'status': 'PREFLIGHT_ONLY', 'reference_input_snapshot_sha256': digest,
            'models': 9, 'seeds': snapshot['seeds'], 'official_test_byte_reads': False,
            'image_forwards': 0, 'fit_calls': 0, 'backward_calls': 0}))


if __name__ == '__main__':
    main()
