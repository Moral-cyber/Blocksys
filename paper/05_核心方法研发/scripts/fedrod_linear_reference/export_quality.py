"""Final-only FedRoD G / G+P logits and fixed predictive-quality export.

No TRAIN fitting, gradients or official TEST. The formal CLI uses the strict
five-TRAIN loader only after signed final checkpoint/source preflight passes.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import sys
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
import prospective as p
import reference as r

QUALITY_PATH = r.ROOT / 'work/calibration_baselines/score_quality.py'
QUALITY_SHA = '5f14f01a25f64c9ed495214aa57f4277370f315ce08b7686b985faa03666fc98'
PINS = {
    'reference.py': '3649edec016ac3b06005a624bbf2dcea22d23965d53637b8ec36875b5be74dfc',
    'train.py': '7ff112e2e3f5575835d1c0358e501a4815231fb6572be0ef33b148dc05eaac70',
    'prospective.py': '9e7325bd8de7c6d3cd32a2dbf060521d1e073d1c1fc86520411fb34e6b7ef501',
    'PLAN.md': 'c5817d080e4ac55ba9a3ea8a4739b1885ba33a84a26be5056e97aed51eacd895',
    'SOURCE_LOCK.json': '2303784cf1288bdc3798577049b39d12acb99b87eed5c5eebfc857aab15df904'}
METHOD = 'fedrod_linear_uniform3'


def own_sources():
    p.verify_sources()
    for name, expected in PINS.items():
        if r.sha(r.HERE / name) != expected:
            raise ValueError('frozen FedRoD source changed: ' + name)
    if r.sha(QUALITY_PATH) != QUALITY_SHA:
        raise ValueError('frozen probability-quality helper changed')
    return {str((r.HERE / name).resolve()): expected for name, expected in PINS.items()} | {
        str(QUALITY_PATH): QUALITY_SHA, str(Path(__file__).resolve()): r.sha(__file__),
        str(r.ROOT / p.PLAN_RELATIVE): p.PLAN_SHA256}


def load_quality():
    own_sources()
    spec = importlib.util.spec_from_file_location('_fedrod_fixed_quality', QUALITY_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


quality = load_quality()


def checked_score(logits, labels):
    # score_quality casts labels internally: reject non-integer GT before it.
    labels = np.asarray(labels)
    if labels.shape != (len(logits),) or labels.dtype.kind not in 'iu' or np.any((labels < 0) | (labels >= 10)):
        raise ValueError('one-dimensional integer dataset GT in[0,9] required')
    return quality.score(logits, labels)


def install_guard():
    attempts = []
    def guard(event, arguments):
        if event == 'open' and isinstance(arguments[0], (str, bytes)):
            name = str(arguments[0])
            if 'test_batch' in name or 't10k' in name:
                attempts.append(name)
                raise ValueError('official TEST bytes forbidden')
    sys.addaudithook(guard)
    return attempts


def state_hash(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def preflight(model_root):
    model_root = Path(model_root).resolve()
    protected = own_sources()
    paths = [model_root / name for name in ('config.json', 'checkpoint.pt', 'result.json')]
    protected.update({str(path): r.sha(path) for path in paths})
    config = json.loads(paths[0].read_text())
    base = config['base']
    if base['partition_seed'] != p.PARTITION_SEED or base['seed'] not in p.TRAINING_SEEDS or base['rounds'] != p.ROUNDS:
        raise ValueError('prospective p20261217 / declared seed / final20 required')
    entry = p.load_frozen_entry()
    with p.module_overrides(entry, base['seed'], base['partition_seed']):
        expected = p.base_config(entry, base['backend'], base['seed'], base['partition_seed'])
    r.validate_config(config, expected)
    r.verify_identities(base['source_identities'])
    protected.update({str((r.ROOT / path).resolve()): expected for path, expected in base['source_identities'].items()})
    for path, expected in config['train_adapter_identities'].items():
        if r.sha(path) != expected:
            raise ValueError('frozen strict TRAIN adapter identity changed')
        protected[str(Path(path).resolve())] = expected
    checkpoint = torch.load(paths[1], map_location='cpu', weights_only=True)
    model = r.validate_checkpoint(checkpoint, config)
    training = json.loads(paths[2].read_text())
    if (len(checkpoint['rows']) != base['rounds'] or training.get('status') != 'DONE'
            or training.get('signature') != config['signature'] or training.get('rows') != checkpoint['rows']):
        raise ValueError('only complete final checkpoint with matching DONE history allowed')
    manifest = config['data_manifest']
    sizes = list(map(len, manifest['train_indices']))
    if (manifest['partition_seed'] != base['partition_seed'] or len(sizes) != r.CLIENTS
            or sizes != manifest['client_sizes'] or any(size <= 0 for size in sizes)
            or not np.allclose(config['weights'], np.asarray(sizes) / sum(sizes), rtol=0, atol=1e-15)):
        raise ValueError('actual partition/count/weights mismatch')
    return {'config': config, 'checkpoint': checkpoint, 'model': model,
            'protected': protected, 'entry': entry, 'model_root': model_root}


def forward(model, head, dataset):
    model.eval()
    head.eval()
    generic, residual, personal, labels = [], [], [], []
    batches, seen = 0, 0
    began = time.perf_counter()
    with torch.inference_mode():
        for x, y in DataLoader(dataset, batch_size=256, shuffle=False, num_workers=0):
            if x.device.type != 'cpu' or y.device.type != 'cpu':
                raise ValueError('fixed CPU final export required')
            h = r.features(model, x)
            g, local = model.classifier[-1](h), head(h)
            combined = g + local
            generic.append(g.numpy().copy())
            residual.append(local.numpy().copy())
            personal.append(combined.numpy().copy())
            labels.append(y.numpy().copy())
            batches += 1
            seen += len(y)
    return {'generic_raw': np.concatenate(generic), 'personal_residual': np.concatenate(residual),
        'generic_plus_local_residual': np.concatenate(personal), 'labels': np.concatenate(labels)}, {
        'feature_forward_images': seen, 'feature_forward_batches': batches,
        'generic_forward_images': seen, 'generic_forward_batches': batches,
        'residual_forward_images': seen, 'residual_forward_batches': batches,
        'seconds': time.perf_counter() - began}


def export(bundle, out, holdouts, manifest, data_identities, adapter_identities):
    out = Path(out).resolve()
    if out.exists():
        raise ValueError('fresh final export output required')
    config, checkpoint, model = bundle['config'], bundle['checkpoint'], bundle['model']
    if (manifest != config['data_manifest'] or data_identities != config['data_identities']
            or adapter_identities != config['train_adapter_identities'] or len(holdouts) != r.CLIENTS):
        raise ValueError('strict loader data/source/partition differs from signed training')
    all_ids = [int(i) for ids in manifest['holdout_indices'] for i in ids]
    train_ids = [int(i) for ids in manifest['train_indices'] for i in ids]
    if len(set(all_ids)) != len(all_ids) or set(all_ids) & set(train_ids):
        raise ValueError('holdout IDs overlap or repeat')
    model_before = state_hash(model)
    heads = []
    for state in checkpoint['heads']:
        head = r.new_head()
        head.load_state_dict(state)
        heads.append(head)
    head_before = [state_hash(head) for head in heads]
    began, clients = time.perf_counter(), []
    out.mkdir(parents=True)
    for client, (head, dataset) in enumerate(zip(heads, holdouts)):
        if not len(dataset) or not np.array_equal(dataset.indices, manifest['holdout_indices'][client]):
            raise ValueError('exact nonempty holdout ID order required')
        values, cost = forward(model, head, dataset)
        labels = values['labels']
        if (not np.array_equal(labels, dataset.labels[dataset.indices]) or cost['feature_forward_images'] != len(dataset)
                or not np.array_equal(values['generic_plus_local_residual'], values['generic_raw'] + values['personal_residual'])):
            raise ValueError('GT order/forward count or exact G+P sum mismatch')
        metrics = {name: checked_score(values[name], labels) for name in ('generic_raw', 'generic_plus_local_residual')}
        np.savez_compressed(out / f'{METHOD}_client{client}.npz',
            holdout_ids=np.asarray(dataset.indices, dtype=np.uint32), **values)
        clients.append({'client': client, 'holdout_samples': len(dataset), 'metrics': metrics,
            'forward_cost': cost, 'head_state_sha256': head_before[client]})
    unchanged_models = state_hash(model) == model_before and [state_hash(head) for head in heads] == head_before
    if not unchanged_models:
        raise ValueError('inference changed model/head state')
    protected = {**bundle['protected'], **{str(Path(path).resolve()): expected for path, expected in data_identities.items()}}
    if any(r.sha(path) != expected for path, expected in protected.items()):
        raise ValueError('source/config/checkpoint/data changed during final export')
    summaries = quality.summarize(clients)
    for variant in summaries:
        summaries[variant]['client_accuracy_q10'] = float(np.quantile([client['metrics'][variant]['accuracy'] for client in clients], .1))
    report = {'status': 'DONE', 'purpose': 'final-only FedRoD predictive-quality export; descriptive reference, no new gate',
        'results': {METHOD: {'clients': clients, 'summaries': summaries}},
        'model_root': str(bundle['model_root']), 'config_signature': config['signature'],
        'training_seed': config['base']['seed'], 'partition_seed': config['base']['partition_seed'],
        'training_rounds': config['base']['rounds'], 'inference_backend': 'cpu',
        'source_config_data_identities_before_after': protected, 'identities_before_after_equal': True,
        'global_and_personal_states_unchanged': True, 'generic_model_sha256': model_before,
        'actual_feature_forward_images': sum(client['forward_cost']['feature_forward_images'] for client in clients),
        'actual_feature_forward_batches': sum(client['forward_cost']['feature_forward_batches'] for client in clients),
        'actual_generic_forward_images': sum(client['forward_cost']['generic_forward_images'] for client in clients),
        'actual_residual_forward_images': sum(client['forward_cost']['residual_forward_images'] for client in clients),
        'TRAIN_image_forwards': 0, 'TRAIN_fits': 0, 'new_cnn_backwards': 0, 'optimizer_steps': 0,
        'extra_upload_bytes': 0, 'holdout_samples_actual': len(all_ids),
        'official_test_byte_reads': False, 'official_test_inference': False,
        'metric_definitions': {'nll': 'categorical natural-log loss', 'brier_sum_classes': 'sum of ten squared probability-onehot errors, unnormalized [0,2]',
            'ece_15_equal_width': 'fixed15 bins [k/15,(k+1)/15), final includes1; empty bins contribute0',
            'macro': 'mean over20 clients', 'weighted': 'holdout sample-count weighted client metric',
            'pooled_ece': 'merge signed bin confidence/correct sums across clients before absolute value'},
        'cost_scope': 'new export feature/generic/residual forwards counted; one shared feature pass produces both G and G+P. This is additional to any training-entry final evaluation; no TRAIN-logit fit/cache or gradients.',
        'previous_decisions': 'unchanged; lower NLL does not automatically imply lower ECE or Brier',
        'export_seconds': time.perf_counter() - began, 'review': 'REVIEW_UNAVAILABLE'}
    r.atomic_json(out / 'result.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--model-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    attempts = install_guard()
    torch.set_num_threads(1)
    bundle = preflight(args.model_root)
    adapter = bundle['entry'].load_train_adapter()
    _, _, manifest, identities, images, labels = adapter.prepare(bundle['config']['base']['partition_seed'])
    holdouts = [adapter.partitions.IndexedCifar10(images, labels, ids) for ids in manifest['holdout_indices']]
    report = export(bundle, args.out, holdouts, manifest, identities, adapter.source_identities())
    if attempts:
        raise ValueError('official TEST byte access attempted')
    print(json.dumps({'status': report['status'], 'training_seed': report['training_seed'],
        'feature_forward_images': report['actual_feature_forward_images'],
        'summaries': report['results'][METHOD]['summaries']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
