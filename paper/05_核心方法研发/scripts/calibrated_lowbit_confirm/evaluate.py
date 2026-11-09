"""Final-checkpoint clientholdout confirmation with fixed TRAIN-only fitting."""
from pathlib import Path
import argparse
import copy
import hashlib
import json
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import common as c


def forward(model, dataset):
    model.eval()
    logits, labels, batches = [], [], 0
    began = time.perf_counter()
    with torch.inference_mode():
        for x, y in DataLoader(dataset, batch_size=256, shuffle=False):
            logits.append(model(x).numpy().copy())
            labels.append(y.numpy().copy())
            batches += 1
    return np.concatenate(logits), np.concatenate(labels), {
        'image_forwards': len(dataset), 'batch_calls': batches,
        'seconds': time.perf_counter() - began}


def model_hash(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def matched_head(model, dataset, seed):
    adapted = copy.deepcopy(model).train()
    adapted.requires_grad_(False)
    adapted.classifier[-1].requires_grad_(True)
    torch.manual_seed(seed)
    loader = DataLoader(dataset, batch_size=64, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    optimizer = torch.optim.SGD(adapted.classifier[-1].parameters(), lr=.03)
    before_body = torch.cat([parameter.detach().flatten() for parameter in list(adapted.parameters())[:-2]]).clone()
    began, steps = time.perf_counter(), 0
    for x, y in loader:
        optimizer.zero_grad(set_to_none=True)
        nn.functional.cross_entropy(adapted(x), y).backward()
        optimizer.step()
        steps += 1
    after_body = torch.cat([parameter.detach().flatten() for parameter in list(adapted.parameters())[:-2]])
    if not torch.equal(before_body, after_body):
        raise ValueError('matched head changed frozen body')
    return adapted.eval(), {'gradient_evaluations': steps, 'optimizer_steps': steps,
        'train_image_forwards': len(dataset), 'seconds': time.perf_counter() - began,
        'state_bytes': 4 * 650, 'seed': seed}


def evaluate_model(model, training_sets, holdouts, method, seed, out, manifest):
    before = model_hash(model)
    clients = []
    for client, (training, holdout) in enumerate(zip(training_sets, holdouts)):
        counts = np.bincount(training.labels[training.indices], minlength=10)
        if counts.tolist() != manifest['client_label_counts'][client]:
            raise ValueError('TRAIN counts differ from frozen partition')
        train_logits, train_labels, cost = forward(model, training)
        if not np.array_equal(train_labels, training.labels[training.indices]):
            raise ValueError('TRAIN forward ID order differs')
        began = time.perf_counter()
        bias, optimization = c.bias_solver.fit(train_logits, train_labels)
        fit_seconds = time.perf_counter() - began
        prior = np.log((counts + 1.) / (len(training) + 10)).astype(np.float32)
        # Fitting is complete before any holdout forward or label scoring.
        raw, labels, raw_cost = forward(model, holdout)
        adapted, adaptation_cost = matched_head(model, training, seed + 100000 + client)
        learned, learned_labels, learned_cost = forward(adapted, holdout)
        if not np.array_equal(labels, learned_labels) or model_hash(model) != before:
            raise ValueError('holdout order or frozen global model changed')
        raw64, learned64 = raw.astype(np.float64), learned.astype(np.float64)
        scores = {'raw': raw64, 'analytic_prior': raw64 + prior,
            'prioranchor_bias': raw64 + bias.astype(np.float64),
            'matched_learned_head': learned64,
            'matched_learned_head_plus_prior': learned64 + prior}
        metrics = {name: c.bias_solver.stats(value, labels) for name, value in scores.items()}
        np.savez_compressed(out / f'{method}_client{client}.npz',
            train_logits=train_logits, train_labels=train_labels,
            train_ids=np.asarray(training.indices, dtype=np.uint32),
            raw=raw, learned=learned, labels=labels,
            holdout_ids=np.asarray(holdout.indices, dtype=np.uint32),
            train_counts=counts, prior_shift=prior, fitted_bias=bias)
        clients.append({'client': client, 'train_samples': len(training),
            'holdout_samples': len(holdout), 'metrics': metrics,
            'bias_extra_train_forward': cost, 'bias_numpy_fit_seconds': fit_seconds,
            'bias_new_cnn_backward': 0, 'bias_local_state_bytes': bias.nbytes,
            'bias_extra_uplink_bytes': 0, 'bias_optimization': optimization,
            'matched_head_training': adaptation_cost,
            'raw_holdout_forward': raw_cost, 'learned_holdout_forward': learned_cost})
    return {'clients': clients, 'summaries': c.bias_solver.summarize(clients),
        'frozen_global_model_unchanged': True}


def gates(results):
    candidate = results['fedlc_uniform3']['summaries']['prioranchor_bias']
    prior = results['fedlc_uniform3']['summaries']['analytic_prior']
    ce = results['ce_uniform3']['summaries']['prioranchor_bias']
    dense = results['fedlc_dense']['summaries']['prioranchor_bias']
    strongest_head = max(('matched_learned_head', 'matched_learned_head_plus_prior'),
        key=lambda name: results['fedlc_uniform3']['summaries'][name]['macro_accuracy'])
    head = results['fedlc_uniform3']['summaries'][strongest_head]
    checks = {'same_loss_prior_accuracy_plus_half_pp': candidate['macro_accuracy'] >= prior['macro_accuracy'] + .005,
        'same_loss_prior_lower_ce': candidate['macro_loss'] < prior['macro_loss'],
        'ce_bias_accuracy_noninferior_half_pp': candidate['macro_accuracy'] >= ce['macro_accuracy'] - .005,
        'ce_bias_ce_within_001': candidate['macro_loss'] <= ce['macro_loss'] + .01,
        'matched_head_accuracy_noninferior_half_pp': candidate['macro_accuracy'] >= head['macro_accuracy'] - .005,
        'quant_dense_accuracy_noninferior_half_pp': candidate['macro_accuracy'] >= dense['macro_accuracy'] - .005}
    return checks, strongest_head


def run(model_root, out, images, labels, manifest, data_identities):
    if out.exists():
        raise ValueError('fresh evaluation output required')
    sources, inputs, loaded = c.source_identities(), {}, {}
    for method in c.METHODS:
        folder = model_root / method
        for name in ('config.json', 'checkpoint.pt'):
            path = folder / name
            inputs[str(path.resolve())] = c.sha(path)
        config = json.loads((folder / 'config.json').read_text())
        unsigned = {key: value for key, value in config.items() if key != 'signature'}
        if config['signature'] != c.signature(unsigned):
            raise ValueError('training config signature invalid')
        if (config['data_manifest'] != manifest or config['source_identities'] != sources
                or config['data_identities'] != data_identities):
            raise ValueError('evaluation source/data differs from frozen training')
        checkpoint = torch.load(folder / 'checkpoint.pt', weights_only=False)
        model = c.validate_checkpoint(checkpoint, config)
        if len(checkpoint['rows']) != config['rounds']:
            raise ValueError('evaluate only final complete checkpoint')
        loaded[method] = model, config
    configurations = [config for _, config in loaded.values()]
    if len({(config['seed'], config['partition_seed'], config['rounds'], config['backend'])
            for config in configurations}) != 1:
        raise ValueError('all three frozen methods must share protocol')
    out.mkdir(parents=True)
    training = [c.partitions.IndexedCifar10(images, labels, ids) for ids in manifest['train_indices']]
    holdouts = [c.partitions.IndexedCifar10(images, labels, ids) for ids in manifest['holdout_indices']]
    results = {}
    began = time.perf_counter()
    for method, (model, config) in loaded.items():
        results[method] = evaluate_model(model, training, holdouts, method, config['seed'], out, manifest)
        print(json.dumps({'method': method, 'summaries': results[method]['summaries']}), flush=True)
    checks, strongest_head = gates(results)
    c.verify_identities({**sources, **data_identities, **inputs})
    report = {'status': 'DONE', 'decision': 'FOLLOW_UP' if all(checks.values()) else 'STOP',
        'gates': checks, 'matched_head_strongest_variant': strongest_head,
        'candidate_fixed': 'fedlc_uniform3 + prioranchor_bias', 'results': results,
        'training_seed': configurations[0]['seed'], 'partition_seed': configurations[0]['partition_seed'],
        'training_rounds': configurations[0]['rounds'], 'source_identities': sources,
        'data_identities': data_identities, 'frozen_model_inputs': inputs,
        'identities_before_after_equal': True, 'official_test_byte_reads': False,
        'official_test_inference': False,
        'selection_disclosure': 'combination chosen after old-split results; balanced STOP unchanged; new split is exploratory',
        'must_next_if_passed': ['zero-anchor local bias, lambda10, implemented in a new baseline helper without changing frozen solver',
            'vector scaling / BCTS primary baseline', 'two fresh training seeds20261212/20261213',
            'second task', 'strong FedCal vector baseline', 'mechanism attribution and total-cost comparisons'],
        'hyperparameter_policy': 'frozen bias lambda10/Newton20/Armijo; no holdout parameter selection',
        'training_samples_actual': sum(map(len, manifest['train_indices'])),
        'holdout_samples_actual': sum(map(len, manifest['holdout_indices'])),
        'bias_total_train_image_forwards': sum(client['bias_extra_train_forward']['image_forwards']
            for result in results.values() for client in result['clients']),
        'bias_total_cnn_backward': 0, 'bias_state_bytes_per_client': 40, 'bias_extra_uplink_bytes': 0,
        'matched_head_total_gradient_evaluations': sum(client['matched_head_training']['gradient_evaluations']
            for result in results.values() for client in result['clients']),
        'communication_scope': 'training upload actual for quant, analytical for dense; all global downlinks analytical; no network measurement',
        'evaluation_seconds': time.perf_counter() - began}
    c.atomic_json(out / 'result.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--model-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    attempts = c.install_test_guard()
    torch.set_num_threads(1)
    # Read the signed training partition selection before strict TRAIN loading.
    first = json.loads((args.model_root / c.METHODS[0] / 'config.json').read_text())
    _, _, manifest, identities, images, labels = c.prepare(first['partition_seed'])
    report = run(args.model_root.resolve(), args.out.resolve(), images, labels, manifest, identities)
    if attempts:
        raise ValueError('official TEST byte access attempted')
    print(json.dumps({'decision': report['decision'], 'gates': report['gates']}), flush=True)


if __name__ == '__main__':
    main()
