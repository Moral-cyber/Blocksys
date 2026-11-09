"""Exploratory new-split training entry; frozen FedLC loss and uniform3 EF."""
from pathlib import Path
import argparse
import json
import time
import numpy as np
import torch
import common as c


def run(args, method, trainsets, validation, manifest, data_identities):
    if method not in c.METHODS or args.rounds < 1:
        raise ValueError('unsupported method or rounds')
    torch.manual_seed(args.seed)
    model = c.runner.SmallCNN()
    anchor = c.runner.vectorize(model)
    if len(anchor) != c.PARAMETERS:
        raise ValueError('unexpected parameter layout')
    weights = np.asarray([len(dataset) for dataset in trainsets], dtype=np.float64)
    if not len(weights) or np.any(weights <= 0):
        raise ValueError('nonempty full participants required')
    weights /= weights.sum()
    sources = c.source_identities()
    config = {'version': c.VERSION, 'method': method, 'seed': args.seed,
        'partition_seed': args.partition_seed, 'rounds': args.rounds, 'backend': args.backend,
        'source_identities': sources, 'data_identities': data_identities,
        'runtime': c.accelerator.runtime_metadata(args.backend, 'exploratory_calibrated_lowbit_confirm'),
        'data_manifest': manifest, 'weights': weights.tolist(), 'parameters': c.PARAMETERS,
        'batch': 64, 'lr': .03, 'local_epochs': 1,
        'local_train_source': 'frozen balanced_quant.train; same batch seed=seed+100r+i',
        'loss': 'ce' if method.startswith('ce_') else 'fedlc',
        'loss_scope': 'FedLC-inspired explicit tau1 variant, not official reproduction',
        'upload_cap_bytes': None if method.endswith('dense') else c.CAP,
        'communication': 'actual generated decoded uniform3+EF uploads; dense/all downlinks analytical',
        'selection': 'old split informed new exploratory FedLC+uniform3+prior-anchor-bias choice; balanced STOP unchanged',
        'official_test_byte_read': False, 'official_test_inference': False,
        'evaluation': 'per-round fixed TRAIN-holdout globaldev; final checkpoint only for PFL confirmation'}
    config['signature'] = c.signature(config)
    out = args.out / method
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'config.json').exists():
        prior = json.loads((out / 'config.json').read_text())
        digest = prior.pop('signature', None)
        if digest != c.signature(prior) or digest != config['signature']:
            raise ValueError('incompatible output source/config')
    residuals, rows = [torch.zeros_like(anchor) for _ in trainsets], []
    if (out / 'checkpoint.pt').exists():
        checkpoint = torch.load(out / 'checkpoint.pt', weights_only=False)
        model = c.validate_checkpoint(checkpoint, config)
        residuals, rows = checkpoint['residuals'], checkpoint['rows']
    c.atomic_json(out / 'config.json', config)
    for round_id in range(len(rows), args.rounds):
        began = time.perf_counter()
        anchor = c.runner.vectorize(model).clone()
        aggregate, clients = torch.zeros_like(anchor), []
        for client_id, dataset in enumerate(trainsets):
            seed = args.seed + 100 * round_id + client_id
            delta, counters = c.balanced.train(model, dataset, seed, args.backend, config['loss'])
            if not torch.equal(c.runner.vectorize(model), anchor):
                raise ValueError('client training altered global anchor')
            if method.endswith('dense'):
                q = delta
                residuals[client_id].zero_()
                wire_bytes, ef_error = 64 + 4 * c.PARAMETERS, 0.
            else:
                candidate = delta + residuals[client_id]
                encoded = c.codec.encode(candidate, c.CAP, mode='uniform3')
                q = c.codec.decode(encoded.wire)
                if not torch.equal(q, encoded.sent) or len(encoded.wire) > c.CAP:
                    raise ValueError('uniform3 decoder or cap failure')
                residuals[client_id] = candidate - q
                wire_bytes = len(encoded.wire)
                ef_error = float((q + residuals[client_id] - candidate).abs().max())
                if round_id == 0 and client_id == 0:
                    (out / 'first_payload.bin').write_bytes(encoded.wire)
            aggregate.add_(q, alpha=float(weights[client_id]))
            clients.append({'client': client_id, 'seed': seed, **counters,
                'upload_bytes': wire_bytes, 'upload_accounting': 'analytical dense' if method.endswith('dense') else 'actual decoded wire',
                'ef_max_error': ef_error, 'residual_norm': float(residuals[client_id].norm())})
        c.runner.assign_vector(model, anchor + aggregate)
        metrics = c.runner.metrics(model, validation)
        rows.append({'round': round_id + 1, 'validation': metrics, 'clients': clients,
            'upload_bytes': sum(client['upload_bytes'] for client in clients),
            'download_bytes_analytical': len(trainsets) * (64 + 4 * c.PARAMETERS),
            'actual_gradient_evaluations': sum(client['gradient_evaluations'] for client in clients),
            'global_anchor_unchanged_between_clients': True, 'seconds': time.perf_counter() - began})
        temporary = out / 'checkpoint.tmp'
        torch.save({'version': c.VERSION, 'signature': config['signature'], 'config': config,
            'model': model.state_dict(), 'residuals': residuals, 'rows': rows}, temporary)
        temporary.replace(out / 'checkpoint.pt')
        c.atomic_json(out / 'result.json', {'status': 'DONE' if len(rows) == args.rounds else 'RUNNING',
            'rows': rows, 'signature': config['signature'], 'official_test_byte_read': False})
        print(json.dumps({'method': method, 'round': round_id + 1, 'accuracy': metrics['accuracy'],
                          'ce': metrics['loss'], 'seconds': rows[-1]['seconds']}), flush=True)
    c.verify_identities({**sources, **data_identities})
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=20261211)
    parser.add_argument('--partition-seed', type=int, default=20261203)
    parser.add_argument('--rounds', type=int, default=20)
    parser.add_argument('--backend', choices=('mps', 'cpu'), default='mps')
    args = parser.parse_args()
    if args.rounds < 1:
        parser.error('rounds must be positive')
    attempts = c.install_test_guard()
    torch.set_num_threads(1)
    if args.backend == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS unavailable; no fallback')
    trainsets, validation, manifest, identities, _, _ = c.prepare(args.partition_seed)
    if len(trainsets) != 20:
        raise ValueError('fixed plan requires twenty clients')
    for method in c.METHODS:
        run(args, method, trainsets, validation, manifest, identities)
    if attempts:
        raise ValueError('forbidden official TEST byte access attempted')


if __name__ == '__main__':
    main()
