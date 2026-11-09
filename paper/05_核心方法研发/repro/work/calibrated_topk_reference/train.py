"""Fixed CE/FedLC TopK+EF reference; never a new candidate or GO decision."""
from pathlib import Path
import argparse
import json
import time
import numpy as np
import torch
import common as c
import wire


def run(args, method, trainsets, validation, manifest, data_identities):
    if method not in c.METHODS or args.rounds < 1:
        raise ValueError('unsupported method or rounds')
    torch.manual_seed(args.seed)
    model = c.runner.SmallCNN()
    anchor = c.runner.vectorize(model)
    if anchor.numel() != c.PARAMETERS:
        raise ValueError('unexpected parameter layout')
    weights = np.asarray([len(dataset) for dataset in trainsets], dtype=np.float64)
    if not len(weights) or np.any(weights <= 0):
        raise ValueError('positive actual TRAIN weights required')
    weights /= weights.sum()
    sources = c.source_identities()
    config = {'version': c.VERSION, 'method': method, 'seed': args.seed,
        'partition_seed': args.partition_seed, 'rounds': args.rounds, 'backend': args.backend,
        'source_identities': sources, 'data_identities': data_identities,
        'runtime': c.accelerator.runtime_metadata(args.backend, 'strict_topk_reference'),
        'data_manifest': manifest, 'weights': weights.tolist(), 'parameters': c.PARAMETERS,
        'batch': 64, 'lr': .03, 'local_epochs': 1, 'loss': 'ce' if method.startswith('ce_') else 'fedlc',
        'local_train_source': 'frozen balanced_quant.train; same batch seed=seed+100r+i',
        'loss_scope': 'FedLC-inspired explicit tau1 variant, not official reproduction',
        'upload_cap_bytes': c.CAP, 'header_bytes': wire.HEADER.size, 'selected_pairs': c.K,
        'pair_bytes': 8, 'selection': 'descending abs(delta+residual); ties lower index first; sorted ascending wire indices',
        'communication': 'actual decoded 64B header+uint32LE/FP32LE pairs; all global downlinks analytical; no network/authentication',
        'reference_only': True, 'new_go_criterion': False,
        'official_test_byte_read': False, 'official_test_inference': False,
        'evaluation': 'per-round fixed TRAIN-holdout globaldev; final-only frozen local prior/bias/head wrapper'}
    config['signature'] = c.signature(config)
    out = args.out / method
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'config.json').exists():
        previous = json.loads((out / 'config.json').read_text())
        signed = previous.pop('signature', None)
        if signed != c.signature(previous) or signed != config['signature']:
            raise ValueError('incompatible output source/config')
    residuals, rows = [torch.zeros_like(anchor) for _ in trainsets], []
    if (out / 'checkpoint.pt').exists():
        checkpoint = torch.load(out / 'checkpoint.pt', weights_only=True)
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
                raise ValueError('client altered global anchor')
            candidate = delta + residuals[client_id]
            encoded = wire.encode(candidate, c.CAP)
            q = wire.decode(encoded.wire)
            if not torch.equal(q, encoded.sent) or q.shape != anchor.shape or len(encoded.wire) != c.CAP:
                raise ValueError('TopK wire/decoder/accounting failure')
            residuals[client_id] = candidate - q
            if round_id == 0 and client_id == 0:
                (out / 'first_payload.bin').write_bytes(encoded.wire)
            aggregate.add_(q, alpha=float(weights[client_id]))
            clients.append({'client': client_id, 'seed': seed, **counters,
                'upload_bytes': len(encoded.wire), 'upload_accounting': 'actual decoded wire',
                'selected_pairs': encoded.indices.numel(),
                'ef_max_error': float((q + residuals[client_id] - candidate).abs().max()),
                'residual_norm': float(residuals[client_id].norm())})
        c.runner.assign_vector(model, anchor + aggregate)
        metrics = c.runner.metrics(model, validation)
        rows.append({'round': round_id + 1, 'validation': metrics, 'clients': clients,
            'upload_bytes': sum(client['upload_bytes'] for client in clients),
            'download_bytes_analytical': len(trainsets) * 547560,
            'actual_gradient_evaluations': sum(client['gradient_evaluations'] for client in clients),
            'global_anchor_unchanged_between_clients': True, 'seconds': time.perf_counter() - began})
        temporary = out / 'checkpoint.tmp'
        torch.save({'version': c.VERSION, 'signature': config['signature'], 'config': config,
            'model': model.state_dict(), 'residuals': residuals, 'rows': rows}, temporary)
        temporary.replace(out / 'checkpoint.pt')
        c.atomic_json(out / 'result.json', {'status': 'DONE' if len(rows) == args.rounds else 'RUNNING',
            'rows': rows, 'signature': config['signature'], 'reference_only': True, 'official_test_byte_read': False})
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
    if (args.seed, args.partition_seed, args.rounds) != (20261211, 20261203, 20):
        parser.error('fixed reference requires seed20261211/partition20261203/rounds20')
    attempts = c.install_test_guard()
    torch.set_num_threads(1)
    if args.backend == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS unavailable; no fallback')
    trainsets, validation, manifest, identities, _, _ = c.prepare(args.partition_seed)
    if len(trainsets) != 20:
        raise ValueError('fixed reference requires20 clients')
    for method in c.METHODS:
        run(args, method, trainsets, validation, manifest, identities)
    if attempts:
        raise ValueError('official TEST access attempted')


if __name__ == '__main__':
    main()
