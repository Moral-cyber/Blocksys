"""Independent FedRoD-linear entry. Default DESIGN_ONLY never loads raw data."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import platform
import sys
import time

sys.dont_write_bytecode = True
import torch
import reference as r

SEED = 20261211
PARTITION_SEED = 20261203
ROUNDS = 20


def base_config(backend):
    sources = r.source_identities()
    # Bind data-loader transitive sources before importing it or opening raw data.
    paths = list((r.ROOT / 'work/method_followup').glob('*.py'))
    paths += [r.ROOT / 'work/prior_bias/solver.py',
        r.ROOT / 'work/balanced_quant/development_p20261193_s20261201/ce_uniform3/config.json',
        r.ROOT / 'work/tinyproto_diag/tools/sparse_update_benchmark.py']
    for path in paths:
        sources[str(path.relative_to(r.ROOT))] = r.sha(path)
    return {'version': r.VERSION, 'method': 'fedrod_linear_uniform3', 'seed': SEED,
        'partition_seed': PARTITION_SEED, 'rounds': ROUNDS, 'clients': r.CLIENTS,
        'backend': backend, 'parameters': r.PARAMETERS, 'personal_parameters': r.HEAD_PARAMETERS,
        'shared_layout': r.state_layout(r.new_model(SEED)),
        'personal_layout': r.state_layout(r.new_head()),
        'source_identities': sources, 'batch': 64, 'lr': .03, 'local_epochs': 1,
        'momentum': 0., 'weight_decay': 0., 'upload_cap_bytes': r.CAP,
        'aggregation': 'full participation; fixed TRAIN-sample-count weights; actual decoded wire',
        'local_seed': 'seed + 100 * round_zero_based + client_id',
        'generic_loss': 'FedRoD Eq6 gamma1 BSM: CE(g+log(raw TRAIN counts),y); zero counts=-inf',
        'personal_loss': 'CE(g.detach()+P(feature.detach()),y)',
        'update': 'BRM+ERM; one shared forward, one combined backward and SGD step per batch',
        'personal_initialization': 'all zero; disclosed implementation choice, not fixed by paper',
        'personal_persistence': 'one 650-FP32 residual Linear per client; never aggregate/upload',
        'final_prediction': 'B.2 Eq13 global generic + persistent local residual logits',
        'paper_endpoint_note': 'main text local theta/psi endpoint differs; B.2 global endpoint adopted',
        'scope': 'plain linear, no hypernetwork or meta-BSM; not official hyperparameter reproduction',
        'communication': 'actual uniform3 wire+EF uplink; analytical downlink; no network/chain',
        'official_test_byte_read': False, 'official_test_inference': False,
        'evaluation': 'fixed TRAIN-holdout; generic_raw and generic_plus_local_residual',
        'runtime': {'python': platform.python_version(), 'torch': str(torch.__version__),
                    'platform': platform.platform(), 'backend': backend},
        'review': 'REVIEW_UNAVAILABLE / LOCAL_CHECK_ONLY'}


def preflight(out, base):
    """Reject bad resume state before the data-loader module is imported."""
    config_path, checkpoint_path = out / 'config.json', out / 'checkpoint.pt'
    if checkpoint_path.exists() and not config_path.exists():
        raise ValueError('checkpoint exists without config; refused before data loading')
    if not config_path.exists():
        return None, None
    config = json.loads(config_path.read_text())
    r.validate_config(config, base)
    checkpoint = None
    if checkpoint_path.exists():
        # Only local trusted experiment checkpoints; no external model downloads.
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        r.validate_checkpoint(checkpoint, config)
    return config, checkpoint


def load_train_adapter():
    name = '_fedrod_linear_train_only_adapter'
    spec = importlib.util.spec_from_file_location(name, r.ROOT / 'work/calibrated_lowbit_confirm/common.py')
    adapter = importlib.util.module_from_spec(spec)
    sys.modules[name] = adapter
    spec.loader.exec_module(adapter)
    return adapter


def final_summary(client_metrics):
    result = {}
    for name in ('generic_raw', 'generic_plus_local_residual'):
        rows = [item[name] for item in client_metrics]
        accuracies = torch.tensor([item['accuracy'] for item in rows], dtype=torch.float64)
        total = sum(item['samples'] for item in rows)
        result[name] = {'client_macro_accuracy': float(accuracies.mean()),
            'client_q10_accuracy': float(torch.quantile(accuracies, .1)),
            'sample_weighted_accuracy': sum(v['samples'] * v['accuracy'] for v in rows) / total,
            'client_macro_ce': sum(v['ce'] for v in rows) / len(rows),
            'sample_weighted_ce': sum(v['samples'] * v['ce'] for v in rows) / total,
            'holdout_samples': total}
    return result


def formal(args, base, saved_config, checkpoint):
    # This function is unreachable without the explicit formal switch.
    if args.backend == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS unavailable; no fallback or data loading')
    adapter = load_train_adapter()
    attempts = adapter.install_test_guard()
    trainsets, validation, manifest, identities, images, labels = adapter.prepare(PARTITION_SEED)
    if len(trainsets) != r.CLIENTS or any(len(ds) <= 0 for ds in trainsets):
        raise ValueError('fixed plan requires twenty nonempty clients')
    counts = manifest['client_label_counts']
    if len(counts) != r.CLIENTS:
        raise ValueError('wrong TRAIN count manifest')
    total = sum(map(len, trainsets))
    weights = [len(ds) / total for ds in trainsets]
    config = {'base': base, 'weights': weights, 'data_manifest': manifest,
        'data_identities': identities, 'train_adapter_identities': adapter.source_identities()}
    config['signature'] = r.signature(config)
    r.validate_config(config, base)
    if saved_config is not None and saved_config != config:
        raise ValueError('data manifest/input identities differ from resume config')
    model = r.new_model(SEED)
    heads = [r.new_head() for _ in trainsets]
    residuals = [torch.zeros(r.PARAMETERS, dtype=torch.float32) for _ in trainsets]
    rows = []
    if checkpoint is not None:
        model = r.validate_checkpoint(checkpoint, config, model)
        for head, state in zip(heads, checkpoint['heads']):
            head.load_state_dict(state)
        residuals, rows = checkpoint['residuals'], checkpoint['rows']
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    r.atomic_json(out / 'config.json', config)
    io_path = out / 'checkpoint_costs.jsonl'
    io_costs = [json.loads(line) for line in io_path.read_text().splitlines()] if io_path.exists() else []
    for round_id in range(len(rows), ROUNDS):
        began = time.perf_counter()
        anchor = r.vectorize(model).clone()
        aggregate, clients = torch.zeros_like(anchor), []
        for client_id, dataset in enumerate(trainsets):
            seed = SEED + 100 * round_id + client_id
            delta, state, counters = r.train_local(model, heads[client_id], dataset,
                                                  counts[client_id], seed, args.backend)
            heads[client_id].load_state_dict(state)
            q, residuals[client_id], wire, encoding = r.encode_shared(delta, residuals[client_id])
            aggregate.add_(q, alpha=weights[client_id])
            if not torch.equal(r.vectorize(model), anchor):
                raise ValueError('client altered global anchor')
            if round_id == 0 and client_id == 0:
                (out / 'first_payload.bin').write_bytes(wire)
            clients.append({'client': client_id, 'seed': seed, **counters, **encoding})
        r.assign_vector(model, anchor + aggregate)
        metrics, eval_cost = r.evaluate(model, None, validation, args.backend)
        rows.append({'round': round_id + 1, 'validation': metrics, 'validation_cost': eval_cost,
            'clients': clients, 'upload_bytes': sum(v['upload_bytes'] for v in clients),
            'download_bytes_analytical': r.CLIENTS * (64 + 4 * r.PARAMETERS),
            'actual_gradient_evaluations': sum(v['combined_backward_calls'] for v in clients),
            'head_gradient_scalars': sum(v['residual_gradient_scalars'] for v in clients),
            'global_anchor_unchanged_between_clients': True,
            'round_seconds_excluding_checkpoint_io': time.perf_counter() - began})
        io_cost = r.save_checkpoint(out / 'checkpoint.pt', config, model, heads, residuals, rows)
        io_costs.append({'round': round_id + 1, **io_cost})
        with io_path.open('a') as stream:
            stream.write(json.dumps(io_costs[-1], allow_nan=False) + '\n')
        r.atomic_json(out / 'result.json', {'status': 'TRAINING_DONE' if len(rows) == ROUNDS else 'RUNNING',
            'signature': config['signature'], 'rows': rows, 'checkpoint_costs': io_costs,
            'official_test_byte_read': False, 'formal_training_executed': True})
        print(json.dumps({'round': round_id + 1, **metrics}), flush=True)
    holdouts = [adapter.partitions.IndexedCifar10(images, labels, ids)
                for ids in manifest['holdout_indices']]
    final_metrics, final_costs = [], []
    for client_id, dataset in enumerate(holdouts):
        metrics, cost = r.evaluate(model, heads[client_id], dataset, args.backend)
        final_metrics.append(metrics)
        final_costs.append({'client': client_id, **cost})
    r.verify_identities(base['source_identities'])
    adapter.verify_identities({**config['train_adapter_identities'], **identities})
    if attempts:
        raise ValueError('forbidden TEST byte access attempted')
    costs = {'extra_personal_weight_bytes_per_client': 4 * r.HEAD_PARAMETERS,
        'extra_personal_weight_bytes_all_clients': r.CLIENTS * 4 * r.HEAD_PARAMETERS,
        'ef_fp32_bytes_per_client': 4 * r.PARAMETERS,
        'global_fp32_weight_bytes': 4 * r.PARAMETERS,
        'downlink_header_bytes_analytical': 64,
        'personal_fit_TRAIN_logit_cache_bytes': 0,
        'personal_independent_backward_seconds': None,
        'activation_peak_bytes': None,
        'activation_note': 'not measured; forward activations are not included in 2600B weights',
        'personal_linear_multiply_add_pairs_per_image': 64 * 10,
        'personal_linear_bias_additions_per_image': 10,
        'comparison_note': 'matched_head already freezes body; 40B bias vs 80B vector saves only 40B; '
            'posthoc bias additionally needs FP32 TRAIN logits 40*n bytes plus FP64/design/Hessian work',
        'checkpoint_costs': io_costs}
    r.atomic_json(out / 'result.json', {'status': 'DONE', 'signature': config['signature'],
        'rows': rows, 'final_client_metrics': final_metrics, 'final_summary': final_summary(final_metrics),
        'final_holdout_costs': final_costs, 'cost_accounting': costs,
        'formal_training_executed': True, 'official_test_byte_read': False,
        'review': 'REVIEW_UNAVAILABLE / LOCAL_CHECK_ONLY'})


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--out', type=Path, default=r.HERE / 'formal_p20261203_s20261211')
    parser.add_argument('--backend', choices=('mps', 'cpu'), default='mps')
    parser.add_argument('--execute-formal', action='store_true',
        help='explicitly authorize and launch the fixed 20-round TRAIN experiment')
    args = parser.parse_args()
    torch.set_num_threads(1)
    base = base_config(args.backend)
    saved_config, checkpoint = preflight(args.out, base)
    if not args.execute_formal:
        print(json.dumps({'status': 'DESIGN_ONLY', 'formal_training_executed': False,
            'raw_data_opened': False, 'train_adapter_imported': False,
            'resume_preflight': 'VALIDATED' if saved_config else 'NO_SAVED_STATE',
            'base_config': base}, indent=2, ensure_ascii=False, allow_nan=False))
        return
    formal(args, base, saved_config, checkpoint)


if __name__ == '__main__':
    main()
