"""Final TopK reference evaluation with the exact frozen local prior/bias/head helper."""
from pathlib import Path
import argparse
import json
import time
import torch
import common as c

frozen = c.load_frozen('_topk_frozen_confirm_evaluation', c.BASE / 'evaluate.py', c.BASE_PINS['evaluate.py'])
frozen.c = c.base


def run(model_root, out, images, labels, manifest, data_identities, uniform_roots=()):
    if out.exists():
        raise ValueError('fresh evaluation output required')
    sources, inputs, loaded, histories = c.source_identities(), {}, {}, {}
    for method in c.METHODS:
        folder = model_root / method
        for name in ('config.json', 'checkpoint.pt', 'result.json'):
            path = folder / name
            inputs[str(path.resolve())] = c.sha(path)
        config = json.loads((folder / 'config.json').read_text())
        if config['signature'] != c.signature({key: value for key, value in config.items() if key != 'signature'}):
            raise ValueError('training config self-signature invalid')
        if (config['method'] != method or config['data_manifest'] != manifest
                or config['source_identities'] != sources or config['data_identities'] != data_identities
                or not config['reference_only'] or config['new_go_criterion']):
            raise ValueError('reference source/data/protocol mismatch')
        checkpoint = torch.load(folder / 'checkpoint.pt', weights_only=True)
        model = c.validate_checkpoint(checkpoint, config)
        history = json.loads((folder / 'result.json').read_text())
        if len(checkpoint['rows']) != config['rounds'] or history['status'] != 'DONE' or history['rows'] != checkpoint['rows'] or history['signature'] != config['signature']:
            raise ValueError('only final complete checkpoint/history allowed')
        loaded[method], histories[method] = (model, config), history['rows']
    configurations = [config for _, config in loaded.values()]
    if len({(config['seed'], config['partition_seed'], config['rounds'], config['backend']) for config in configurations}) != 1:
        raise ValueError('both reference arms must share protocol')
    # Read-only comparisons: only seed11 is a paired initialization comparison.
    references = []
    for root in uniform_roots:
        path = Path(root) / 'result.json'
        inputs[str(path.resolve())] = c.sha(path)
        reference = json.loads(path.read_text())
        if reference['status'] != 'DONE' or reference['partition_seed'] != configurations[0]['partition_seed'] or reference['training_rounds'] != configurations[0]['rounds']:
            raise ValueError('uniform reference must be complete on the same partition/rounds')
        arms = {}
        for method in ('ce_uniform3', 'fedlc_uniform3'):
            config_paths = [Path(name) for name in reference['frozen_model_inputs'] if name.endswith('/' + method + '/config.json')]
            if len(config_paths) != 1:
                raise ValueError('unambiguous frozen uniform config required')
            config_path = config_paths[0]
            if c.sha(config_path) != reference['frozen_model_inputs'][str(config_path)]:
                raise ValueError('frozen uniform config changed')
            training_path = config_path.parent / 'result.json'
            for item in (config_path, training_path):
                inputs[str(item.resolve())] = c.sha(item)
            config = json.loads(config_path.read_text())
            history = json.loads(training_path.read_text())
            if config['data_manifest'] != manifest or config['seed'] != reference['training_seed'] or config['upload_cap_bytes'] != c.CAP or history['status'] != 'DONE' or len(history['rows']) != config['rounds'] or history['signature'] != config['signature']:
                raise ValueError('uniform reference partition/seed/cap/history mismatch')
            arms[method] = {'actual_upload_bytes': sum(row['upload_bytes'] for row in history['rows']),
                'download_bytes_analytical': sum(row['download_bytes_analytical'] for row in history['rows']),
                'actual_upload_bytes_per_client_round': 54067, 'cap_bytes': config['upload_cap_bytes'],
                'summaries': reference['results'][method]['summaries']}
        references.append({'training_seed': reference['training_seed'], 'paired_initialization_seed': reference['training_seed'] == configurations[0]['seed'],
            'original_candidate_decision_unchanged': reference['decision'], 'arms': arms})
    out.mkdir(parents=True)
    training = [c.partitions.IndexedCifar10(images, labels, ids) for ids in manifest['train_indices']]
    holdouts = [c.partitions.IndexedCifar10(images, labels, ids) for ids in manifest['holdout_indices']]
    began, results = time.perf_counter(), {}
    for method, (model, config) in loaded.items():
        results[method] = frozen.evaluate_model(model, training, holdouts, method, config['seed'], out, manifest)
        print(json.dumps({'method': method, 'summaries': results[method]['summaries']}), flush=True)
    c.verify_identities({**sources, **data_identities, **inputs})
    report = {'status': 'DONE', 'purpose': 'strict same-cap TopK+EF reference only; no new candidate or GO criterion',
        'decision': 'REFERENCE_ONLY', 'gates': None, 'results': results,
        'training_seed': configurations[0]['seed'], 'partition_seed': configurations[0]['partition_seed'],
        'training_rounds': configurations[0]['rounds'], 'source_identities': sources,
        'data_identities': data_identities, 'frozen_model_inputs': inputs,
        'identities_before_after_equal': True, 'official_test_byte_reads': False, 'official_test_inference': False,
        'training_samples_actual': sum(map(len, manifest['train_indices'])),
        'holdout_samples_actual': sum(map(len, manifest['holdout_indices'])),
        'bias_total_train_image_forwards': sum(client['bias_extra_train_forward']['image_forwards'] for result in results.values() for client in result['clients']),
        'bias_total_cnn_backward': 0, 'bias_state_bytes_per_client': 40, 'bias_extra_uplink_bytes': 0,
        'matched_head_total_gradient_evaluations': sum(client['matched_head_training']['gradient_evaluations'] for result in results.values() for client in result['clients']),
        'training_communication': {method: {'actual_upload_bytes': sum(row['upload_bytes'] for row in rows),
            'download_bytes_analytical': sum(row['download_bytes_analytical'] for row in rows),
            'actual_upload_bytes_per_client_round': c.CAP, 'cap_bytes': c.CAP,
            'actual_gradient_evaluations': sum(row['actual_gradient_evaluations'] for row in rows)} for method, rows in histories.items()},
        'uniform_reference_comparisons': references,
        'budget_scope': 'same54808B cap; TopK actual54808B vs uniform3 actual54067B, +741B/client/round; only same trainingseed is paired; dense/global downlinks analytical; no network/authentication',
        'previous_decisions': 'all old balanced/prior/bias/confirm decisions remain unchanged',
        'review': 'REVIEW_UNAVAILABLE', 'evaluation_seconds': time.perf_counter() - began}
    c.atomic_json(out / 'result.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--model-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--uniform-evaluation', type=Path, action='append')
    args = parser.parse_args()
    attempts = c.install_test_guard()
    torch.set_num_threads(1)
    first = json.loads((args.model_root / c.METHODS[0] / 'config.json').read_text())
    if (first['seed'], first['partition_seed'], first['rounds']) != (20261211, 20261203, 20):
        raise ValueError('fixed reference requires seed11/partition20261203/final20')
    _, _, manifest, identities, images, labels = c.prepare(first['partition_seed'])
    uniform = args.uniform_evaluation if args.uniform_evaluation is not None else [c.BASE / f'evaluation_p20261203_s{seed}' for seed in (20261211, 20261212, 20261213)]
    report = run(args.model_root.resolve(), args.out.resolve(), images, labels, manifest, identities, uniform)
    if attempts:
        raise ValueError('official TEST byte access attempted')
    print(json.dumps({'decision': report['decision'], 'new_gates': report['gates']}), flush=True)


if __name__ == '__main__':
    main()
