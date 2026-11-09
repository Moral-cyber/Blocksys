"""Final Fashion clientholdout evaluation through the frozen confirmation flow."""
from pathlib import Path
import argparse
import json
import torch
import adapter


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--model-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    attempts = adapter.install_test_guard()
    torch.set_num_threads(1)
    first = json.loads((args.model_root / adapter.common.METHODS[0] / 'config.json').read_text())
    if first['data_manifest'].get('task') != 'FashionMNIST':
        raise ValueError('Fashion wrapper refuses non-Fashion checkpoints')
    _, _, manifest, identities, images, labels = adapter.prepare(first['partition_seed'])
    report = adapter.evaluation.run(args.model_root.resolve(), args.out.resolve(), images, labels, manifest, identities)
    # This wrapper annotates the new output only, preserving the frozen
    # evaluator and its six gates exactly as the inherited implementation.
    report.update({'task': 'FashionMNIST', 'route_overrides': adapter.ROUTES,
        'task_scope': 'second-task transfer boundary, not an optimized Fashion backbone',
        'official_test_byte_read_attempts': attempts,
        'allowed_training_seeds': [20261221, 20261222, 20261223]})
    adapter.common.atomic_json(args.out.resolve() / 'result.json', report)
    if attempts:
        raise ValueError('official TEST byte access attempted')
    print(json.dumps({'decision': report['decision'], 'gates': report['gates']}), flush=True)


if __name__ == '__main__':
    main()
