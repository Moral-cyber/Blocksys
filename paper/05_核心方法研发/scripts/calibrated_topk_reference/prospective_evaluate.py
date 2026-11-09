"""Standalone prospective final evaluation; reuse the frozen two-method generic run."""
from pathlib import Path
import argparse
import json
import torch
import common as c
import evaluate as fixed_evaluate
import prospective as p


def run(model_root, out, images, labels, manifest, data_identities, protocol_plan,
        uniform_roots=(), *, fixture=False):
    first = json.loads((model_root / c.METHODS[0] / 'config.json').read_text())
    seed = first['seed']
    context, metadata = p.make_context(protocol_plan, seed, fixture=fixture)
    for method in c.METHODS:
        config = json.loads((model_root / method / 'config.json').read_text())
        if (config['seed'] != seed or config['partition_seed'] != p.PARTITION
                or (not fixture and config['rounds'] != p.ROUNDS)
                or config['runtime'].get('prospective_reference') != metadata):
            raise ValueError('signed prospective source/runtime override mismatch')
    if manifest.get('partition_seed') != p.PARTITION:
        raise ValueError('prospective evaluation partition mismatch')
    bound = p.bind(fixed_evaluate.run, context)
    report = bound(model_root, out, images, labels, manifest, data_identities, uniform_roots)
    # Additional disclosure is part of this new output, never an original gate.
    report['prospective_protocol'] = metadata
    report['reference_kind'] = 'same-cap TopK+EF on prospectively declared new partition/seed'
    context.verify_identities(context.source_identities())
    context.atomic_json(out / 'result.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--model-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--protocol-plan', type=Path, required=True)
    parser.add_argument('--uniform-evaluation', type=Path, action='append', default=[])
    args = parser.parse_args()
    first = json.loads((args.model_root / c.METHODS[0] / 'config.json').read_text())
    if first['partition_seed'] != p.PARTITION or first['seed'] not in p.SEEDS or first['rounds'] != p.ROUNDS:
        raise ValueError('prospective complete partition/seed/20round protocol required')
    context, _ = p.make_context(args.protocol_plan, first['seed'])
    attempts = context.install_test_guard()
    torch.set_num_threads(1)
    _, _, manifest, identities, images, labels = context.prepare(p.PARTITION)
    report = run(args.model_root.resolve(), args.out.resolve(), images, labels, manifest, identities,
                 args.protocol_plan, args.uniform_evaluation)
    if attempts:
        raise ValueError('official TEST byte access attempted')
    print(json.dumps({'decision': report['decision'], 'gates': report['gates'],
                      'partition_seed': report['partition_seed'], 'training_seed': report['training_seed']}), flush=True)


if __name__ == '__main__':
    main()
