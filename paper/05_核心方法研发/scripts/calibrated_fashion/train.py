"""Fashion task wrapper; no changes to the frozen CIFAR confirmation sources."""
from pathlib import Path
import argparse
import torch
import adapter


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=20261221)
    parser.add_argument('--partition-seed', type=int, default=20261207)
    parser.add_argument('--rounds', type=int, default=20)
    parser.add_argument('--backend', choices=('mps', 'cpu'), default='mps')
    args = parser.parse_args()
    if args.rounds < 1:
        parser.error('positive rounds required')
    attempts = adapter.install_test_guard()
    torch.set_num_threads(1)
    if args.backend == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS unavailable; no fallback')
    trainsets, validation, manifest, identities, _, _ = adapter.prepare(args.partition_seed)
    for method in adapter.common.METHODS:
        adapter.training.run(args, method, trainsets, validation, manifest, identities)
    if attempts:
        raise ValueError('official TEST byte access attempted')


if __name__ == '__main__':
    main()
