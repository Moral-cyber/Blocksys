"""Shared strict-TRAIN loading, frozen identities and checkpoint validation."""
from pathlib import Path
import hashlib
import importlib.util
import json
import pickle
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE.parent / 'method_followup'))
import run_followup as runner
import run_partitions as partitions
import run_accelerated as accelerator

METHODS = ('ce_uniform3', 'fedlc_uniform3', 'fedlc_dense')
VERSION = 1
CAP = 54808
PARAMETERS = 136874
PINNED = {
    'work/balanced_quant/run.py': 'b94f6a3a985839e6781582c6203e3dd8154fc97cf7bcc8edfe9d4648240625d4',
    'work/mixed_precision/codec.py': '46f3601e5c107f192d359227b21aea2734cc5e798674f2dccbbdf46fc796d863',
    'work/prior_bias/solver.py': '6b8dda5e86438f78f3918f8601ec3ce9e7099417d3b42ac55bfbeb4014132b96'}
REFERENCE = ROOT / 'work/balanced_quant/development_p20261193_s20261201/ce_uniform3/config.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frozen_module(name, relative):
    path = ROOT / relative
    if sha(path) != PINNED[relative]:
        raise ValueError('frozen dependency differs from pinned source: ' + relative)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


balanced = frozen_module('_confirm_frozen_balanced', 'work/balanced_quant/run.py')
codec = frozen_module('_confirm_frozen_uniform3', 'work/mixed_precision/codec.py')
bias_solver = frozen_module('_confirm_frozen_bias', 'work/prior_bias/solver.py')


def install_test_guard():
    attempts = []
    def audit(event, arguments):
        if event == 'open' and isinstance(arguments[0], (str, bytes)):
            if 'test_batch' in str(arguments[0]):
                attempts.append(str(arguments[0]))
                raise ValueError('official TEST byte read forbidden')
    sys.addaudithook(audit)
    return attempts


def source_identities():
    paths = [HERE / name for name in ('PLAN.md', 'common.py', 'train.py', 'evaluate.py')]
    paths += [ROOT / relative for relative in PINNED]
    paths += [Path(partitions.__file__), Path(runner.__file__), Path(accelerator.__file__),
              Path(sys.modules[runner.SmallCNN.__module__].__file__), REFERENCE]
    for relative, expected in PINNED.items():
        if sha(ROOT / relative) != expected:
            raise ValueError('pinned source changed')
    return {str(path.resolve()): sha(path) for path in paths}


def verify_identities(identities):
    if any(sha(path) != expected for path, expected in identities.items()):
        raise ValueError('source or data input changed during execution')


def signature(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def load_train_only(raw):
    reference = json.loads(REFERENCE.read_text())['data_manifest']
    expected = {row['name']: row['sha256'] for row in reference['official_train_batch_files']}
    images, labels, files, identities = [], [], [], {str(REFERENCE): sha(REFERENCE)}
    for number in range(1, 6):
        path = raw / 'cifar-10-batches-py' / f'data_batch_{number}'
        digest = sha(path)
        if digest != expected[path.name]:
            raise ValueError('official TRAIN batch differs from fixed reference')
        identities[str(path.resolve())] = digest
        with path.open('rb') as stream:
            batch = pickle.load(stream, encoding='latin1')
        images.append(np.asarray(batch['data'], dtype=np.uint8))
        labels.extend(batch['labels'])
        files.append({'name': path.name, 'sha256': digest})
    images = np.concatenate(images).reshape(50000, 3, 32, 32).transpose(0, 2, 3, 1)
    labels = np.asarray(labels, dtype=np.int64)
    label_hash = hashlib.sha256(labels.astype('<i8').tobytes()).hexdigest()
    if (images.shape != (50000, 32, 32, 3) or labels.shape != (50000,)
            or not np.array_equal(np.bincount(labels, minlength=10), np.full(10, 5000))
            or label_hash != reference['official_train_labels_sha256']):
        raise ValueError('TRAIN row order or labels mismatch')
    return images, labels, {'official_train_batch_files': files,
        'official_train_labels_sha256': label_hash,
        'official_split': 'CIFAR10 TRAIN only; strict five-file read',
        'normalization': '(FP32(uint8_CHW)/255 - 0.5)/0.5'}, identities


def prepare(partition_seed, raw=None):
    images, labels, provenance, identities = load_train_only(partitions.RAW if raw is None else raw)
    train_ids, holdout_ids, validation_ids, manifest = partitions.partition_indices(
        labels, alpha=.1, split_seed=partition_seed, mixing=0., validation_size=2000)
    manifest.update({'source': 'Cifar10IndexA0p1', **provenance,
                     'official_test_policy': 'all test_batch byte reads rejected by audit hook'})
    trainsets = [partitions.IndexedCifar10(images, labels, ids) for ids in train_ids]
    validation = partitions.IndexedCifar10(images, labels, validation_ids)
    return trainsets, validation, manifest, identities, images, labels


def validate_checkpoint(checkpoint, config):
    if checkpoint.get('version') != VERSION or checkpoint.get('signature') != config['signature']:
        raise ValueError('incompatible checkpoint version/signature')
    if checkpoint.get('config') != config:
        raise ValueError('incompatible checkpoint full source/config')
    rows = checkpoint.get('rows')
    if (not isinstance(rows, list) or len(rows) > config['rounds']
            or [row.get('round') for row in rows] != list(range(1, len(rows) + 1))):
        raise ValueError('incompatible checkpoint history')
    residuals = checkpoint.get('residuals')
    if not isinstance(residuals, list) or len(residuals) != len(config['weights']):
        raise ValueError('incompatible checkpoint EF count')
    for value in residuals:
        if (not isinstance(value, torch.Tensor) or value.shape != (PARAMETERS,)
                or value.dtype != torch.float32 or value.device.type != 'cpu'
                or not torch.isfinite(value).all()):
            raise ValueError('incompatible checkpoint EF tensor')
    model = runner.SmallCNN()
    model.load_state_dict(checkpoint['model'])
    if not all(torch.isfinite(value).all() for value in model.state_dict().values()):
        raise ValueError('nonfinite checkpoint model')
    if config['method'].endswith('dense') and any(torch.count_nonzero(value) for value in residuals):
        raise ValueError('dense checkpoint must have zero residual')
    return model
