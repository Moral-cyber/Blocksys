"""Explicit process-local Fashion routing around frozen confirmation modules."""
from pathlib import Path
import hashlib
import importlib.util
import json
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CONFIRM = HERE.parent / 'calibrated_lowbit_confirm'
PINNED = {
    'work/calibrated_lowbit_confirm/common.py': '3f2e4693488d7e5a35a182e5340edb9d73f5cafc45b7c0e61b1e2cc392a22d9a',
    'work/calibrated_lowbit_confirm/train.py': '7fe8777db0ab8b2309140b39026a9bdbec42cd09ba9d2b8ef0c89b831a5641bd',
    'work/calibrated_lowbit_confirm/evaluate.py': 'afcb8a47dc9c303b3e7d558e4e91b50b560cd10dd0287956969d61ab27e13250',
    'work/method_followup/run_fashion.py': 'f87e281acaf9a46a62eb31be588c9505ea7bb53c67162fd8fcf2ccc3440111d4'}
TRAIN_FILES = {
    'train-images-idx3-ubyte.gz': {
        'md5': '8d4fb7e6c68d591d4c3dfef9ec88bf0d',
        'sha256': '3aede38d61863908ad78613f6a32ed271626dd12800ba2636569512369268a84'},
    'train-labels-idx1-ubyte.gz': {
        'md5': '25c81989df183df01b3e8a0aad5dffbe',
        'sha256': 'a04f17134ac03560a47e3764e11b92fc97de4d1bfaf8ba1a3aa29af54cc90845'}}
LABELS_SHA256 = 'e3245b63f7c40d1c8b652835f19744d970c1613b40ddac6bcfc87b9c46e16a0b'
ROUTES = [
    'confirm.common.prepare = calibrated_fashion.adapter.prepare',
    'confirm.common.partitions = frozen run_fashion module; IndexedCifar10 alias = FashionIndexDataset proxy',
    'confirm.common.source_identities = wrapper + all frozen dependencies',
    'confirm.common.accelerator.runtime_metadata = explicit Fashion route metadata',
    'confirm.common.install_test_guard = rejects test_batch and t10k',
    'frozen confirm.train.run and confirm.evaluate.run unchanged',
    'frozen run_fashion.load_official_train/source_file_manifest never called']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


for relative, expected in PINNED.items():
    if sha(ROOT / relative) != expected:
        raise ValueError('frozen dependency mismatch: ' + relative)
common = load('_fashion_frozen_confirm_common', CONFIRM / 'common.py')
base_sources = common.source_identities()
fashion = load('_fashion_frozen_dataset', HERE.parent / 'method_followup/run_fashion.py')
previous_common = sys.modules.get('common')
sys.modules['common'] = common
try:
    training = load('_fashion_frozen_confirm_training', CONFIRM / 'train.py')
    evaluation = load('_fashion_frozen_confirm_evaluation', CONFIRM / 'evaluate.py')
finally:
    if previous_common is None:
        del sys.modules['common']
    else:
        sys.modules['common'] = previous_common
original_runtime_metadata = common.accelerator.runtime_metadata


class FashionPartitionsProxy:
    """Only expose frozen dataset routing required by final evaluation."""
    __file__ = fashion.__file__
    IndexedCifar10 = fashion.FashionIndexDataset


def install_test_guard():
    attempts = []
    def audit(event, arguments):
        if event == 'open' and isinstance(arguments[0], (str, bytes)):
            path = str(arguments[0])
            if 'test_batch' in path or 't10k' in path:
                attempts.append(path)
                raise ValueError('Fashion/CIFAR official TEST byte read forbidden')
    sys.addaudithook(audit)
    return attempts


def source_identities():
    paths = [Path(path) for path in base_sources]
    paths += [ROOT / relative for relative in PINNED]
    paths += [HERE / name for name in ('PLAN.md', 'adapter.py', 'train.py', 'evaluate.py')]
    for relative, expected in PINNED.items():
        if sha(ROOT / relative) != expected:
            raise ValueError('frozen dependency changed: ' + relative)
    return {str(path.resolve()): sha(path) for path in paths}


def runtime_metadata(backend, inherited_route):
    inherited = original_runtime_metadata(backend, inherited_route)
    inherited.update({'accelerated_route': 'calibrated_fashion_wrapper',
        'dataset_task': 'FashionMNIST', 'route_overrides': ROUTES,
        'wrapper_source_identities': source_identities(),
        'data_loader_scope': 'only two official TRAIN gzip files, strict MD5/SHA/label-hash',
        'official_test_policy': 'no test_batch or t10k bytes, including integrity checks',
        'preprocessing': 'uint8 gray28 -> zero-pad2 -> repeat3 -> FP32 -> (x/255-.5)/.5',
        'backbone': 'same SmallCNN136874 cross-task protocol; no optimal-Fashion architecture claim',
        'inherited_metadata_scope': 'inherited helper source hashes identify code; actual prepare/dataset route is explicitly overridden above'})
    return inherited


def load_train_only(raw=None):
    raw = fashion.RAW if raw is None else Path(raw)
    identities, files = {}, []
    for name, expected in TRAIN_FILES.items():
        path = raw / name
        data = path.read_bytes()
        md5, digest = hashlib.md5(data).hexdigest(), hashlib.sha256(data).hexdigest()
        if md5 != expected['md5'] or digest != expected['sha256']:
            raise ValueError('Fashion TRAIN compressed identity mismatch: ' + name)
        if fashion.EXPECTED_MD5[name] != expected['md5']:
            raise ValueError('official source MD5 constant changed')
        identities[str(path.resolve())] = digest
        files.append({'file': name, 'compressed_bytes': len(data), 'md5': md5, 'sha256': digest,
                      'expected_official_md5': expected['md5'], 'official_md5_verified': True})
    images = fashion._read_idx(raw / 'train-images-idx3-ubyte.gz', magic=2051,
                               expected_shape=(60000, 28, 28))
    labels = fashion._read_idx(raw / 'train-labels-idx1-ubyte.gz', magic=2049,
                               expected_shape=(60000,)).astype(np.int64)
    label_hash = hashlib.sha256(labels.astype('<i8').tobytes()).hexdigest()
    if not np.array_equal(np.bincount(labels, minlength=10), np.full(10, 6000)) or label_hash != LABELS_SHA256:
        raise ValueError('Fashion TRAIN original label rows/hash mismatch')
    return images, labels, {'source_gzip_files': files,
        'official_train_labels_sha256': label_hash, 'official_split': 'FashionMNIST official TRAIN 60000 only',
        'strict_loader': 'two TRAIN gzip only; test gzip not opened'}, identities


def prepare(partition_seed, raw=None):
    images, labels, provenance, identities = load_train_only(raw)
    train_ids, _, validation_ids, manifest = fashion.partition_indices(
        labels, alpha=.1, split_seed=partition_seed, mixing=0., validation_size=2000)
    manifest.update({'source': 'FashionIndexA0p1', 'task': 'FashionMNIST', **provenance,
        'route_overrides': ROUTES, 'official_test_policy': 'all t10k/test_batch byte reads rejected',
        'preprocessing': 'uint8 gray28 -> pad2 -> repeat3 -> normalize[-1,1]'})
    training_sets = [fashion.FashionIndexDataset(images, labels, ids) for ids in train_ids]
    validation = fashion.FashionIndexDataset(images, labels, validation_ids)
    return training_sets, validation, manifest, identities, images, labels


common.prepare = prepare
common.partitions = FashionPartitionsProxy()
common.source_identities = source_identities
common.install_test_guard = install_test_guard
common.accelerator.runtime_metadata = runtime_metadata
