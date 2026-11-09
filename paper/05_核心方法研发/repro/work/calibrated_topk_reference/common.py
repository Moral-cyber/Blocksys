"""Pinned reuse of strict TRAIN data and unchanged local/evaluation components."""
from pathlib import Path
import hashlib
import importlib.util
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE = HERE.parent / 'calibrated_lowbit_confirm'
METHODS = ('ce_topk', 'fedlc_topk')
VERSION, CAP, PARAMETERS, K = 1, 54808, 136874, 6843
BASE_PINS = {
    'PLAN.md': 'ac6f2f4b9582e95897f9d9dedcd3af3481114c3824674cc1cee02832a3144597',
    'common.py': '3f2e4693488d7e5a35a182e5340edb9d73f5cafc45b7c0e61b1e2cc392a22d9a',
    'train.py': '7fe8777db0ab8b2309140b39026a9bdbec42cd09ba9d2b8ef0c89b831a5641bd',
    'evaluate.py': 'afcb8a47dc9c303b3e7d558e4e91b50b560cd10dd0287956969d61ab27e13250'}
PAIR_PATH = HERE.parent / 'method_followup/guarded_codec.py'
PAIR_SHA = '3dcb6636f12d6dc17fe023a978a9016ce10593ed3da5472b1ad498a33b8c948e'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_frozen(name, path, expected):
    if sha(path) != expected:
        raise ValueError('frozen dependency changed: ' + str(path))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


for name, expected in BASE_PINS.items():
    if sha(BASE / name) != expected:
        raise ValueError('frozen confirm source changed: ' + name)
base = load_frozen('_topk_frozen_confirm_common', BASE / 'common.py', BASE_PINS['common.py'])
pairs = load_frozen('_topk_frozen_canonical_pairs', PAIR_PATH, PAIR_SHA)
runner, accelerator, partitions = base.runner, base.accelerator, base.partitions
balanced, bias_solver = base.balanced, base.bias_solver
prepare, install_test_guard = base.prepare, base.install_test_guard
signature, atomic_json, verify_identities = base.signature, base.atomic_json, base.verify_identities
validate_checkpoint = base.validate_checkpoint


def source_identities():
    for name, expected in BASE_PINS.items():
        if sha(BASE / name) != expected:
            raise ValueError('frozen confirm source changed')
    if sha(PAIR_PATH) != PAIR_SHA:
        raise ValueError('frozen pair serializer changed')
    result = base.source_identities()
    result.update({str((HERE / name).resolve()): sha(HERE / name)
                   for name in ('PLAN.md', 'common.py', 'wire.py', 'train.py', 'evaluate.py')})
    result[str(PAIR_PATH.resolve())] = PAIR_SHA
    return result
