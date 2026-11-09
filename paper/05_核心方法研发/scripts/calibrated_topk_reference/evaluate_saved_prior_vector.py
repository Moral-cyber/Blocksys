"""Decision-free two-TopK schema adapter around the pinned saved-array evaluator."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import sys
import types

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TOPK_MODELS = ('ce_topk', 'fedlc_topk')
ORIGINAL = HERE.parent / 'prior_vector/evaluate_saved.py'
SOLVER = HERE.parent / 'prior_vector/solver.py'
PINS = {
    ORIGINAL: '7b45a055df8c3e03421fd9edd6749c1e2bfba1203864e0b41695413db1fb8b05',
    SOLVER: '9c3196eb3e6ff7f5f32073e3bc4aae8d986ac7e44bdf0d719258343847d4dbae',
    HERE.parent / 'calibration_baselines/solver.py': '10b11405e6662cb06d08a20549db92e2948b730c426aa1f0b912b958f1211f1c',
    HERE.parent / 'calibration_baselines/probe.py': 'd1fab28871824163a6d2fc2ee44e043698537b00dabc0e02af2618531adcd0c7'}
PLAN_SHA = '703d5956c9edd24d5a03501a75f678a43222d19bd3dd1ede515ed77545b0830b'
REGISTRATION_SHA = '1eb7d284c43992a3cca58e31448a4b595b88e67abdf167c24a605342bf2f0568'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_original():
    for path, expected in PINS.items():
        if sha(path) != expected:
            raise ValueError('frozen evaluator/solver/reader changed: ' + str(path))
    solver = load_module('_topk_saved_frozen_prior_solver', SOLVER)
    previous = sys.modules.get('solver')
    sys.modules['solver'] = solver
    try:
        original = load_module('_topk_saved_frozen_evaluator', ORIGINAL)
    finally:
        if previous is None:
            del sys.modules['solver']
        else:
            sys.modules['solver'] = previous
    return original


def run(input_root, out, protocol_plan, protocol_registration):
    input_root, out, plan, registration = [Path(value).resolve() for value in (input_root, out, protocol_plan, protocol_registration)]
    if sha(plan) != PLAN_SHA or sha(registration) != REGISTRATION_SHA:
        raise ValueError('exact explicit main PLAN and preregistration required')
    registration_data = json.loads(registration.read_text())
    if registration_data['frozen_sources'].get('work/predictive_pfl_confirm/PLAN.md') != PLAN_SHA or not registration_data['official_test_forbidden']:
        raise ValueError('registered PLAN/TEST policy mismatch')
    existing = json.loads((input_root / 'result.json').read_text())
    if existing.get('status') != 'DONE' or set(existing.get('results', {})) != set(TOPK_MODELS):
        raise ValueError('complete exact ce_topk/fedlc_topk saved-array schema required')
    for method in TOPK_MODELS:
        if [client['client'] for client in existing['results'][method]['clients']] != list(range(20)):
            raise ValueError('all20 ordered clients required for both TopK models')
    original = load_original()
    sources = {str(path.resolve()): expected for path, expected in PINS.items()} | {
        str(Path(__file__).resolve()): sha(__file__), str(plan): PLAN_SHA, str(registration): REGISTRATION_SHA}
    identity = {'wrapper_path': str(Path(__file__).resolve()), 'wrapper_sha256': sources[str(Path(__file__).resolve())],
        'original_evaluator_path': str(ORIGINAL.resolve()), 'original_evaluator_sha256': PINS[ORIGINAL],
        'accepted_method_tuple': list(TOPK_MODELS), 'input_schema': 'topk_two',
        'adaptation': 'same original run code with independent globals: model-tuple acceptance and output schema/identity only',
        'solver_reader_math_metric_changes': 0, 'original_module_globals_unchanged': True}
    original_tuple, original_atomic = original.CONFIRM_MODELS, original.atomic_json

    def atomic_json(path, value):
        if path.name == 'inputs_before.json':
            value = {**value, **sources}
        elif path.name == 'result.json':
            if any(sha(filename) != expected for filename, expected in sources.items()):
                raise ValueError('wrapper/evaluator/solver/protocol source changed')
            value['input_schema'] = 'topk_two'
            value['schema_adapter'] = identity
            value['source_identities'].update(sources)
            value['input_sha256_before_after_equal'].update(sources)
            value['training_seed'] = existing.get('training_seed')
            value['partition_seed'] = existing.get('partition_seed')
            if 'gates' in value or 'decision' in value:
                raise ValueError('schema adapter must not produce research gates/decision')
        return original_atomic(path, value)

    namespace = dict(original.run.__globals__)
    namespace['CONFIRM_MODELS'] = TOPK_MODELS
    namespace['atomic_json'] = atomic_json
    bound = types.FunctionType(original.run.__code__, namespace, original.run.__name__, original.run.__defaults__, original.run.__closure__)
    bound.__kwdefaults__ = original.run.__kwdefaults__
    result = bound(input_root, out, plan, registration)
    if original.CONFIRM_MODELS != original_tuple or original.atomic_json is not original_atomic or any(sha(filename) != expected for filename, expected in sources.items()):
        raise ValueError('original evaluator globals/source changed')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--input-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--protocol-plan', type=Path, required=True)
    parser.add_argument('--protocol-registration', type=Path, required=True)
    args = parser.parse_args()
    run(args.input_root, args.out, args.protocol_plan, args.protocol_registration)


if __name__ == '__main__':
    main()
