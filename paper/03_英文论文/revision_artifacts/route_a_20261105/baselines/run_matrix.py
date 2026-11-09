"""Sequential remaining baseline matrix, with immutable-source verification."""
from pathlib import Path
import importlib.util
import json
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('_locked_baseline_matrix', HERE / 'adapter.py')
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)
adapter.verify(json.loads((HERE / 'SOURCE_LOCK.json').read_text())['inputs'])
acceptance = json.loads((HERE / 'FIRST_SEED_ACCEPTANCE.json').read_text())
if acceptance.get('status') != 'PASS_EXECUTION_INTEGRITY':
    raise ValueError('first fixed seed execution preflight required')
for task, (_, seeds) in adapter.TASKS.items():
    for seed in seeds:
        adapter.train(task, seed)
print(json.dumps({'status': 'MATRIX_TRAINING_COMPLETE', 'runs': 12, 'new_seed_selection': False}), flush=True)
