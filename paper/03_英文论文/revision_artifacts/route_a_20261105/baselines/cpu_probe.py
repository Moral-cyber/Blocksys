"""CPU feasibility timing only; not a training baseline result or paper metric."""
from pathlib import Path
import hashlib
import importlib.util
import json
import time

import torch

HERE = Path(__file__).resolve().parent
source = HERE / 'adapter.py'
spec = importlib.util.spec_from_file_location('_locked_baseline_cpu_probe', source)
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)
lock = json.loads((HERE / 'SOURCE_LOCK.json').read_text())
a.verify(lock['inputs'])
attempts = a.test_guard()
torch.set_num_threads(1)
c, sets, validation, manifest, identities, _, _ = a.checked_data('cifar', 20261231)
torch.manual_seed(20261231)
model = c.runner.SmallCNN()
rows = []
for arm, mu in a.ARMS.items():
    began = time.perf_counter()
    delta, counters = a.local_train(c, model, sets[0], 20261231, 'cpu', mu)
    seconds = time.perf_counter() - began
    rows.append({'arm': arm, 'client': 0, 'train_samples': len(sets[0]),
                 'batches': counters['steps'], 'seconds': seconds,
                 'seconds_per_batch': seconds / counters['steps'],
                 'estimated_one_round_local_seconds': seconds / counters['steps'] * sum((len(ds) + 63) // 64 for ds in sets),
                 'finite_update': bool(torch.isfinite(delta).all())})
a.verify(lock['inputs']); a.verify(identities)
result = {'status': 'CPU_FEASIBILITY_PROBE_COMPLETE', 'scope': 'client timing only; no aggregation/checkpoint/TEST; not a paper baseline',
          'source_adapter_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
          'rows': rows, 'forbidden_open_attempts': attempts, 'official_test_reads': 0}
a.write(HERE / 'CPU_PROBE.json', result)
print(json.dumps(result, indent=2))
