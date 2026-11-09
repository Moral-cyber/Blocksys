"""Sequential process-isolated launcher for the twelve predeclared runs."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
MATRIX = [(task, partition, seed)
          for task, seeds in [('cifar', (20261231, 20261232, 20261233)),
                              ('fashion', (20261241, 20261242, 20261243))]
          for partition in (20261105, 20261106) for seed in seeds]

def write(value):
    p = HERE/'MATRIX_STATUS.json'; temp = p.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2)+'\n'); temp.replace(p)

record = {'started_at': datetime.now(timezone.utc).isoformat(), 'status': 'RUNNING',
          'launcher_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'runs': []}
for task, partition, seed in MATRIX:
    args = [sys.executable, '-B', str(HERE/'adapter.py'), '--stage', 'run', '--task', task,
            '--partition', str(partition), '--seed', str(seed)]
    log = HERE/f'{task}_p{partition}_s{seed}.log'
    with log.open('a') as stream:
        stream.write('\nCOMMAND '+json.dumps(args)+'\n'); stream.flush()
        print(json.dumps({'status': 'START_RUN', 'task': task, 'partition': partition, 'seed': seed}), flush=True)
        result = subprocess.run(args, stdout=stream, stderr=subprocess.STDOUT)
    record['runs'].append({'task': task, 'partition': partition, 'seed': seed, 'exit_code': result.returncode,
                           'log': str(log), 'finished_at': datetime.now(timezone.utc).isoformat()})
    if result.returncode:
        record['status'] = 'FAILED'; write(record)
        print(log.read_text()[-6000:], flush=True); sys.exit(result.returncode)
    write(record)
    print(json.dumps({'status': 'FINISHED_RUN', 'task': task, 'partition': partition, 'seed': seed}), flush=True)
result = subprocess.run([sys.executable, '-B', str(HERE/'adapter.py'), '--stage', 'summarize'])
record['status'] = 'DONE' if result.returncode == 0 else 'SUMMARY_FAILED'
record['finished_at'] = datetime.now(timezone.utc).isoformat(); write(record)
sys.exit(result.returncode)
