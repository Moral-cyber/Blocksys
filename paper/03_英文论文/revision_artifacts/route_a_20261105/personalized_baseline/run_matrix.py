"""Serial six-run pFedMe training + model freeze; deliberately has no scoring path."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
PYTHON = Path('/opt/anaconda3/envs/Tinyproto/bin/python')
ADAPTER = HERE / 'adapter.py'
MATRIX = [('cifar', seed) for seed in (20261231, 20261232, 20261233)] + [
    ('fashion', seed) for seed in (20261241, 20261242, 20261243)
]


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    temporary = Path(path).with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def frozen_inputs():
    lock = read(HERE / 'SOURCE_LOCK.json')
    for filename, expected in lock['inputs'].items():
        path = Path(filename)
        if not path.is_file() or sha(path) != expected:
            raise ValueError(f'Frozen pFedMe input changed/missing: {path}')
    if str(ADAPTER) not in lock['inputs']:
        raise ValueError('Existing source lock does not bind this adapter')
    if not PYTHON.is_file():
        raise ValueError('Required Tinyproto interpreter is unavailable; no runtime fallback')
    if read(HERE / 'FIXTURES.json')['status'] != 'PASS':
        raise ValueError('pFedMe fixtures have not passed')
    preflight = read(HERE / 'DATA_PREFLIGHT.json')
    if preflight['status'] != 'PASS':
        raise ValueError('Original data membership preflight has not passed')
    return {'runner_sha256': sha(Path(__file__)),
            'adapter_sha256': sha(ADAPTER),
            'source_lock_sha256': sha(HERE / 'SOURCE_LOCK.json'),
            'source_input_count': len(lock['inputs']),
            'python': str(PYTHON), 'matrix': MATRIX,
            'training_then_model_lock_only': True, 'official_test_scoring': False}


def existing_runs():
    values = []
    for task, seed in MATRIX:
        output = HERE / 'runs' / task / f's{seed}' / 'pfedme_dense'
        result = read(output / 'result.json') if (output / 'result.json').exists() else {}
        values.append({'task': task, 'seed': seed, 'status': result.get('status', 'NOT_STARTED'),
                       'rounds': len(result.get('rows', [])),
                       'checkpoint_exists': (output / 'checkpoint.pt').is_file()})
    return values


def run_command(command, log_path):
    started = now()
    clock = time.monotonic()
    with log_path.open('a', buffering=1) as stream:
        stream.write(json.dumps({'actual_command': command, 'started_at': started}) + '\n')
        process = subprocess.Popen(command, cwd=HERE, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1,
                                   env={**os.environ, 'PYTHONUNBUFFERED': '1'})
        try:
            for line in process.stdout:
                stream.write(line)
                print(line, end='', flush=True)
            code = process.wait()
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise
    return {'argv': command, 'started_at': started, 'completed_at': now(),
            'elapsed_seconds': time.monotonic() - clock, 'exit_code': code,
            'log_path': str(log_path), 'log_sha256': sha(log_path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true',
                        help='Run six fixed MPS train/resume commands serially, then model-lock')
    args = parser.parse_args()
    identity = frozen_inputs()
    if not args.execute:
        print(json.dumps({'status': 'PREFLIGHT_PASS_NO_EXECUTION',
                          'identity': identity, 'existing_runs': existing_runs()}, indent=2))
        return 0
    output = HERE / 'matrix_execution'
    output.mkdir(exist_ok=True)
    with (output / 'runner.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        source_lock = output / 'MATRIX_SOURCE_LOCK.json'
        if source_lock.exists():
            if read(source_lock) != json.loads(json.dumps(identity)):
                raise ValueError('Matrix runner/source identity changed; do not overwrite existing run')
        else:
            write(source_lock, identity)
        attempt = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        receipt = output / f'run_{attempt}.json'
        report = {'status': 'RUNNING', 'started_at': now(), 'identity': identity,
                  'commands': [], 'official_test_scoring': False,
                  'selection': 'Fixed six seeds/final20; no result-driven choices'}
        write(receipt, report)
        try:
            for task, seed in MATRIX:
                frozen_inputs()
                command = [str(PYTHON), str(ADAPTER), '--stage', 'train',
                           '--task', task, '--seed', str(seed)]
                item = run_command(command, output / f'{attempt}_{task}_s{seed}.log')
                report['commands'].append(item)
                write(receipt, report)
                if item['exit_code']:
                    raise RuntimeError(f'pFedMe train failed: {task}/{seed}; resume only after resolving the recorded failure')
                result = read(HERE / 'runs' / task / f's{seed}' / 'pfedme_dense/result.json')
                if result['status'] != 'DONE' or len(result['rows']) != 20 or result['official_test_reads'] != 0:
                    raise ValueError(f'Incomplete/invalid final20 result: {task}/{seed}')
            frozen_inputs()
            command = [str(PYTHON), str(ADAPTER), '--stage', 'model-lock']
            item = run_command(command, output / f'{attempt}_model_lock.log')
            report['commands'].append(item)
            write(receipt, report)
            if item['exit_code']:
                raise RuntimeError('pFedMe model-lock failed')
            model_lock = read(HERE / 'MODEL_LOCK.json')
            if model_lock['status'] != 'ALL_SIX_PERSONAL_MODELS_FROZEN':
                raise ValueError('Unexpected model-lock status')
            report.update(status='ALL_SIX_TRAINED_AND_MODEL_LOCKED_NO_TEST_SCORING',
                          completed_at=now(), model_lock_sha256=sha(HERE / 'MODEL_LOCK.json'),
                          runs=existing_runs())
            write(receipt, report)
            print(json.dumps({'status': report['status'], 'receipt': str(receipt)}, indent=2))
            return 0
        except BaseException as error:
            report.update(status='FAILED_OR_INTERRUPTED', completed_at=now(),
                          error=f'{type(error).__name__}: {error}', runs=existing_runs())
            write(receipt, report)
            raise


if __name__ == '__main__':
    sys.exit(main())
