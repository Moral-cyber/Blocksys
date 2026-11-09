"""Root-launched sequential MPS matrix; this file never downloads data."""
from pathlib import Path
from datetime import datetime,timezone
import argparse
import json
import subprocess
import sys

HERE = Path(__file__).resolve().parent
SEEDS = (20261231,20261232,20261233)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw',type=Path,required=True)
    args = parser.parse_args()
    data = json.loads((HERE/'DATA_PREFLIGHT.json').read_text())
    if data['status']!='PASS_DATA_READY' or data['raw']!=str(args.raw.resolve()):
        raise ValueError('matching PASS data preflight required')
    status = {'status':'RUNNING','started_at':datetime.now(timezone.utc).isoformat(),'runs':[]}
    for seed in SEEDS:
        log = HERE/f'cifar100_s{seed}.log'
        print(json.dumps({'status':'START_RUN','task':'cifar100','seed':seed}),flush=True)
        with log.open('a') as stream:
            result = subprocess.run([sys.executable,'-B',str(HERE/'adapter.py'),'--stage','run','--seed',str(seed),
                                     '--raw',str(args.raw)],stdout=stream,stderr=subprocess.STDOUT)
        status['runs'].append({'seed':seed,'exit_code':result.returncode,'log':str(log),
                               'finished_at':datetime.now(timezone.utc).isoformat()})
        status['status'] = 'RUNNING' if result.returncode==0 else 'FAILED'
        (HERE/'MATRIX_STATUS.json').write_text(json.dumps(status,indent=2)+'\n')
        if result.returncode:
            raise SystemExit(result.returncode)
    result = subprocess.run([sys.executable,'-B',str(HERE/'adapter.py'),'--stage','summarize'])
    status['status'] = 'DONE' if result.returncode==0 else 'FAILED_SUMMARY'
    status['finished_at'] = datetime.now(timezone.utc).isoformat()
    (HERE/'MATRIX_STATUS.json').write_text(json.dumps(status,indent=2)+'\n')
    raise SystemExit(result.returncode)


if __name__=='__main__':
    main()
