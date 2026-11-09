"""Finite completion of the two already fixed Fashion repetitions, no search."""
from pathlib import Path
from datetime import datetime
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
BASE = 'work/predictive_pfl_confirm'
PLAN = f'{BASE}/PLAN.md'
REG = f'{BASE}/PREREGISTRATION.json'

def execute(arguments, log, completed):
    if (ROOT / completed).is_file():
        result = json.loads((ROOT / completed).read_text())
        if result.get('status') != 'DONE':
            raise ValueError('Existing incomplete result: '+completed)
        print(json.dumps({'event':'REUSE_COMPLETE','path':completed}),flush=True)
        return
    event={'time':datetime.now().astimezone().isoformat(),'event':'START','arguments':arguments,'log':log}
    print(json.dumps(event),flush=True)
    with (ROOT / log).open('w') as stream:
        process=subprocess.run([sys.executable,'-B',*arguments],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
    if process.returncode:
        raise RuntimeError('Failed; inspect '+log)
    print(json.dumps({'event':'END','completed':completed}),flush=True)

def main():
    for seed in (20261242,20261243):
        tag=f'fashion_p20261219_s{seed}'
        train=f'{BASE}/development_{tag}'
        evaluation=f'{BASE}/evaluation_{tag}'
        vector=f'{BASE}/adaptation_{tag}'
        assess=f'{BASE}/assessment_{tag}'
        if seed==20261243:
            execute(['work/calibrated_fashion/train.py','--out',train,'--backend','mps','--seed',str(seed),
                     '--partition-seed','20261219','--rounds','20'],train+'.log',train+'/fedlc_dense/result.json')
        execute(['work/calibrated_fashion/evaluate.py','--model-root',train,'--out',evaluation],
                evaluation+'.log',evaluation+'/result.json')
        execute(['work/prior_vector/evaluate_saved.py','--input-root',evaluation,'--out',vector,
                 '--protocol-plan',PLAN,'--protocol-registration',REG],vector+'.log',vector+'/result.json')
        execute([f'{BASE}/assess.py','--task','fashion','--seed',str(seed),'--evaluation-root',evaluation,
                 '--adaptation-root',vector,'--out',assess],assess+'.log',assess+'/result.json')
        calibration=f'work/calibration_baselines/predictive_{tag}'
        execute(['work/calibration_baselines/probe.py','--input-root',evaluation,'--out',calibration],
                calibration+'.log',calibration+'/result.json')
        bcts=f'{BASE}/bcts_{tag}'
        execute(['work/prior_vector/bcts_control_probe.py','--input-root',evaluation,'--vector-root',vector,
                 '--out',bcts,'--protocol-plan',PLAN,'--protocol-registration',REG],bcts+'.log',bcts+'/result.json')
    print(json.dumps({'status':'DONE','scope':'fixed Fashion seeds61242/43 completed'}),flush=True)

if __name__=='__main__':
    main()
