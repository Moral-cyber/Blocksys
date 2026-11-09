"""Export the five completed task/seed predictor sets, no new FL training."""
from pathlib import Path
import json
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
BASE='work/predictive_pfl_confirm'

def main():
    for task,partition,seeds in (('cifar',20261217,(20261231,20261232,20261233)),
                                 ('fashion',20261219,(20261241,20261242))):
        for seed in seeds:
            tag=f'{task}_p{partition}_s{seed}'
            out=f'work/predictive_final_test/train_states_{tag}'
            if (ROOT/out/'result.json').is_file():
                if json.loads((ROOT/out/'result.json').read_text()).get('status')!='DONE':
                    raise ValueError('Incomplete existing result')
                continue
            command=[sys.executable,'-B','work/predictive_final_test/export_train_states.py',
                '--task',task,'--model-root',f'{BASE}/development_{tag}',
                '--evaluation-root',f'{BASE}/evaluation_{tag}',
                '--adaptation-root',f'{BASE}/adaptation_{tag}',
                '--calibration-root',f'work/calibration_baselines/predictive_{tag}',
                '--bcts-root',f'{BASE}/bcts_{tag}','--out',out]
            print(json.dumps({'event':'START','task':task,'seed':seed,'out':out}),flush=True)
            with (ROOT/(out+'.log')).open('w') as log:
                subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
            print(json.dumps({'event':'DONE','task':task,'seed':seed}),flush=True)

if __name__=='__main__':main()
