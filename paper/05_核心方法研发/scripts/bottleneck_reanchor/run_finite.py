"""Run exactly four prerecord baseline conditions, no recurring scheduler."""
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent
ENTRY=HERE.parent/'method_followup/run_accelerated.py'


def main():
    for mixing in (0,1):
        argv=[sys.executable,str(ENTRY),'--backend','mps','--dataset','Cifar10IndexA0p1',
            '--split-seed','20261143','--seeds','20261171','--rounds','20','--fraction','.05',
            '--variants','dense,topk_ef','--mode','train','--mixing',str(mixing),
            '--validation-size','2000','--lr','.03','--local-epochs','1','--threads','1',
            '--out',str(HERE/f'mixing{mixing}')]
        subprocess.run(argv,check=True)


if __name__=='__main__':
    main()
