"""Finite, logged reference repetitions: seeds61232/33 only, no parameter search."""
from pathlib import Path
import argparse
from datetime import datetime
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--reference', choices=('topk', 'fedrod'), required=True)
    args = parser.parse_args()
    plan = 'work/predictive_pfl_confirm/PLAN.md'
    registration = 'work/predictive_pfl_confirm/PREREGISTRATION.json'
    journal = ROOT / f'work/predictive_pfl_confirm/reference_{args.reference}_runner.jsonl'
    def execute(arguments, log):
        before = {'time': datetime.now().astimezone().isoformat(), 'event': 'START', 'arguments': arguments, 'log': log}
        with journal.open('a') as handle:
            handle.write(json.dumps(before) + '\n')
        print(json.dumps(before), flush=True)
        with (ROOT / log).open('w') as stream:
            process = subprocess.run([sys.executable, '-B', *arguments], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        after = {'time': datetime.now().astimezone().isoformat(), 'event': 'END', 'returncode': process.returncode, 'log': log}
        with journal.open('a') as handle:
            handle.write(json.dumps(after) + '\n')
        print(json.dumps(after), flush=True)
        if process.returncode:
            raise RuntimeError('Reference failed; inspect ' + log)
    for seed in (20261232, 20261233):
        if args.reference == 'topk':
            base = 'work/calibrated_topk_reference'
            train = f'{base}/development_p20261217_s{seed}'
            evaluation = f'{base}/evaluation_p20261217_s{seed}'
            adaptation = f'{base}/adaptation_p20261217_s{seed}'
            execute([f'{base}/prospective.py', '--out', train, '--protocol-plan', plan,
                     '--seed', str(seed), '--partition-seed', '20261217', '--rounds', '20', '--backend', 'mps'], train + '.log')
            execute([f'{base}/prospective_evaluate.py', '--model-root', train, '--out', evaluation,
                     '--protocol-plan', plan], evaluation + '.log')
            execute([f'{base}/evaluate_saved_prior_vector.py', '--input-root', evaluation, '--out', adaptation,
                     '--protocol-plan', plan, '--protocol-registration', registration], adaptation + '.log')
        else:
            base = 'work/fedrod_linear_reference'
            train = f'{base}/development_p20261217_s{seed}'
            quality = f'{base}/quality_p20261217_s{seed}'
            execute([f'{base}/prospective.py', '--partition-seed', '20261217', '--seed', str(seed),
                     '--backend', 'mps', '--out', train, '--execute-formal'], train + '.log')
            execute([f'{base}/export_quality.py', '--model-root', train, '--out', quality], quality + '.log')
    print(json.dumps({'status': 'DONE', 'reference': args.reference, 'training_seeds': [20261232, 20261233]}), flush=True)


if __name__ == '__main__':
    main()
