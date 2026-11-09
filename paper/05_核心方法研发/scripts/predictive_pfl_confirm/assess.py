"""Assess the separately preregistered NLL question; old STOPs are immutable."""
from pathlib import Path
import argparse
import hashlib
import json

HERE = Path(__file__).resolve().parent
PLAN_SHA = '703d5956c9edd24d5a03501a75f678a43222d19bd3dd1ede515ed77545b0830b'
TASKS = {'cifar': (20261217, (20261231, 20261232, 20261233)),
         'fashion': (20261219, (20261241, 20261242, 20261243))}
METHODS = {'ce_uniform3', 'fedlc_uniform3', 'fedlc_dense'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def assess(results):
    summaries = {method: row['summaries'] for method, row in results.items()}
    candidate = summaries['fedlc_uniform3']['prioranchored_vector']
    controls = {'zero_vector': summaries['fedlc_uniform3']['zero_anchor_vector'],
                'prior': summaries['fedlc_uniform3']['analytic_prior'],
                'ten_bias': summaries['fedlc_uniform3']['prioranchor_bias'],
                'ce_vector': summaries['ce_uniform3']['prioranchored_vector'],
                'dense_vector': summaries['fedlc_dense']['prioranchored_vector']}
    differences = {name: {'delta_accuracy_pp': 100 * (candidate['macro_accuracy'] - ref['macro_accuracy']),
                          'nll_reduction': ref['macro_loss'] - candidate['macro_loss']}
                   for name, ref in controls.items()}
    checks = {
        'zero_vector_nll_reduction_at_least_001': differences['zero_vector']['nll_reduction'] >= .01,
        'zero_vector_accuracy_within_half_pp': differences['zero_vector']['delta_accuracy_pp'] >= -.5,
        'prior_nll_reduction_at_least_001': differences['prior']['nll_reduction'] >= .01,
        'prior_accuracy_within_02pp': differences['prior']['delta_accuracy_pp'] >= -.2,
        'ten_bias_nll_within_0005': differences['ten_bias']['nll_reduction'] >= -.005,
        'ten_bias_accuracy_within_half_pp': differences['ten_bias']['delta_accuracy_pp'] >= -.5,
        'ce_vector_nll_within_001': differences['ce_vector']['nll_reduction'] >= -.01,
        'ce_vector_accuracy_within_half_pp': differences['ce_vector']['delta_accuracy_pp'] >= -.5,
        'dense_vector_nll_within_001': differences['dense_vector']['nll_reduction'] >= -.01,
        'dense_vector_accuracy_within_half_pp': differences['dense_vector']['delta_accuracy_pp'] >= -.5}
    return candidate, differences, checks


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--task', choices=TASKS, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--evaluation-root', type=Path, required=True)
    parser.add_argument('--adaptation-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    partition, seeds = TASKS[args.task]
    if args.seed not in seeds or args.out.exists():
        raise ValueError('predeclared seed and fresh output required')
    plan = HERE / 'PLAN.md'
    registration = HERE / 'PREREGISTRATION.json'
    if sha(plan) != PLAN_SHA:
        raise ValueError('prospective protocol changed')
    registered = json.loads(registration.read_text())
    root = HERE.parents[1]
    for relative, expected in registered['frozen_sources'].items():
        if sha(root / relative) != expected:
            raise ValueError('preregistered method source changed: ' + relative)
    evaluation_path = args.evaluation_root.resolve() / 'result.json'
    adaptation_path = args.adaptation_root.resolve() / 'result.json'
    evaluation = json.loads(evaluation_path.read_text())
    adapted = json.loads(adaptation_path.read_text())
    if (evaluation['status'] != 'DONE' or adapted['status'] != 'DONE'
            or evaluation['training_seed'] != args.seed
            or evaluation['partition_seed'] != partition
            or evaluation['training_rounds'] != 20
            or set(adapted['results']) != METHODS):
        raise ValueError('complete paired prospective input required')
    if (evaluation.get('official_test_byte_reads') is not False
            or evaluation.get('official_test_inference') is not False):
        raise ValueError('development-only TEST guard required')
    frozen_inputs = adapted.get('input_sha256_before_after_equal', {})
    if (Path(adapted.get('input_root', '')).resolve() != args.evaluation_root.resolve()
            or frozen_inputs.get(str(evaluation_path)) != sha(evaluation_path)
            or adapted.get('protocol_plan_sha256') != PLAN_SHA
            or adapted.get('protocol_registration') != str(registration.resolve())):
        raise ValueError('adaptation must bind this exact evaluation and prospective protocol')
    candidate, differences, checks = assess(adapted['results'])
    inputs = {str(path.resolve()): sha(path) for path in
              (plan, registration, Path(__file__), evaluation_path, adaptation_path)}
    decision = 'FIRST_SEED_FOLLOW_UP' if all(checks.values()) else 'FIRST_SEED_STOP'
    if args.seed != seeds[0]:
        decision = 'PLANNED_REPETITION_RECORDED; task-level aggregation required'
    result = {'status': 'DONE', 'decision': decision,
              'question': 'prospective personalized NLL at preserved accuracy; earlier accuracy STOP unchanged',
              'task': args.task, 'seed': args.seed, 'partition_seed': partition,
              'candidate': candidate, 'differences': differences, 'first_seed_conditions': checks,
              'source_identities': inputs, 'scope': 'TRAIN internal development holdout, not official TEST',
              'rounds': 20, 'original_evaluation_accuracy_decision_unchanged': evaluation['decision'],
              'external_cross_model_review': 'REVIEW_UNAVAILABLE'}
    args.out.mkdir(parents=True)
    (args.out / 'result.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
