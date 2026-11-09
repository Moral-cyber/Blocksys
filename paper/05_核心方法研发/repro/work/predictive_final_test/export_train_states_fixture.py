"""CPU synthetic checks for TRAIN-state copying and exact frozen head replay."""
from pathlib import Path
import copy
import json
import tempfile

import numpy as np
import torch
from torch.utils.data import TensorDataset

import export_train_states as e


def main():
    torch.set_num_threads(1)
    attempts = e.install_no_test_guard()
    common, frozen = e.runtime('cifar')
    before = {str(e.ROOT / name): e.sha(e.ROOT / name) for name in e.PINS}
    protected = common.source_identities()
    scratch = Path(tempfile.mkdtemp(prefix='train_state_export_fixture_', dir=e.ROOT / 'work'))
    torch.manual_seed(20269901)
    model = common.runner.SmallCNN().eval()
    model_before = frozen.model_hash(model)
    maximum = 0.
    verified = 0
    examples = []
    for client in range(20):
        torch.manual_seed(20269910 + client)
        train_x = torch.randn(4 + client % 3, 3, 32, 32)
        train_y = torch.randint(0, 10, (len(train_x),))
        hold_x = torch.randn(3 + client % 2, 3, 32, 32)
        hold_y = torch.randint(0, 10, (len(hold_x),))
        training, holdout = TensorDataset(train_x, train_y), TensorDataset(hold_x, hold_y)
        seed = 20261231 + 100000 + client
        expected_head, _ = frozen.matched_head(model, training, seed)
        expected_raw, labels, _ = frozen.forward(model, holdout)
        expected_learned, _, _ = frozen.forward(expected_head, holdout)
        weight, bias, cost = e.rebuild_head(frozen, model, training, holdout, seed,
                                          expected_raw, expected_learned, labels)
        assert np.array_equal(weight, expected_head.classifier[-1].weight.detach().numpy())
        assert np.array_equal(bias, expected_head.classifier[-1].bias.detach().numpy())
        assert cost['observed']['head_backward_calls'] == 1
        assert cost['observed']['train_forward_samples'] == len(training)
        assert cost['observed']['verification_forward_samples'] == 2 * len(holdout)
        assert cost['body_backward_calls'] == cost['new_calibration_fits'] == cost['new_FL_rounds'] == 0
        maximum = max(maximum, cost['holdout_learned_max_abs_error'], cost['holdout_raw_max_abs_error'])
        verified += 1
        examples.append((training, holdout, seed, expected_raw, expected_learned, labels))
    assert frozen.model_hash(model) == model_before and maximum == 0
    # Both real wrappers must import and retain the same frozen head code.
    fashion_common, fashion_frozen = e.runtime('fashion')
    assert fashion_frozen.matched_head.__code__.co_code == frozen.matched_head.__code__.co_code
    assert fashion_common.prepare.__module__ == '_train_state_fashion_adapter'
    for name in fashion_common.source_identities():
        protected[name] = e.sha(name)

    rng = np.random.default_rng(20269950)
    y = np.array([0, 0, 2, 8, 8, 8], dtype=np.int64)
    counts = np.bincount(y, minlength=10)
    base = {'train_logits': rng.normal(size=(6, 10)).astype(np.float32), 'train_labels': y,
            'train_counts': counts, 'train_ids': np.arange(6, dtype=np.uint32),
            'holdout_ids': np.arange(6, 10, dtype=np.uint32),
            'raw': rng.normal(size=(4, 10)).astype(np.float32),
            'learned': rng.normal(size=(4, 10)).astype(np.float32),
            'prior_shift': np.log((counts + 1.) / 16).astype(np.float32),
            'fitted_bias': np.linspace(-.3, .3, 10, dtype=np.float32)}
    common_arrays = {'train_ids': base['train_ids'], 'holdout_ids': base['holdout_ids'],
                     'prioranchor_bias': base['fitted_bias']}
    zero_vector = np.r_[np.ones(10), np.zeros(10)].astype(np.float32)
    prior_vector = np.r_[np.ones(10), base['fitted_bias']].astype(np.float32)
    vector = {**common_arrays, 'zero_anchor_vector': zero_vector, 'prioranchored_vector': prior_vector}
    calibration = {**common_arrays, 'zero_anchor_bias': np.zeros(10, dtype=np.float32),
                   'temperature_scaling': np.ones(1, dtype=np.float32),
                   'bcts': np.r_[1., np.zeros(10)].astype(np.float32), 'vector_scaling': zero_vector}
    bcts = {**vector, 'prioranchored_bcts': np.r_[1., base['fitted_bias']].astype(np.float32)}
    states, actual = e.collect_states(base, vector, calibration, bcts)
    assert np.array_equal(actual, counts) and set(states) == set(e.STATE_DIMENSIONS)
    target = scratch / 'client.npz'
    np.savez_compressed(target, head_weight=weight, head_bias=bias, train_ids=base['train_ids'],
                        holdout_ids=base['holdout_ids'], train_counts=actual,
                        **{'state_' + name: value for name, value in states.items()})
    saved = e.arrays(target)
    assert all(np.array_equal(saved['state_' + name], value) for name, value in states.items())
    assert all(not np.shares_memory(states[name], value) for name, value in (
        ('analytic_prior', base['prior_shift']), ('prioranchored_vector', vector['prioranchored_vector'])))
    # Direct compatibility with the separately frozen final scorer.
    import scorer
    scorer.validate_states([states for _ in range(20)], np.repeat(counts[None, :], 20, axis=0))
    negative = []
    for case, mutate in (
            ('FP64_deployment_state', lambda a: a[1].update(prioranchored_vector=prior_vector.astype(np.float64))),
            ('wrong_prior', lambda a: a[0]['prior_shift'].__setitem__(0, 0.)),
            ('membership_mismatch', lambda a: a[2]['train_ids'].__setitem__(0, 99)),
            ('fractional_counts', lambda a: a[0].update(train_counts=counts.astype(np.float64))),
            ('nonpositive_temperature', lambda a: a[2]['temperature_scaling'].__setitem__(0, 0.)),
            ('paired_vector_mismatch', lambda a: a[3]['zero_anchor_vector'].__setitem__(0, 2.))):
        values = [copy.deepcopy(x) for x in (base, vector, calibration, bcts)]
        mutate(values)
        try:
            e.collect_states(*values)
        except ValueError as error:
            negative.append({'case': case, 'error': str(error)})
        else:
            raise AssertionError('invalid state accepted: ' + case)
    training, holdout, seed, raw, learned, labels = examples[0]
    try:
        e.rebuild_head(frozen, model, training, holdout, seed, raw, learned + .01, labels)
    except ValueError as error:
        assert 'exceeds fixed1e-6' in str(error)
        negative.append({'case': 'changed_saved_head_logits', 'error': str(error)})
    else:
        raise AssertionError('changed saved head logits accepted')
    try:
        e.run('cifar', scratch, scratch, scratch, scratch, scratch, scratch)
    except ValueError as error:
        assert 'fresh' in str(error)
        negative.append({'case': 'existing_output', 'error': str(error)})
    else:
        raise AssertionError('existing output accepted')
    assert all(e.sha(path) == digest for path, digest in {**before, **protected}.items())
    assert not attempts
    evidence = {'status': 'PASS', 'scope': 'CPU synthetic head replay/state-copy contract only',
                'scratch': str(scratch), 'clients_with_exact_head_weight_bias_replay': verified,
                'max_raw_and_head_holdout_error': maximum, 'head_backward_observed_per_client': 1,
                'same_frozen_matched_head_code_cifar_fashion': True,
                'copied_calibration_families': list(states), 'scorer_state_API_accepted': True,
                'negative_checks': negative, 'source_hashes_unchanged': {**before, **protected},
                'exporter_sha256': e.sha(e.__file__), 'fixture_sha256': e.sha(__file__),
                'official_test_open_attempts': attempts, 'official_test_bytes': 0,
                'real_TRAIN_export_run': False, 'new_FL_rounds': 0, 'new_calibration_fits': 0,
                'external_review': 'REVIEW_UNAVAILABLE'}
    e.atomic_json(e.HERE / 'EXPORT_TRAIN_STATES_FIXTURES.json', evidence)
    print(json.dumps(evidence, indent=2))


if __name__ == '__main__':
    main()
