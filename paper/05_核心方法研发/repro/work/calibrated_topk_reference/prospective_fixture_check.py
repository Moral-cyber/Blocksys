"""Prospective wrapper CPU toy; no official data preparation or formal training."""
from pathlib import Path
from types import SimpleNamespace
import contextlib
import hashlib
import io
import json
import tempfile
import numpy as np
import torch
import common as c
import prospective as p
import prospective_evaluate as pe

torch.set_num_threads(1)
OUT = Path(tempfile.mkdtemp(prefix='prospective_topk_fixture_'))
PLAN = c.HERE.parent / 'predictive_pfl_confirm/PLAN.md'
context0, _ = p.make_context(PLAN, p.SEEDS[0], fixture=True)
sources = context0.source_identities()
original_source_function, original_runtime_function = c.source_identities, c.accelerator.runtime_metadata
original_train_globals = (p.fixed_train.run.__globals__['c'], p.fixed_train.run.__globals__['wire'])
original_evaluate_context = pe.fixed_evaluate.run.__globals__['c']
report = {'scope': 'synthetic CPU wrapper integration only; no official images/TEST or formal preparation/training',
    'fixture_out': str(OUT), 'protocol_plan': str(PLAN.resolve()), 'protocol_plan_sha256': p.PLAN_SHA,
    'source_identities': sources, 'checks': {}}

synthetic_labels = np.repeat(np.arange(10), 5000)
partition1 = c.partitions.partition_indices(synthetic_labels, alpha=.1, split_seed=p.PARTITION, mixing=0., validation_size=2000)
partition2 = c.partitions.partition_indices(synthetic_labels, alpha=.1, split_seed=p.PARTITION, mixing=0., validation_size=2000)
assert partition1[3] == partition2[3] and partition1[3]['partition_seed'] == p.PARTITION
report['checks']['pure_ID_new_partition'] = {'partition_seed': p.PARTITION,
    'deterministic': True, 'synthetic_train_samples': partition1[3]['train_samples'],
    'synthetic_holdout_samples': partition1[3]['holdout_samples'], 'original_images_read': 0}

rng = np.random.default_rng(61217)
images = rng.integers(0, 256, size=(12, 32, 32, 3), dtype=np.uint8)
labels = np.array([0, 0, 1, 2, 2, 2, 4, 4, 0, 1, 2, 4])
train_ids, hold_ids = [[0, 1, 2], [3, 4, 5, 6, 7]], [[8, 9], [10, 11]]
manifest = {'fixture_only': True, 'partition_seed': p.PARTITION, 'train_indices': train_ids,
    'holdout_indices': hold_ids, 'validation_indices': [8, 9, 10, 11],
    'client_label_counts': [np.bincount(labels[ids], minlength=10).tolist() for ids in train_ids]}


class Logged(c.partitions.IndexedCifar10):
    def __init__(self, *arguments):
        super().__init__(*arguments)
        self.order = []

    def __getitem__(self, i):
        self.order.append(int(self.indices[i]))
        return super().__getitem__(i)


datasets = [Logged(images, labels, ids) for ids in train_ids]
validation = c.partitions.IndexedCifar10(images, labels, [8, 9, 10, 11])
seed_records, signatures = {}, {}
for seed in p.SEEDS:
    context, metadata = p.make_context(PLAN, seed, fixture=True)
    captured, active = {}, {}

    def observed_train(model, dataset, batch_seed, backend, loss):
        dataset.order.clear()
        anchor = c.runner.vectorize(model).clone()
        result = c.balanced.train(model, dataset, batch_seed, backend, loss)
        active['calls'].append({'seed': batch_seed, 'order': dataset.order.copy(), 'anchor': anchor})
        return result

    # Observation hooks live only in this copied context, never original globals.
    context.balanced = SimpleNamespace(train=observed_train)
    bound = p.bind(p.fixed_train.run, context)
    assert bound.__code__ is p.fixed_train.run.__code__ and bound.__globals__ is not p.fixed_train.run.__globals__
    args = SimpleNamespace(out=OUT / f'seed{seed}', seed=seed, partition_seed=p.PARTITION, rounds=2, backend='cpu')
    for method in c.METHODS:
        active['calls'] = []
        with contextlib.redirect_stdout(io.StringIO()):
            bound(args, method, datasets, validation, manifest, {})
        captured[method] = active['calls']
        folder = args.out / method
        config = json.loads((folder / 'config.json').read_text())
        checkpoint = torch.load(folder / 'checkpoint.pt', weights_only=True)
        assert config['seed'] == seed and config['partition_seed'] == p.PARTITION
        assert config['source_identities'] == sources
        assert config['runtime']['prospective_reference'] == metadata
        assert config['signature'] == c.signature({k: v for k, v in config.items() if k != 'signature'}) == checkpoint['signature']
        assert config['upload_cap_bytes'] == 54808 and config['selected_pairs'] == 6843
        assert len((folder / 'first_payload.bin').read_bytes()) == 54808
        assert [item['seed'] for item in captured[method]] == [seed, seed + 1, seed + 100, seed + 101]
        signatures.setdefault(method, []).append(config['signature'])
    assert [(r['seed'], r['order']) for r in captured['ce_topk']] == [(r['seed'], r['order']) for r in captured['fedlc_topk']]
    assert torch.equal(captured['ce_topk'][0]['anchor'], captured['fedlc_topk'][0]['anchor'])
    seed_records[str(seed)] = {'config_seed': seed, 'partition_seed': p.PARTITION, 'toy_rounds': 2,
        'formal_rounds': p.ROUNDS, 'same_batch_seed_order_and_initialization': True,
        'signature_binds_explicit_source_runtime_overrides': True, 'actual_wire_per_client': 54808}
assert all(len(set(values)) == 3 for values in signatures.values())
report['checks']['three_allowed_seed_contexts'] = seed_records

bad_plan = OUT / 'wrong_PLAN.md'
bad_plan.write_text('not the frozen prospective plan')
rejected = []
for name, plan, seed in [('wrong_plan', bad_plan, p.SEEDS[0]), ('old_seed', PLAN, 20261211)]:
    try:
        p.make_context(plan, seed, fixture=True)
    except ValueError:
        rejected.append(name)
    else:
        raise AssertionError('accepted ' + name)
first_root = OUT / f'seed{p.SEEDS[0]}'
first_args = SimpleNamespace(out=first_root, seed=p.SEEDS[0], partition_seed=20261203, rounds=2, backend='cpu')
try:
    p.run(first_args, datasets, validation, manifest, {}, PLAN, fixture=True)
except ValueError:
    rejected.append('old_partition')
else:
    raise AssertionError('accepted old partition')
config_path = first_root / 'ce_topk/config.json'
original_bytes = config_path.read_bytes()
bad_config = json.loads(original_bytes)
bad_config['runtime']['prospective_reference']['partition_seed_override'] = 20261203
config_path.write_text(json.dumps(bad_config))
try:
    try:
        pe.run(first_root, OUT / 'must_not_create', images, labels, manifest, {}, PLAN, fixture=True)
    except ValueError:
        rejected.append('runtime_override_mismatch_before_forward')
    else:
        raise AssertionError('accepted bad runtime override')
finally:
    config_path.write_bytes(original_bytes)
assert not (OUT / 'must_not_create').exists()
report['checks']['negative_protocol_rejections'] = rejected

with contextlib.redirect_stdout(io.StringIO()):
    result = pe.run(first_root, OUT / 'evaluation', images, labels, manifest, {}, PLAN, fixture=True)
assert result['training_seed'] == 20261231 and result['partition_seed'] == p.PARTITION
assert result['decision'] == 'REFERENCE_ONLY' and result['gates'] is None
assert result['prospective_protocol']['protocol_plan_sha256'] == p.PLAN_SHA
assert result['source_identities'] == sources and result['bias_total_train_image_forwards'] == 16
report['checks']['standalone_generic_evaluation'] = {'new_partition_and_seed_accepted': True,
    'source_runtime_override_checked': True, 'old_fixed_main_not_called': True,
    'saved_array_schema_unchanged': True, 'decision': result['decision'], 'gates': result['gates']}
assert c.source_identities is original_source_function
assert c.accelerator.runtime_metadata is original_runtime_function
assert p.fixed_train.run.__globals__['c'] is original_train_globals[0]
assert p.fixed_train.run.__globals__['wire'] is original_train_globals[1]
assert pe.fixed_evaluate.run.__globals__['c'] is original_evaluate_context
context0.verify_identities(sources)
report.update({'status': 'PASS', 'original_common_train_evaluate_globals_unchanged': True,
    'source_identities_unchanged': True, 'formal_ML_training_started': False,
    'harness_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'review': 'REVIEW_UNAVAILABLE'})
(c.HERE / 'PROSPECTIVE_FIXTURES.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'status': 'PASS', 'fixture_out': str(OUT), 'seeds': list(p.SEEDS),
                  'partition_seed': p.PARTITION, 'original_globals_unchanged': True,
                  'formal_ML_training_started': False}, ensure_ascii=False))
