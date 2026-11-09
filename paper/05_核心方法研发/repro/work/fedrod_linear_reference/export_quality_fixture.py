"""Synthetic nonzero-P final-export fixture; no TRAIN fitting or real images."""
from pathlib import Path
import hashlib
import io
import json
import tempfile
import numpy as np
import torch
import torch.nn.functional as F
import export_quality as e

torch.set_num_threads(1)
r, p = e.r, e.p
OUT = Path(tempfile.mkdtemp(prefix='fedrod_quality_fixture_'))
protected = e.own_sources()
attempts = e.install_guard()
try:
    Path('/tmp/fedrod_quality_dummy_test_batch').read_bytes()
except ValueError:
    pass
else:
    raise AssertionError('TEST guard did not reject dummy path')
assert len(attempts) == 1
entry = p.load_frozen_entry()
with p.module_overrides(entry, 20261231, p.PARTITION_SEED):
    base = p.base_config(entry, 'cpu', 20261231, p.PARTITION_SEED)
# Importing the strict adapter is harmless; prepare() is deliberately never called.
adapter = entry.load_train_adapter()
rng = np.random.default_rng(6131)
images = rng.integers(0, 256, size=(70, 32, 32, 3), dtype=np.uint8)
labels = (np.arange(70) * 3 % 10).astype(np.int64)
train_ids = [list(range(2 * i, 2 * i + 2)) for i in range(20)]
hold_ids, start = [], 40
for i in range(20):
    count = 1 if i < 10 else 2
    hold_ids.append(list(range(start, start + count)))
    start += count
manifest = {'fixture_only': True, 'partition_seed': p.PARTITION_SEED,
    'train_indices': train_ids, 'holdout_indices': hold_ids,
    'client_sizes': [2] * 20, 'client_holdout_sizes': list(map(len, hold_ids)),
    'client_label_counts': [np.bincount(labels[ids], minlength=10).tolist() for ids in train_ids]}
config = {'base': base, 'weights': [.05] * 20, 'data_manifest': manifest,
    'data_identities': {}, 'train_adapter_identities': adapter.source_identities()}
config['signature'] = r.signature(config)
model = r.new_model(20261231)
heads = [r.new_head() for _ in range(20)]
for i, head in enumerate(heads):
    with torch.no_grad():
        head.weight.copy_(torch.from_numpy(rng.normal(0, .01, size=(10, 64)).astype(np.float32)))
        head.bias[i % 10] = .25
assert all(torch.count_nonzero(head.weight) > 0 for head in heads)
rows = [{'round': i + 1, 'fixture_only_no_training': True} for i in range(20)]
model_root = OUT / 'synthetic_complete_checkpoint'
model_root.mkdir()
r.atomic_json(model_root / 'config.json', config)
r.save_checkpoint(model_root / 'checkpoint.pt', config, model, heads,
    [torch.zeros(r.PARAMETERS, dtype=torch.float32) for _ in range(20)], rows)
r.atomic_json(model_root / 'result.json', {'status': 'DONE', 'signature': config['signature'],
    'rows': rows, 'formal_training_executed': False, 'fixture_only': True})

# Final checkpoint/config rejection occurs before any feature forward.
checkpoint_path, config_path = model_root / 'checkpoint.pt', model_root / 'config.json'
checkpoint_bytes, config_bytes = checkpoint_path.read_bytes(), config_path.read_bytes()
rejections = []
for name, mutation in [('incomplete19', lambda x: x.update(rows=x['rows'][:19])),
        ('nonfinite_P', lambda x: x['heads'][0]['weight'].fill_(float('nan'))),
        ('signature_mismatch', lambda x: x.update(signature='bad'))]:
    broken = torch.load(io.BytesIO(checkpoint_bytes), weights_only=True)
    mutation(broken)
    torch.save(broken, checkpoint_path)
    try:
        e.preflight(model_root)
    except ValueError:
        rejections.append(name)
    else:
        raise AssertionError('accepted ' + name)
    finally:
        checkpoint_path.write_bytes(checkpoint_bytes)
broken = json.loads(config_bytes)
broken['weights'][0] = .2
config_path.write_text(json.dumps(broken))
try:
    e.preflight(model_root)
except ValueError:
    rejections.append('config_self_signature')
else:
    raise AssertionError('accepted unsigned altered config')
finally:
    config_path.write_bytes(config_bytes)

bundle = e.preflight(model_root)
feature_calls, hidden = [], []
hook1 = bundle['model'].features.register_forward_hook(lambda module, inputs, output: feature_calls.append(len(inputs[0])))
hook2 = bundle['model'].classifier[-2].register_forward_hook(lambda module, inputs, output: hidden.append(output.detach().clone()))
datasets = [adapter.partitions.IndexedCifar10(images, labels, ids) for ids in hold_ids]
try:
    result = e.export(bundle, OUT / 'quality', datasets, manifest, {}, adapter.source_identities())
finally:
    hook1.remove()
    hook2.remove()
assert feature_calls == [len(ids) for ids in hold_ids] and len(hidden) == 20
assert result['actual_feature_forward_images'] == 30 and result['actual_feature_forward_batches'] == 20
assert result['actual_generic_forward_images'] == result['actual_residual_forward_images'] == 30
assert result['TRAIN_image_forwards'] == result['TRAIN_fits'] == result['new_cnn_backwards'] == result['optimizer_steps'] == 0
clients = result['results'][e.METHOD]['clients']
manual, metric_error = {}, 0.
for i, client in enumerate(clients):
    with np.load(OUT / 'quality' / f'{e.METHOD}_client{i}.npz', allow_pickle=False) as arrays:
        with torch.inference_mode():
            g = F.linear(hidden[i], bundle['model'].classifier[-1].weight, bundle['model'].classifier[-1].bias).numpy()
            local = F.linear(hidden[i], heads[i].weight, heads[i].bias).numpy()
        assert np.array_equal(arrays['generic_raw'], g)
        assert np.array_equal(arrays['personal_residual'], local)
        assert np.array_equal(arrays['generic_plus_local_residual'], g + local)
        assert not np.array_equal(g, g + local)
        assert np.array_equal(arrays['labels'], labels[hold_ids[i]])
        assert np.array_equal(arrays['holdout_ids'], hold_ids[i])
        for variant in ('generic_raw', 'generic_plus_local_residual'):
            z, y = arrays[variant].astype(np.float64), arrays['labels']
            logp = z - np.logaddexp.reduce(z, axis=1)[:, None]
            probability = np.exp(logp)
            target = np.eye(10)[y]
            correct = z.argmax(1) == y
            confidence = probability.max(1)
            gaps = []
            for j in range(15):
                mask = (confidence >= j / 15) & ((confidence < (j + 1) / 15) if j < 14 else (confidence <= 1 + 1e-14))
                gaps.append(float(np.sum(correct[mask]) - np.sum(confidence[mask])))
            values = {'accuracy': float(correct.mean()), 'nll': float(-logp[np.arange(len(y)), y].mean()),
                'brier_sum_classes': float(np.mean(np.sum((probability - target) ** 2, axis=1))),
                'ece_15_equal_width': float(np.sum(np.abs(gaps)) / len(y))}
            metric_error = max(metric_error, *(abs(value - client['metrics'][variant][name]) for name, value in values.items()))
            manual.setdefault(variant, []).append((len(y), values, gaps))
assert metric_error < 1e-12
summary_error = 0.
for variant, values in manual.items():
    summary = result['results'][e.METHOD]['summaries'][variant]
    n = np.asarray([v[0] for v in values])
    for name in values[0][1]:
        x = np.asarray([v[1][name] for v in values])
        summary_error = max(summary_error, abs(float(x.mean()) - summary['macro_' + name]),
            abs(float(np.sum(x * n) / n.sum()) - summary['weighted_' + name]))
    pooled = np.sum([v[2] for v in values], axis=0)
    summary_error = max(summary_error, abs(float(np.sum(np.abs(pooled)) / n.sum()) - summary['pooled_ece_15_equal_width']))
assert summary_error < 1e-12
uniform = e.checked_score(np.zeros((4, 10)), np.array([0, 1, 2, 3]))
assert abs(uniform['nll'] - np.log(10)) < 1e-14 and abs(uniform['brier_sum_classes'] - .9) < 1e-14 and abs(uniform['ece_15_equal_width'] - .15) < 1e-14
try:
    e.checked_score(np.zeros((2, 10)), np.array([0.1, 1.9]))
except ValueError:
    rejections.append('fractional_GT_before_metric_cast')
else:
    raise AssertionError('accepted fractional GT')
assert all(r.sha(path) == value for path, value in protected.items())
assert len(attempts) == 1
report = {'status': 'PASS', 'scope': 'synthetic final checkpoint with nonzero P; no real training/official image inference',
    'fixture_out': str(OUT), 'source_identities': protected, 'source_identities_unchanged': True,
    'nonzero_P_for_all20_clients': True, 'G_P_GplusP_bitwise_equal_to_independent_linear': True,
    'one_feature_forward_per_batch_verified_by_hooks': True,
    'actual_feature_forward_images': 30, 'actual_feature_forward_batches': 20,
    'TRAIN_image_forwards': 0, 'TRAIN_fits': 0, 'new_cnn_backwards': 0,
    'max_handcomputed_metric_error': metric_error, 'max_handcomputed_summary_error': summary_error,
    'rejected': rejections, 'dummy_TEST_path_rejected': True, 'official_files_read': 0,
    'formal_ML_training_started': False, 'harness_sha256': r.sha(__file__), 'review': 'REVIEW_UNAVAILABLE'}
r.atomic_json(r.HERE / 'EXPORT_QUALITY_FIXTURES.json', report)
print(json.dumps({key: report[key] for key in ('status', 'fixture_out', 'actual_feature_forward_images',
    'actual_feature_forward_batches', 'max_handcomputed_metric_error', 'max_handcomputed_summary_error',
    'formal_ML_training_started')}, ensure_ascii=False))
