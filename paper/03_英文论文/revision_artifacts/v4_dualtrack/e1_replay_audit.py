"""Independent saved-array replay; no model inference, fitting, or TEST access.

Run after the full E1 matrix, or use --allow-partial for a dated snapshot.
This script is deliberately outside E1_SOURCE_LOCK: it cannot affect training.
"""
from pathlib import Path
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import struct
import sys

sys.dont_write_bytecode = True
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
SEEDS = {'cifar': (20261231, 20261232, 20261233),
         'fashion': (20261241, 20261242, 20261243)}
ALPHAS = (.1, .5, 1.)
PARTITIONS = {'cifar': 20261217, 'fashion': 20261219}
HASH_CACHE = {}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    path = Path(path)
    st = path.stat()
    key = (str(path), st.st_size, st.st_mtime_ns)
    if key not in HASH_CACHE:
        HASH_CACHE[key] = hashlib.sha256(path.read_bytes()).hexdigest()
    return HASH_CACHE[key]


def hashes(mapping):
    for path, digest in mapping.items():
        require(sha(path) == digest, 'Changed input/output: ' + path)


def metric(scores, labels):
    """Independent logaddexp-reduction calculation, no frozen solver import."""
    require(np.isfinite(scores).all(), 'nonfinite scores')
    loss = np.logaddexp.reduce(scores, axis=1) - scores[np.arange(len(labels)), labels]
    return float(loss.mean()), float(np.mean(scores.argmax(axis=1) == labels))


def payload(path):
    """Independent parsing of the saved first uniform3 payload per run."""
    wire = path.read_bytes()
    magic, version, size, block, blocks, reserved = struct.unpack_from('<8sIIII40s', wire)
    require(magic == b'MPQCODEC' and version == 1 and size == 136874 and
            block == 256 and blocks == (size + 255) // 256 and not any(reserved),
            'Unexpected payload header')
    require(len(wire) == 54067, 'Unexpected actual uniform3 payload bytes')
    offset, decoded = 64, []
    for i in range(blocks):
        width = wire[offset]
        scale = struct.unpack_from('<f', wire, offset + 1)[0]
        require(width == 3 and math.isfinite(scale) and scale >= 0, 'Invalid uniform3 block')
        offset += 5
        count = min(block, size - block * i)
        length = (count * width + 7) // 8
        packed = wire[offset:offset + length]
        require(len(packed) == length, 'Truncated packed block')
        bits = np.unpackbits(np.frombuffer(packed, dtype=np.uint8), bitorder='little')
        require(not bits[count * width:].any(), 'Nonzero padding bits')
        codes = (bits[:count * width].reshape(count, width) * (1 << np.arange(width))).sum(axis=1)
        require(np.all(codes != 7), 'Unused maximum code found')
        require(scale != 0 or np.all(codes == 3), 'Noncanonical zero block')
        decoded.append((codes.astype(np.float32) - 3.) * np.float32(scale))
        offset += length
    values = np.concatenate(decoded)
    require(offset == len(wire) and values.shape == (size,) and np.isfinite(values).all(),
            'Payload decode failure')
    return {'bytes': len(wire), 'numel': int(values.size), 'blocks': blocks,
            'all_block_widths': 3, 'sha256': sha(path)}


def replay(ep):
    r = json.loads(ep.read_text())
    cfg = json.loads((ep.parent / 'fedlc_uniform3/config.json').read_text())
    manifest = cfg['data_manifest']
    monitor = np.asarray(manifest['validation_indices'])
    require(r['status'] == 'COMPLETE' and len(r['clients']) == 20 and r['rounds'] == 20 and
            r['official_test_reads'] == 0 and not r['forbidden_open_attempts'], 'Incomplete evaluation')
    require(cfg['method'] == 'fedlc_uniform3' and cfg['backend'] == 'mps' and cfg['rounds'] == 20 and
            cfg['parameters'] == 136874 and cfg['batch'] == 64 and cfg['lr'] == .03 and
            cfg['local_epochs'] == 1 and cfg['loss'] == 'fedlc' and manifest['alpha'] == r['alpha'] and
            cfg['seed'] == r['seed'] and cfg['partition_seed'] == PARTITIONS[r['task']] and
            manifest['num_clients'] == 20 and manifest['mixing'] == 0, 'Run config differs from plan')
    cfg_unsigned = dict(cfg); signature = cfg_unsigned.pop('signature')
    require(signature == hashlib.sha256(json.dumps(cfg_unsigned, sort_keys=True).encode()).hexdigest(),
            'Config signature mismatch')
    hashes({**r['inputs'], **r['outputs']})
    checkpoint = ep.parent / 'fedlc_uniform3/checkpoint.pt'
    require(sha(checkpoint) == r['checkpoint_sha256'], 'Checkpoint hash mismatch')
    ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
    require(ck['config'] == cfg and ck['signature'] == signature and len(ck['rows']) == 20,
            'Checkpoint does not match final20 config')
    require(len(ck['residuals']) == 20 and all(v.shape == (136874,) and v.dtype == torch.float32 and
            v.device.type == 'cpu' and bool(torch.isfinite(v).all()) for v in ck['residuals']),
            'Invalid error feedback residuals')
    require(all(torch.isfinite(v).all() for v in ck['model'].values()), 'Nonfinite model checkpoint')
    train_result = json.loads((ep.parent / 'fedlc_uniform3/result.json').read_text())
    require(train_result['status'] == 'DONE' and train_result['rows'] == ck['rows'] and
            train_result['signature'] == signature and not train_result['official_test_byte_read'],
            'Training result mismatch')
    max_ef = 0.
    for rnd, row in enumerate(train_result['rows']):
        require(row['round'] == rnd + 1 and len(row['clients']) == 20 and
                row['global_anchor_unchanged_between_clients'] and row['upload_bytes'] == 20 * 54067,
                'Invalid training round accounting')
        for client, c in enumerate(row['clients']):
            require(c['client'] == client and c['seed'] == r['seed'] + 100 * rnd + client and
                    c['upload_bytes'] == 54067 and c['upload_accounting'] == 'actual decoded wire' and
                    c['train_counts'] == manifest['client_label_counts'][client], 'Invalid client training record')
            require(c['steps'] == c['gradient_evaluations'] == math.ceil(manifest['client_sizes'][client] / 64),
                    'Unexpected local epoch/batch count')
            expected_shift = -np.maximum(np.asarray(c['train_counts']), 1.) ** (-.25)
            require(np.allclose(c['logit_shift'], expected_shift, rtol=1e-6, atol=1e-7), 'FedLC shift differs')
            max_ef = max(max_ef, c['ef_max_error'])
    require(max_ef < 1e-6, 'Excessive EF identity error')
    wire = payload(ep.parent / 'fedlc_uniform3/first_payload.bin')
    full_train = np.concatenate(manifest['train_indices'])
    full_holdout = np.concatenate(manifest['holdout_indices'])
    require(np.array_equal(np.sort(np.r_[full_train, full_holdout]),
                           np.arange(50000 if r['task'] == 'cifar' else 60000)), 'Membership coverage failure')
    require(np.intersect1d(full_train, full_holdout).size == 0 and len(monitor) == 2000 and
            len(np.unique(monitor)) == 2000 and np.isin(monitor, full_holdout).all(), 'Invalid monitor/holdout split')
    errors = {'metric_max_abs': 0., 'train_objective_max_abs': 0., 'gradient_norm_max_abs': 0.}
    deltas, sample_counts, stop_reasons, gradients = [], [], Counter(), []
    score_metrics = {name: {'loss': [], 'accuracy': []} for name in ['raw', 'zero_anchor_vector', 'prioranchored_vector']}
    for i, c in enumerate(r['clients']):
        require(c['client'] == i, 'Client ordering mismatch')
        with np.load(ep.parent / f'client_arrays/client{i}.npz') as a:
            z = a['holdout_logits'].astype(np.float64); y = a['holdout_labels']
            tz = a['train_logits'].astype(np.float64); ty = a['train_labels']
            require(z.shape == (len(y), 10) and tz.shape == (len(ty), 10) and
                    np.isfinite(tz).all(), 'Invalid saved logits')
            require(np.array_equal(a['train_ids'], manifest['train_indices'][i]) and
                    np.array_equal(a['holdout_ids'], np.setdiff1d(manifest['holdout_indices'][i], monitor)) and
                    np.array_equal(a['original_holdout_ids'], manifest['holdout_indices'][i]), 'Saved row IDs mismatch')
            require(np.intersect1d(a['train_ids'], a['holdout_ids']).size == 0 and
                    np.intersect1d(a['holdout_ids'], monitor).size == 0, 'Scoring/fitting overlap')
            counts = np.bincount(ty, minlength=10)
            require(counts.tolist() == c['train_counts'] == manifest['client_label_counts'][i] and
                    np.array_equal(counts, a['train_counts']), 'Saved TRAIN counts mismatch')
            anchor = np.log(counts + 1.); anchor -= anchor.mean()
            require(np.allclose(anchor, a['prior_anchor'], rtol=0, atol=1e-14), 'Prior anchor differs from TRAIN counts')
            losses = {}
            for variant in score_metrics:
                if variant == 'raw':
                    scores = z
                else:
                    state = a[variant]
                    require(state.dtype == np.float32 and state.shape == (20,) and state.nbytes == 80 and
                            np.isfinite(state).all(), 'Invalid 80B deployment state')
                    scores = z * state[:10].astype(np.float64) + state[10:].astype(np.float64)
                nll, accuracy = metric(scores, y); losses[variant] = nll
                errors['metric_max_abs'] = max(errors['metric_max_abs'],
                    abs(nll - c['metrics'][variant]['nll']), abs(accuracy - c['metrics'][variant]['accuracy']))
                score_metrics[variant]['loss'].append(nll); score_metrics[variant]['accuracy'].append(accuracy)
            deltas.append(losses['prioranchored_vector'] - losses['zero_anchor_vector'])
            sample_counts.append(len(y))
            require(c['holdout_samples'] == c['evaluation_holdout_count'] == len(y) and
                    c['train_samples'] == len(ty), 'Saved sample counts differ')
            for name, cent in [('zero_fit', np.zeros(10)), ('prior_fit', anchor)]:
                fit = c[name]; theta = np.array(fit['parameters_float64']); center = np.r_[np.ones(10), cent]
                scores = tz * theta[:10] + theta[10:]; logsum = np.logaddexp.reduce(scores, axis=1)
                objective = float((logsum - scores[np.arange(len(ty)), ty]).sum() + 5. * np.dot(theta - center, theta - center))
                e = np.exp(scores - logsum[:, None]); e[np.arange(len(ty)), ty] -= 1.
                gradient = np.r_[(e * tz).sum(0), e.sum(0)] + 10. * (theta - center)
                errors['train_objective_max_abs'] = max(errors['train_objective_max_abs'], abs(objective - fit['final_objective']))
                errors['gradient_norm_max_abs'] = max(errors['gradient_norm_max_abs'],
                    abs(np.max(np.abs(gradient)) - fit['projected_gradient_inf']))
                require(fit['regularization_scale'] == fit['regularization_bias'] == 10 and
                        fit['stop_reason'] != 'maximum_steps' and fit['projected_gradient_inf'] < 1e-5 and
                        np.all(np.diff([s['objective'] for s in fit['trace']]) <= 1e-10), 'Unresolved vector fit')
                stop_reasons[fit['stop_reason']] += 1; gradients.append(fit['projected_gradient_inf'])
    require(errors['metric_max_abs'] < 1e-12 and errors['train_objective_max_abs'] < 1e-7 and
            errors['gradient_norm_max_abs'] < 1e-7, 'Independent numerical replay mismatch')
    weights = np.asarray(sample_counts) / sum(sample_counts)
    for variant, metrics in score_metrics.items():
        for metric_name, values in metrics.items():
            require(abs(np.mean(values) - r['summaries'][variant]['macro_' + metric_name]) < 1e-12 and
                    abs(np.dot(values, weights) - r['summaries'][variant]['weighted_' + metric_name]) < 1e-12,
                    'Aggregate metric mismatch')
    require(abs(np.mean(deltas) - r['prior_minus_zero_macro_nll']) < 1e-12 and
            abs(np.mean(deltas) + r['zero_minus_prior_macro_nll_gain']) < 1e-12, 'Gain sign or macro weighting mismatch')
    return {'path': str(ep), 'task': r['task'], 'alpha': r['alpha'], 'seed': r['seed'],
            'rounds': 20, 'clients': 20, 'evaluated_predictor_states': 40,
            'scoring_rows': sum(sample_counts), 'macro_gain': -float(np.mean(deltas)),
            'errors': errors, 'max_fit_gradient_inf': max(gradients), 'solver_stop_reasons': dict(stop_reasons),
            'max_ef_identity_error': max_ef, 'first_payload': wire, 'evaluation_sha256': sha(ep)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--allow-partial', action='store_true')
    args = parser.parse_args()
    attempted_test_reads = []
    def guard(event, values):
        if event == 'open' and isinstance(values[0], (str, bytes)) and any(s in str(values[0]) for s in ('test_batch', 't10k')):
            attempted_test_reads.append(str(values[0])); raise ValueError('Official TEST access forbidden in replay')
    sys.addaudithook(guard)
    torch.set_num_threads(1)
    lock = json.loads((HERE / 'E1_SOURCE_LOCK.json').read_text()); hashes(lock['inputs'])
    protected = json.loads((HERE / 'protected_files.json').read_text())['sha256']; hashes(protected)
    results, missing = [], []
    for task, seeds in SEEDS.items():
        for alpha in ALPHAS:
            for seed in seeds:
                ep = HERE / 'e1_runs' / task / ('alpha_' + str(alpha)) / ('s' + str(seed)) / 'evaluation.json'
                if not ep.exists():
                    missing.append(str(ep)); continue
                results.append(replay(ep))
    require(args.allow_partial or not missing, 'Full audit requires all 18 evaluations; use --allow-partial for a snapshot')
    aggregate_check = 'NOT_AVAILABLE_IN_PARTIAL_SNAPSHOT'
    aggregate_path = HERE / 'e1_alpha_sweep.json'
    if not missing:
        aggregate = json.loads(aggregate_path.read_text()); require(aggregate['status'] == 'COMPLETE', 'Incomplete aggregate')
        hashes(aggregate['inputs'])
        for task, data in aggregate['tasks'].items():
            for row in data['rows']:
                gains = [next(r['macro_gain'] for r in results if r['task'] == task and r['alpha'] == row['alpha'] and r['seed'] == seed) for seed in SEEDS[task]]
                stored = row['zero_minus_prior_macro_nll_gain']
                require(np.allclose(gains, stored['individual_model_seed_values'], rtol=0, atol=1e-12) and
                        abs(np.mean(gains) - stored['mean']) < 1e-12 and
                        abs(np.std(gains, ddof=1) - stored['sample_sd']) < 1e-12, 'Sweep mean/SD differs')
        require(sha(aggregate['figure']) == aggregate['figure_sha256'], 'Figure identity mismatch')
        aggregate_check = 'PASS'
    require(not attempted_test_reads, 'TEST access attempted')
    output = {'status': 'PASS' if not missing else 'PASS_AVAILABLE_RUNS',
              'created_at': datetime.now(timezone.utc).isoformat(), 'audited_completed_runs': len(results),
              'expected_runs': 18, 'runs': results, 'missing': missing,
              'checks': {'source_lock_intact': True, 'protected_file_count': len(protected),
                         'protected_files_unchanged': True, 'aggregate_replay': aggregate_check,
                         'no_new_fits_or_inference': True, 'official_test_image_bytes_opened_by_audit': 0},
              'interpretation_limits': [
                  'One partition realization per alpha, three model seeds; no universal or causal heterogeneity law.',
                  'Frozen fitting uses SUM(NLL) + (10/2)||state-anchor||^2. Normalized effective ridge is 10/n; sample size changes relative anchor influence. This may explain part of E2 sample-size association but remains untested.',
                  'Replay verifies saved logits/states/metrics and memberships, not independent regeneration of training or logits.',
                  'Each round logs actual decoded payload accounting; only the saved first payload is independently decoded here.']}
    target = HERE / 'e1_independent_audit.json'; temporary = target.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(output, indent=2, ensure_ascii=False, allow_nan=False) + '\n'); temporary.replace(target)
    print(json.dumps({'status': output['status'], 'audited_runs': len(results), 'missing': len(missing),
                      'max_metric_error': max((r['errors']['metric_max_abs'] for r in results), default=0.),
                      'audit_path': str(target)}), flush=True)


if __name__ == '__main__':
    main()
