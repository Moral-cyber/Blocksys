"""CPU synthetic checks only: no raw datasets, no downloads, no formal training."""
from __future__ import annotations

import contextlib
import copy
import io
import json
from pathlib import Path
import statistics
import sys
import tempfile
import time

sys.dont_write_bytecode = True
RAW_OPEN_ATTEMPTS = []


def audit(event, arguments):
    if event == 'open' and isinstance(arguments[0], (str, bytes)):
        path = str(arguments[0])
        if '/rawdata/' in path or 'data_batch_' in path or 'test_batch' in path:
            RAW_OPEN_ATTEMPTS.append(path)
            raise RuntimeError('raw CIFAR data forbidden in local fixtures')


sys.addaudithook(audit)
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import reference as r
import train as entry

CHECKS = []


def check(condition, label):
    if not bool(condition):
        raise AssertionError(label)
    CHECKS.append(label)


def rejects(function, label):
    try:
        function()
    except (ValueError, RuntimeError):
        CHECKS.append(label)
        return
    raise AssertionError('did not reject: ' + label)


def all_grads_none(model):
    return all(parameter.grad is None for parameter in model.parameters())


def gradients(model):
    return [None if parameter.grad is None else parameter.grad.detach().clone()
            for parameter in model.parameters()]


def one_step(model, head, x, y, counts, joint):
    model.zero_grad(set_to_none=True)
    head.zero_grad(set_to_none=True)
    brm, erm = r.losses(model, head, x, y, counts)
    optimizer = torch.optim.SGD(list(model.parameters()) + list(head.parameters()), lr=.03)
    (brm + erm if joint else brm).backward()
    optimizer.step()


def main():
    torch.set_num_threads(1)
    frozen = r.verify_frozen_sources()
    local_sources_before = r.source_identities()
    rng = torch.Generator().manual_seed(9182)
    x = torch.randn(7, 3, 32, 32, generator=rng)
    y = torch.tensor([0, 1, 3, 1, 0, 3, 3], dtype=torch.long)
    counts = torch.bincount(y, minlength=10)
    model = r.new_model(entry.SEED)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(entry.SEED)
        original_class_model = r.SmallCNN()
    check(torch.equal(r.vectorize(model), r.vectorize(original_class_model)),
          'global initialization exactly matches frozen SmallCNN seed')
    check(r.vectorize(model).numel() == 136874, 'shared vector has 136874 coordinates')
    before_rng = torch.get_rng_state().clone()
    zero_head = r.new_head()
    check(torch.equal(before_rng, torch.get_rng_state()), 'personal-head initialization preserves RNG state')
    check(sum(p.numel() for p in zero_head.parameters()) == 650, 'residual Linear has 650 parameters')
    check(all(torch.count_nonzero(p) == 0 for p in zero_head.parameters()), 'residual initialization is exactly zero')
    g, p = r.branch_logits(model, zero_head, x)
    check(torch.equal(g, model(x)), 'split frozen feature/generic matches original forward exactly')
    check(torch.equal(g, p), 'zero residual makes personal logits equal generic logits')

    head = r.new_head()
    with torch.no_grad():
        head.weight.copy_(torch.randn(head.weight.shape, generator=rng) * .02)
        head.bias.copy_(torch.linspace(-.1, .1, 10))
    erm_model, erm_head = copy.deepcopy(model), copy.deepcopy(head)
    _, erm = r.losses(erm_model, erm_head, x, y, counts)
    erm.backward()
    check(all_grads_none(erm_model), 'ERM gives no gradient to shared body or generic head')
    check(all(p.grad is not None and torch.isfinite(p.grad).all() for p in erm_head.parameters()),
          'ERM gives finite gradients to personal head')
    brm_model, brm_head = copy.deepcopy(model), copy.deepcopy(head)
    brm, _ = r.losses(brm_model, brm_head, x, y, counts)
    brm.backward()
    check(all_grads_none(brm_head), 'BRM gives no gradient to personal head')
    check(all(p.grad is not None and torch.isfinite(p.grad).all() for p in brm_model.parameters()),
          'BRM gives finite gradients to shared body and generic head')
    joint_model, joint_head = copy.deepcopy(model), copy.deepcopy(head)
    brm, erm = r.losses(joint_model, joint_head, x, y, counts)
    (brm + erm).backward()
    shared_gradient_error = max(float((a - b).abs().max())
        for a, b in zip(gradients(joint_model), gradients(brm_model)))
    head_gradient_error = max(float((a - b).abs().max())
        for a, b in zip(gradients(joint_head), gradients(erm_head)))
    check(shared_gradient_error == 0., 'joint shared gradient equals independent BRM exactly')
    check(head_gradient_error == 0., 'joint personal gradient equals independent ERM exactly')
    gm, gp = copy.deepcopy(model), copy.deepcopy(head)
    jm, jp = copy.deepcopy(model), copy.deepcopy(head)
    one_step(gm, gp, x, y, counts, False)
    one_step(jm, jp, x, y, counts, True)
    check(torch.equal(r.vectorize(gm), r.vectorize(jm)), 'joint generic SGD update equals BRM-only update')
    check(not torch.equal(r.vectorize(jp), r.vectorize(head)), 'personal head actually changes after SGD')
    check(torch.equal(r.vectorize(gp), r.vectorize(head)), 'BRM-only SGD leaves personal head unchanged')

    with torch.no_grad():
        g, personal = r.branch_logits(model, head, x)
        residual = head(r.features(model, x))
    check(torch.equal(personal, g + residual), 'personal inference is exact G+P logit sum')
    check(not torch.equal(personal, residual), 'personal prediction is not residual alone')
    zero_counts = torch.zeros(10, dtype=torch.long)
    rejects(lambda: r.balanced_softmax(g, y, zero_counts), 'all-zero counts rejected')
    missing_y = y.clone()
    missing_y[0] = 9
    rejects(lambda: r.balanced_softmax(g, missing_y, counts), 'target from missing TRAIN class rejected')
    missing_model = copy.deepcopy(model)
    loss = r.balanced_softmax(missing_model(x), y, counts)
    loss.backward()
    check(torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in missing_model.parameters()),
          'missing-class unsmoothed BSM has finite loss and gradients')
    single_counts = torch.zeros(10, dtype=torch.long)
    single_counts[3] = 7
    single_model = copy.deepcopy(model)
    single = r.balanced_softmax(single_model(x), torch.full_like(y, 3), single_counts)
    single.backward()
    check(float(single.detach()) == 0., 'single-class BSM loss is finite and exactly zero')
    check(all(torch.count_nonzero(p.grad) == 0 for p in single_model.parameters()),
          'single-class BSM gradients are finite zero')
    reference_loss = nn.functional.cross_entropy(g + counts.float().log(), y)
    check(torch.equal(r.balanced_softmax(g, y, counts), reference_loss), 'BSM exactly uses raw log counts')
    check(not torch.equal(reference_loss, nn.functional.cross_entropy(g + (counts.float() + 1).log(), y)),
          'raw BSM explicitly differs from add-one smoothed existing baseline')

    dataset = TensorDataset(x, y)
    anchor = r.vectorize(model).clone()
    observed_forwards = {'feature': 0, 'generic': 0, 'personal': 0}
    def count_forward(name):
        def hook(module, arguments, result):
            observed_forwards[name] += 1
        return hook
    handles = [model.features.register_forward_hook(count_forward('feature')),
        model.classifier[-1].register_forward_hook(count_forward('generic')),
        zero_head.register_forward_hook(count_forward('personal'))]
    delta1, state1, counters1 = r.train_local(model, zero_head, dataset, counts.tolist(), 111, 'cpu')
    for handle in handles:
        handle.remove()
    check(observed_forwards == {'feature': 1, 'generic': 1, 'personal': 1},
          'actual train_local module hooks observe exactly one feature/G/P forward')
    ordered_x, ordered_y = next(iter(DataLoader(dataset, batch_size=64, shuffle=True,
        generator=torch.Generator().manual_seed(111), num_workers=0)))
    manual_generic, manual_personal = copy.deepcopy(model), copy.deepcopy(zero_head)
    one_step(manual_generic, manual_personal, ordered_x, ordered_y, counts, False)
    check(torch.equal(delta1, r.vectorize(manual_generic) - anchor),
          'actual train_local generic update exactly equals same shuffled BRM-only minibatch')
    manual_head = r.new_head()
    manual_head_optimizer = torch.optim.SGD(manual_head.parameters(), lr=.03)
    with torch.no_grad():
        ordered_feature = r.features(model, ordered_x)
        ordered_generic = model.classifier[-1](ordered_feature)
    nn.functional.cross_entropy(ordered_generic + manual_head(ordered_feature), ordered_y).backward()
    manual_head_optimizer.step()
    check(all(torch.equal(state1[k], manual_head.state_dict()[k]) for k in state1),
          'actual train_local personal update equals independent fixed-feature fixed-generic ERM')
    check(torch.equal(anchor, r.vectorize(model)), 'local training never changes global anchor')
    check(torch.count_nonzero(delta1) > 0, 'synthetic local shared update is nonzero')
    head1 = r.new_head()
    head1.load_state_dict(state1)
    check(torch.count_nonzero(r.vectorize(head1)) > 0, 'local returned persistent head is nonzero')
    check(counters1['combined_backward_calls'] == 1 and counters1['sgd_steps'] == 1,
          'one synthetic minibatch uses one combined backward and one SGD step')
    for field in ('feature_forward_batches', 'generic_forward_batches', 'residual_forward_batches',
                  'body_gradient_updates', 'generic_gradient_updates', 'residual_gradient_updates'):
        check(counters1[field] == 1, 'cost counter ' + field)
    check(counters1['residual_gradient_scalars'] == 650, 'cost counts all 650 personal gradient scalars')
    check(counters1['independent_residual_backward_seconds'] is None,
          'exclusive head-backward time is not falsely attributed')
    q1, ef1, wire1, encoded1 = r.encode_shared(delta1, torch.zeros_like(delta1))
    check(len(wire1) == 54067 <= r.CAP, 'actual uniform3 shared wire is 54067B within 54808B cap')
    check(q1.shape == (136874,) and encoded1['personal_parameters_uploaded'] == 0,
          'actual wire/EF contains shared parameters only')
    check(torch.equal(q1, r.load_codec().decode(wire1)), 'aggregate candidate comes from real wire decoder')
    check(torch.equal(ef1, delta1 - q1), 'first-round EF is exact candidate minus decoded upload')
    rejects(lambda: r.encode_shared(torch.cat((delta1, r.vectorize(head1))), torch.zeros(137524)),
            'appending personal head to shared wire is rejected by fixed layout')

    resumed_global = copy.deepcopy(model)
    r.assign_vector(resumed_global, anchor + q1)
    delta2, state2, counters2 = r.train_local(resumed_global, head1, dataset, counts.tolist(), 112, 'cpu')
    ordered_x2, ordered_y2 = next(iter(DataLoader(dataset, batch_size=64, shuffle=True,
        generator=torch.Generator().manual_seed(112), num_workers=0)))
    manual_generic2, manual_head2 = copy.deepcopy(resumed_global), copy.deepcopy(head1)
    one_step(manual_generic2, manual_head2, ordered_x2, ordered_y2, counts, False)
    check(torch.equal(delta2, r.vectorize(manual_generic2) - r.vectorize(resumed_global)),
          'actual train_local with nonzero persistent P still equals BRM-only shared update')
    _, reset_state2, _ = r.train_local(resumed_global, r.new_head(), dataset, counts.tolist(), 112, 'cpu')
    check(any(not torch.equal(state2[k], reset_state2[k]) for k in state2),
          'second round preserves local head instead of resetting')
    q2, ef2, wire2, encoded2 = r.encode_shared(delta2, ef1)
    check(torch.equal(ef2, delta2 + ef1 - q2), 'second-round EF includes preceding local EF exactly once')
    decoded2 = r.load_codec().decode(wire2)
    aggregate = torch.zeros_like(anchor).add_(q1, alpha=.25).add_(q2, alpha=.75)
    expected = torch.zeros_like(anchor).add_(r.load_codec().decode(wire1), alpha=.25).add_(decoded2, alpha=.75)
    check(torch.equal(aggregate, expected), 'sample-weighted aggregation equals independently decoded wires')
    check(not torch.equal(aggregate, .5 * q1 + .5 * q2), 'nonuniform weighting fixture distinguishes equal-weight bug')

    base = entry.base_config('cpu')
    config = {'base': base, 'weights': [1 / 20] * 20,
              'data_manifest': {'synthetic': True}, 'data_identities': {}, 'train_adapter_identities': {}}
    config['signature'] = r.signature(config)
    r.validate_config(config, base)
    heads = [r.new_head() for _ in range(20)]
    heads[0].load_state_dict(state1)
    residuals = [torch.zeros(r.PARAMETERS) for _ in range(20)]
    residuals[0] = ef1.clone()
    rows = [{'round': 1, 'synthetic': True}]
    with tempfile.TemporaryDirectory(prefix='fedrod_fixture_', dir=r.HERE) as directory:
        out = Path(directory)
        r.atomic_json(out / 'config.json', config)
        io_cost = r.save_checkpoint(out / 'checkpoint.pt', config, resumed_global, heads, residuals, rows)
        loaded = torch.load(out / 'checkpoint.pt', map_location='cpu', weights_only=False)
        loaded_model = r.validate_checkpoint(loaded, config)
        check(torch.equal(r.vectorize(loaded_model), r.vectorize(resumed_global)), 'checkpoint restores exact shared state')
        loaded_head = r.new_head()
        loaded_head.load_state_dict(loaded['heads'][0])
        check(torch.equal(r.vectorize(loaded_head), r.vectorize(head1)), 'checkpoint restores exact persistent personal head')
        check(torch.equal(loaded['residuals'][0], ef1), 'checkpoint restores exact local EF')
        delta_reload, state_reload, _ = r.train_local(loaded_model, loaded_head, dataset, counts.tolist(), 112, 'cpu')
        check(torch.equal(delta_reload, delta2) and all(torch.equal(state_reload[k], state2[k]) for k in state2),
              'save/reload followed by next local step equals uninterrupted continuation')
        _, preflight_ck = entry.preflight(out, base)
        check(preflight_ck is not None, 'resume preflight validates checkpoint without importing data adapter')
        changed_base = copy.deepcopy(base)
        changed_base['source_identities']['synthetic_changed_source.py'] = '0' * 64
        rejects(lambda: entry.preflight(out, changed_base), 'changed source config rejected before data loading')
        changed_layout = copy.deepcopy(base)
        changed_layout['personal_layout']['weight']['shape'] = [10, 63]
        rejects(lambda: entry.preflight(out, changed_layout), 'changed personal layout rejected before data loading')
        bad = copy.deepcopy(loaded)
        bad['heads'][0]['weight'] = torch.zeros(10, 63)
        rejects(lambda: r.validate_checkpoint(bad, config), 'wrong head shape rejected')
        bad = copy.deepcopy(loaded)
        bad['heads'] = bad['heads'][:-1]
        rejects(lambda: r.validate_checkpoint(bad, config), 'wrong persistent head count rejected')
        bad = copy.deepcopy(loaded)
        bad['residuals'][0] = bad['residuals'][0].double()
        rejects(lambda: r.validate_checkpoint(bad, config), 'wrong EF dtype rejected')
        bad = copy.deepcopy(loaded)
        bad['model'][next(iter(bad['model']))].view(-1)[0] = float('nan')
        rejects(lambda: r.validate_checkpoint(bad, config), 'nonfinite shared state rejected')
        bad = copy.deepcopy(loaded)
        bad['heads'][0]['bias'][0] = float('nan')
        rejects(lambda: r.validate_checkpoint(bad, config), 'nonfinite personal state rejected')
        bad = copy.deepcopy(loaded)
        bad['signature'] = 'wrong'
        rejects(lambda: r.validate_checkpoint(bad, config), 'wrong checkpoint signature rejected')
        bad = copy.deepcopy(loaded)
        bad['rows'] = [{'round': 2}]
        rejects(lambda: r.validate_checkpoint(bad, config), 'noncontiguous round history rejected')
        bad = copy.deepcopy(loaded)
        bad['rows'] = ['not a round record']
        rejects(lambda: r.validate_checkpoint(bad, config), 'malformed round history rejected')
        bad = copy.deepcopy(loaded)
        bad['heads'][0]['weight'] = torch.zeros(10, 63)
        torch.save(bad, out / 'checkpoint.pt')
        rejects(lambda: entry.preflight(out, base),
                'actual saved incompatible head rejected by preflight before data-loader import')
        torch.save(loaded, out / 'checkpoint.pt')
        # Exercise actual DESIGN_ONLY main path, no subprocess or data-loader import.
        argv = sys.argv
        output = io.StringIO()
        try:
            sys.argv = ['train.py', '--backend', 'cpu', '--out', str(out)]
            with contextlib.redirect_stdout(output):
                entry.main()
        finally:
            sys.argv = argv
        default_output = json.loads(output.getvalue())
        check(default_output['status'] == 'DESIGN_ONLY' and not default_output['formal_training_executed'],
              'default training entry is DESIGN_ONLY')
    check('_fedrod_linear_train_only_adapter' not in sys.modules,
          'toy and default entry never imported strict TRAIN loader')
    check('torchvision' not in sys.modules, 'toy never imports torchvision dataset stack')
    eval_metrics, eval_cost = r.evaluate(resumed_global, head1, dataset, 'cpu')
    with torch.no_grad():
        eg, ep = r.branch_logits(resumed_global, head1, x)
    for name, logits in (('generic_raw', eg), ('generic_plus_local_residual', ep)):
        check(eval_metrics[name]['accuracy'] == float((logits.argmax(1) == y).double().mean()),
              'evaluation exact ' + name + ' accuracy')
        check(abs(eval_metrics[name]['ce'] - float(nn.functional.cross_entropy(logits, y))) < 3e-7,
              'evaluation ' + name + ' CE matches direct logits')
    check(eval_cost['feature_forward_batches'] == 1 and eval_cost['residual_forward_batches'] == 1,
          'holdout evaluation shares one feature pass for G and G+P')

    # Same synthetic batch CPU timings only; interleave arms to reduce ordering effect.
    times = {'brm_only': [], 'joint': []}
    for iteration in range(14):
        order = (False, True) if iteration % 2 == 0 else (True, False)
        for joint in order:
            tm, th = copy.deepcopy(model), copy.deepcopy(head)
            start = time.perf_counter()
            one_step(tm, th, x, y, counts, joint)
            elapsed = time.perf_counter() - start
            if iteration >= 2:
                times['joint' if joint else 'brm_only'].append(elapsed)
    timing = {name: {'repetitions': len(values), 'median_seconds': statistics.median(values),
                     'min_seconds': min(values), 'max_seconds': max(values)}
              for name, values in times.items()}
    timing['median_joint_minus_brm_seconds'] = timing['joint']['median_seconds'] - timing['brm_only']['median_seconds']
    timing['scope'] = 'CPU synthetic 7x3x32x32; 2 warmups then 12 interleaved samples; no GPU/real-data inference'
    check(r.verify_frozen_sources() == frozen, 'all 13 frozen source SHA256 values unchanged')
    check(r.source_identities() == local_sources_before, 'new core/entry/plan sources unchanged during fixtures')
    check(len(RAW_OPEN_ATTEMPTS) == 0, 'raw CIFAR open attempts are zero')
    result = {'status': 'PASS', 'checks_passed': len(CHECKS), 'checks': CHECKS,
        'shared_gradient_max_error': shared_gradient_error, 'personal_gradient_max_error': head_gradient_error,
        'actual_wire_bytes': [len(wire1), len(wire2)], 'shared_coordinates': r.PARAMETERS,
        'personal_head_parameters': 650, 'personal_head_fp32_bytes_per_client': 2600,
        'local_train_cost_first_synthetic_batch': counters1, 'synthetic_eval_cost': eval_cost,
        'synthetic_checkpoint_cost': io_cost, 'cpu_toy_timing': timing,
        'source_files_unchanged': len(frozen), 'source_identities': local_sources_before,
        'raw_data_open_attempts': 0, 'raw_data_opened': False, 'formal_training_executed': False,
        'real_performance_verified': False, 'review': 'REVIEW_UNAVAILABLE / LOCAL_CHECK_ONLY'}
    r.atomic_json(r.HERE / 'LOCAL_FIXTURES.json', result)
    summary = f'''# FedRoD-linear CPU synthetic 检验：{len(CHECKS)} PASS

本轮未读原始数据、未联网、未启动正式训练；默认入口 DESIGN_ONLY，数据loader/torchvision均未导入，CIFAR打开尝试0。原13个冻结源SHA256与新增入口/核心/方案在检查前后均一致。

共享136874参数，额外每client Linear(64,10)为650参数/2600B。ERM不能给body/generic梯度；BRM不能给P梯度；联合梯度与独立分支最大误差均为0，通用SGD更新逐位等于同batch BRM-only。P实际更新、跨轮保留，保存/恢复后下一步与不中断训练逐位相同。推理严格为G+P logits。原始BSM缺失类有限、单类loss/梯度为0，全零计数/不可能target拒绝。

两轮synthetic共享delta真实编码均54067B（cap54808B），P不进wire/EF；实际decode用于非等权聚合，EF精确更新。恢复的source/layout/signature/head/EF/nonfinite/history异常均拒绝。一次合成batch各共享/个人forward1次，联合backward/SGD1次，个人梯度650标量；holdout两输出共享一次feature pass。

CPU toy：7张合成图，2预热＋12交错重复。BRM-only中位{timing['brm_only']['median_seconds']:.6f}s，联合{timing['joint']['median_seconds']:.6f}s，差{timing['median_joint_minus_brm_seconds']:.6f}s。只说明此toy的墙钟，不能外推GPU、真实吞吐或准确率；独占head backward时间保持null。2600B不含activation/EF/metadata。

原文终点global/local表述差异、零初始化选择、非官方超参复现及代价取舍见PLAN.md。尚无20轮结果，不能据此主张精度或成本优胜。REVIEW_UNAVAILABLE / LOCAL_CHECK_ONLY。
'''
    (r.HERE / 'LOCAL_FIXTURES.md').write_text(summary)
    print(json.dumps({'status': 'PASS', 'checks_passed': len(CHECKS),
        'wire_bytes': [len(wire1), len(wire2)], 'raw_data_open_attempts': 0,
        'formal_training_executed': False, 'cpu_toy_timing': timing}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
