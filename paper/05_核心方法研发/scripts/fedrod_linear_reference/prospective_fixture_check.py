"""CPU-only prospective-wrapper fixtures, no raw input or formal training."""
from __future__ import annotations

import contextlib
import copy
import io
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile

sys.dont_write_bytecode = True
RAW_OPEN_ATTEMPTS = []


def audit(event, arguments):
    if event == 'open' and isinstance(arguments[0], (str, bytes)):
        path = str(arguments[0])
        if '/rawdata/' in path or 'data_batch_' in path or 'test_batch' in path:
            RAW_OPEN_ATTEMPTS.append(path)
            raise RuntimeError('raw CIFAR data forbidden in prospective fixtures')


sys.addaudithook(audit)
import torch
from torch.utils.data import DataLoader, TensorDataset
import prospective as p

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


class SyntheticDispatchComplete(Exception):
    pass


def main():
    torch.set_num_threads(1)
    p.verify_sources()
    before = {name: p.sha(p.HERE / name) for name in p.FROZEN_SOURCES}
    wrapper_sha = p.sha(p.__file__)
    fixture_sha = p.sha(__file__)
    entry = p.load_frozen_entry()
    r = entry.r
    frozen_before = r.verify_frozen_sources()
    old_globals = {name: getattr(entry, name) for name in ('SEED', 'PARTITION_SEED', 'ROUNDS')}
    check(old_globals == {'SEED': 20261211, 'PARTITION_SEED': 20261203, 'ROUNDS': 20},
          'private imported entry retains frozen defaults before override')
    check(p.sha(p.ROOT / p.PLAN_RELATIVE) == p.PLAN_SHA256, 'new prospective PLAN hash matches predeclared value')
    rejects(lambda: p.overrides(20261211, p.PARTITION_SEED), 'old training seed rejected')
    rejects(lambda: p.overrides(20261231, 20261203), 'old partition rejected')
    rejects(lambda: p.base_config(entry, 'cpu', 20261231, p.PARTITION_SEED),
            'base cannot silently use old module globals outside override scope')
    rng = torch.Generator().manual_seed(138)
    x = torch.randn(7, 3, 32, 32, generator=rng)
    x[:, 0, 0, 0] = torch.arange(7, dtype=torch.float32)
    y = torch.tensor([0, 1, 3, 1, 0, 3, 3], dtype=torch.long)
    counts = torch.bincount(y, minlength=10).tolist()
    dataset = TensorDataset(x, y)
    records = []
    with tempfile.TemporaryDirectory(prefix='prospective_fixture_', dir=p.HERE) as temporary:
        root = Path(temporary)
        for seed in p.TRAINING_SEEDS:
            with p.module_overrides(entry, seed, p.PARTITION_SEED) as chosen:
                base = p.base_config(entry, 'cpu', seed, p.PARTITION_SEED)
                check((base['seed'], base['partition_seed'], base['rounds']) == (seed, 20261217, 20),
                      f'{seed}: actual base uses new seed, partition and fixed rounds')
                check(base['prospective']['module_global_overrides'] == chosen
                      and base['runtime']['prospective_entry']['module_global_overrides'] == chosen,
                      f'{seed}: overrides recorded in config and runtime')
                wrapper_relative = str(Path(p.__file__).relative_to(p.ROOT))
                check(base['source_identities'][wrapper_relative] == wrapper_sha,
                      f'{seed}: wrapper hash bound in sources')
                check(base['source_identities'][p.PLAN_RELATIVE] == p.PLAN_SHA256,
                      f'{seed}: new PLAN hash bound in sources')
                check(base['upload_cap_bytes'] == 54808 and base['parameters'] == 136874,
                      f'{seed}: shared codec cap/layout unchanged')
                check(base['personal_loss'] == 'CE(g.detach()+P(feature.detach()),y)',
                      f'{seed}: original detached gradient rule unchanged')
                model = r.new_model(seed)
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(seed)
                    expected_model = r.SmallCNN()
                check(torch.equal(r.vectorize(model), r.vectorize(expected_model)),
                      f'{seed}: initialization exactly matches direct frozen SmallCNN with new seed')
                check(not torch.equal(r.vectorize(model), r.vectorize(r.new_model(20261211))),
                      f'{seed}: initialization differs from old training seed')
                local_seed = seed + 100 * 2 + 7
                expected_loader = DataLoader(dataset, batch_size=64, shuffle=True,
                    generator=torch.Generator().manual_seed(local_seed), num_workers=0)
                expected_order = next(iter(expected_loader))[0][:, 0, 0, 0].tolist()
                observed = {'generator_seed': None, 'order': None}
                original_loader = r.DataLoader
                class RecordingLoader:
                    def __init__(self, *arguments, **keywords):
                        observed['generator_seed'] = keywords['generator'].initial_seed()
                        self.loader = original_loader(*arguments, **keywords)
                    def __iter__(self):
                        for batch_x, batch_y in self.loader:
                            observed['order'] = batch_x[:, 0, 0, 0].tolist()
                            yield batch_x, batch_y
                try:
                    r.DataLoader = RecordingLoader
                    delta, state, counters = r.train_local(model, r.new_head(), dataset,
                                                           counts, local_seed, 'cpu')
                finally:
                    r.DataLoader = original_loader
                check(observed['generator_seed'] == local_seed and observed['order'] == expected_order,
                      f'{seed}: actual local DataLoader uses new absolute minibatch seed and expected order')
                check(counters['combined_backward_calls'] == 1 and counters['residual_gradient_scalars'] == 650,
                      f'{seed}: new-seed CPU minibatch uses unchanged combined update accounting')
                check(torch.count_nonzero(delta) > 0 and any(torch.count_nonzero(v) > 0 for v in state.values()),
                      f'{seed}: synthetic new-seed update changes shared model and local head')
                q, ef, wire, encoding = r.encode_shared(delta, torch.zeros_like(delta))
                check(len(wire) == 54067 and q.numel() == 136874
                      and encoding['personal_parameters_uploaded'] == 0,
                      f'{seed}: real new-seed encoded upload remains shared-only 54067B')
                check(torch.equal(ef, delta - r.load_codec().decode(wire)),
                      f'{seed}: real new-seed wire decode used in EF')

                config = {'base': base, 'weights': [.05] * 20, 'data_manifest': {'synthetic': True},
                          'data_identities': {}, 'train_adapter_identities': {}}
                config['signature'] = r.signature(config)
                out = root / str(seed)
                out.mkdir()
                r.atomic_json(out / 'config.json', config)
                check(entry.preflight(out, base)[0] == config,
                      f'{seed}: matching prospective config resumes without data loader')
                bad = copy.deepcopy(base)
                bad['partition_seed'] = 20261203
                rejects(lambda: entry.preflight(out, bad), f'{seed}: old partition resume rejected')
                bad = copy.deepcopy(base)
                bad['seed'] = 20261211
                rejects(lambda: entry.preflight(out, bad), f'{seed}: old seed resume rejected')
                bad = copy.deepcopy(base)
                bad['source_identities'][wrapper_relative] = '0' * 64
                rejects(lambda: entry.preflight(out, bad), f'{seed}: wrapper hash resume mismatch rejected')
                bad = copy.deepcopy(base)
                bad['prospective']['plan_sha256'] = '0' * 64
                rejects(lambda: entry.preflight(out, bad), f'{seed}: prospective PLAN resume mismatch rejected')
                bad = copy.deepcopy(base)
                bad['runtime']['prospective_entry']['module_global_overrides']['SEED'] = 0
                rejects(lambda: entry.preflight(out, bad), f'{seed}: runtime override resume mismatch rejected')
                records.append({'seed': seed, 'partition_seed': base['partition_seed'],
                    'rounds': base['rounds'], 'base_signature': r.signature(base),
                    'config_signature': config['signature'],
                    'model_initialization_sha256': __import__('hashlib').sha256(r.vectorize(model).numpy().tobytes()).hexdigest(),
                    'local_batch_seed_round2_client7': local_seed, 'local_batch_order': observed['order'],
                    'actual_wire_bytes': len(wire)})
            check({name: getattr(entry, name) for name in old_globals} == old_globals,
                  f'{seed}: scoped globals restore frozen defaults after use')
        check(len({item['base_signature'] for item in records}) == 3,
              'all three training seeds have distinct base signatures')
        check(len({item['model_initialization_sha256'] for item in records}) == 3,
              'all three training seeds have distinct model initializations')
        with p.module_overrides(entry, 20261232, p.PARTITION_SEED):
            other_seed = p.base_config(entry, 'cpu', 20261232, p.PARTITION_SEED)
        rejects(lambda: entry.preflight(root / '20261231', other_seed),
                'a different allowed prospective seed cannot reuse another seed output')

        # Route the unchanged orchestration through a synthetic adapter, then stop
        # before the first train update: confirms actual prepare/init/seed dispatch.
        dispatch = {}
        class SyntheticAdapter:
            def install_test_guard(self):
                return []
            def prepare(self, partition_seed):
                dispatch['partition_seed_received_by_prepare'] = partition_seed
                return [dataset] * 20, dataset, {'client_label_counts': [counts] * 20}, {}, None, None
            def source_identities(self):
                return {}
        original_adapter, original_train = entry.load_train_adapter, r.train_local
        def stop_before_train(model, head, supplied_dataset, supplied_counts, seed, backend):
            dispatch['first_client_training_seed'] = seed
            dispatch['initialization_matches_seed20261231'] = torch.equal(
                r.vectorize(model), r.vectorize(r.new_model(20261231)))
            raise SyntheticDispatchComplete()
        try:
            entry.load_train_adapter = lambda: SyntheticAdapter()
            r.train_local = stop_before_train
            with p.module_overrides(entry, 20261231, p.PARTITION_SEED):
                base = p.base_config(entry, 'cpu', 20261231, p.PARTITION_SEED)
                try:
                    entry.formal(SimpleNamespace(backend='cpu', out=root / 'dispatch_only'), base, None, None)
                except SyntheticDispatchComplete:
                    dispatch['stopped_before_first_train_update'] = True
        finally:
            entry.load_train_adapter, r.train_local = original_adapter, original_train
        check(dispatch == {'partition_seed_received_by_prepare': 20261217,
            'first_client_training_seed': 20261231, 'initialization_matches_seed20261231': True,
            'stopped_before_first_train_update': True},
            'unchanged orchestration routes new partition/init/batch seed through synthetic adapter')
        check(not (root / 'dispatch_only/checkpoint.pt').exists(),
              'synthetic orchestration dispatch stops before any checkpoint/training update')

        args = SimpleNamespace(seed=20261233, partition_seed=p.PARTITION_SEED,
            backend='cpu', out=root / 'design_only', execute_formal=False)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            result = p.execute(args, entry)
        check(result['status'] == 'DESIGN_ONLY' and not result['formal_training_executed'],
              'prospective execute defaults to DESIGN_ONLY')
        check(not args.out.exists(), 'prospective DESIGN_ONLY creates no training output directory')
        check(result['base_config']['seed'] == 20261233
              and result['base_config']['partition_seed'] == 20261217,
              'prospective DESIGN_ONLY reports explicit new seed/partition')
        check({name: getattr(entry, name) for name in old_globals} == old_globals,
              'DESIGN_ONLY also restores module globals after return')
    expected_plan_hash = p.PLAN_SHA256
    try:
        p.PLAN_SHA256 = '0' * 64
        rejects(p.verify_sources, 'wrong predeclared prospective PLAN hash rejected before imports/data')
    finally:
        p.PLAN_SHA256 = expected_plan_hash
    p.verify_sources()
    check({name: p.sha(p.HERE / name) for name in p.FROZEN_SOURCES} == before,
          'old PLAN/reference/train SHA256 values unchanged')
    check(r.verify_frozen_sources() == frozen_before, 'all original 13 frozen dependencies unchanged')
    check(p.sha(p.__file__) == wrapper_sha and p.sha(__file__) == fixture_sha,
          'prospective wrapper and fixture unchanged during checks')
    check('_fedrod_linear_train_only_adapter' not in sys.modules,
          'real TRAIN loader module never imported')
    check('torchvision' not in sys.modules, 'torchvision dataset stack never imported')
    check(not RAW_OPEN_ATTEMPTS, 'raw data open attempts zero')
    result = {'status': 'PASS', 'checks_passed': len(CHECKS), 'checks': CHECKS,
        'new_plan': p.PLAN_RELATIVE, 'new_plan_sha256': p.PLAN_SHA256,
        'wrapper_sha256': wrapper_sha, 'fixture_sha256': fixture_sha,
        'frozen_old_sources': before, 'per_seed': records, 'synthetic_dispatch': dispatch,
        'raw_data_open_attempts': 0, 'formal_training_executed': False,
        'review': 'REVIEW_UNAVAILABLE / LOCAL_CHECK_ONLY'}
    r.atomic_json(p.HERE / 'PROSPECTIVE_FIXTURES.json', result)
    text = f'''# Prospective FedRoD wrapper: {len(CHECKS)} CPU synthetic checks PASS

只新增prospective.py与独立fixtures；旧PLAN/reference/train的SHA256及原13依赖检查前后不变。新PLAN SHA固定703d5956c9edd24d5a03501a75f678a43222d19bd3dd1ede515ed77545b0830b。未联网、未读原始数据，raw打开尝试0，未导入真实TRAIN loader/torchvision，未正式训练。

显式p20261217，训练seed20261231/32/33，固定20轮。模块SEED/PARTITION_SEED/ROUNDS仅在作用域内覆盖，记录于config/runtime；新PLAN与wrapperhash进入source和config签名。三seed初始化逐位匹配直接冻结SmallCNN，新batch顺序匹配seed+100r+i；三seed签名/初始化不同。旧seed/partition、另一允许seed、wrapper/PLAN/runtime身份误配恢复均拒绝。

各seed只跑7张CPU合成图的1个本地batch；真实shared-only uniform3 wire均54067B，EF经实际decode。一次纯合成编排在首个训练更新前截停，确认prepare实际收到p20261217、模型/首batch实际用20261231。默认DESIGN_ONLY且不创建训练目录，退出恢复旧globals。

沿用原文分离梯度、持久650头和G+P预测，未改旧accuracy STOP。冻结入口client_macro_ce即NLL；Brier/ECE仍需锁定的统一最终评估器补齐。没有prospective准确率/NLL结果；REVIEW_UNAVAILABLE / LOCAL_CHECK_ONLY。
'''
    (p.HERE / 'PROSPECTIVE_FIXTURES.md').write_text(text)
    print(json.dumps({'status': 'PASS', 'checks_passed': len(CHECKS),
        'seeds': list(p.TRAINING_SEEDS), 'partition_seed': p.PARTITION_SEED,
        'actual_wire_bytes': [item['actual_wire_bytes'] for item in records],
        'raw_data_open_attempts': 0, 'formal_training_executed': False}, indent=2))


if __name__ == '__main__':
    main()
