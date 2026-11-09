"""Create a portable source layout for the bounded development experiments."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/method_followup/repro'
WORK=OUT/'work'
RECORDS=[]
EXCLUDED=[]
ENTRIES=('partial_fusion','receiver_feedback','normalized_sparse','shadow_training',
         'mixed_precision','head_control','control_reference','balanced_quant',
         'prior_bias','calibrated_lowbit_confirm','calibration_baselines','calibrated_fashion',
         'prior_vector','predictive_pfl_confirm','calibrated_topk_reference','fedrod_linear_reference',
         'predictive_final_test','predictive_manuscript','blocksys_template')
PROSPECTIVE_PINS={
    'work/predictive_pfl_confirm/PLAN.md':'703d5956c9edd24d5a03501a75f678a43222d19bd3dd1ede515ed77545b0830b',
    'work/prior_vector/solver.py':'9c3196eb3e6ff7f5f32073e3bc4aae8d986ac7e44bdf0d719258343847d4dbae',
    'work/prior_vector/evaluate_saved.py':'7b45a055df8c3e03421fd9edd6749c1e2bfba1203864e0b41695413db1fb8b05',
    'work/predictive_pfl_confirm/assess.py':'bdb0bac2d4809d100e3112e1516e14d8f295c8dbe77fa4da34ce0543706b7b17'}


def copy(source,target):
    if source.name=='local_test_keys.json' or any(part.endswith('_private_test_keys') for part in source.parts):
        raise ValueError('private synthetic keys cannot enter portable source layout: '+str(source))
    expected=hashlib.sha256(source.read_bytes()).hexdigest()
    relative=str(target.relative_to(OUT))
    existing=[row for row in RECORDS if row['path']==relative]
    if existing:
        if existing[0]['sha256']!=expected or hashlib.sha256(target.read_bytes()).hexdigest()!=expected:
            raise ValueError('conflicting portable source target: '+relative)
        return
    target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(source,target)
    if hashlib.sha256(target.read_bytes()).hexdigest()!=expected:
        raise RuntimeError('portable copy verification failed: '+str(target))
    RECORDS.append({'source':str(source.relative_to(ROOT)),
        'path':relative,'bytes':target.stat().st_size,'sha256':expected})


def main(argv=None):
    global OUT,WORK
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument('--out',type=Path,default=OUT,
        help='isolated source-layout destination; defaults to the delivery repro directory')
    args=parser.parse_args(argv)
    OUT=args.out.expanduser().resolve()
    WORK=OUT/'work'
    RECORDS.clear();EXCLUDED.clear()
    for relative,expected in PROSPECTIVE_PINS.items():
        if hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()!=expected:
            raise ValueError('prospective declared source changed: '+relative)
    # Prior import checks may have populated the generated layout with caches.
    # They are not source evidence and must not enter the portable bundle.
    for cache in sorted(OUT.rglob('__pycache__'),reverse=True):
        if cache.is_dir():
            shutil.rmtree(cache)
    for p in sorted((ROOT/'work/method_followup').glob('*.py')):
        copy(p,WORK/'method_followup'/p.name)
    for folder in ENTRIES:
        for p in sorted((ROOT/'work'/folder).iterdir()):
            if p.is_file() and p.suffix in ('.py','.md','.tex','.bib'):
                copy(p,WORK/folder/p.name)
            elif p.name not in ('__pycache__','.DS_Store'):
                EXCLUDED.append({'source':str(p.relative_to(ROOT)),
                    'reason':'non-runtime evidence/results staged by sync_method_bundle.py'})
    for filename in ('svproc.cls','spmpsci_unsrt.bst'):
        copy(ROOT/'work/blocksys_template/template'/filename,
             WORK/'blocksys_template/template'/filename)
    # Protocol and source locks are runtime identity inputs; result JSONs stay
    # excluded. The FedRoD lock also declares two exact paper-evidence files.
    for relative in ('work/predictive_pfl_confirm/PREREGISTRATION.json',
            'work/predictive_pfl_confirm/ASSESSMENT_LOCK.json',
            'work/fedrod_linear_reference/SOURCE_LOCK.json'):
        copy(ROOT/relative,OUT/relative)
    fedrod_lock=json.loads((ROOT/'work/fedrod_linear_reference/SOURCE_LOCK.json').read_text())
    for relative,expected in fedrod_lock.items():
        path=Path(relative)
        if path.is_absolute() or '..' in path.parts or not path.parts or path.parts[0]!='work':
            raise ValueError('unsafe locked FedRoD dependency: '+relative)
        if hashlib.sha256((ROOT/path).read_bytes()).hexdigest()!=expected:
            raise ValueError('frozen FedRoD dependency changed: '+relative)
        if not any(row['path']==relative for row in RECORDS):
            copy(ROOT/path,OUT/path)
    # Exact shared warmup and its index/source manifest. Original measurements
    # stay under results/; these two files are required runtime inputs.
    for name in ('config.json','warmup.pt'):
        copy(ROOT/'work/partial_fusion/dev_seed20261143'/name,
             WORK/'partial_fusion/dev_seed20261143'/name)
    # The confirm common loader pins this exact config as its TRAIN file/hash
    # reference. It is a declared runtime input, not a copied checkpoint or
    # a recalculated experiment result. Preserve its original relative path.
    reference=Path('work/balanced_quant/development_p20261193_s20261201/ce_uniform3/config.json')
    copy(ROOT/reference,OUT/reference)
    # Standalone CPU fixture imports keep the same bounded work/ layout.
    # Real image routes still verify and bind an explicitly selected project.
    for name in ('fedavg_update_diagnostic.py','sparse_update_benchmark.py'):
        copy(ROOT/'work/tinyproto_diag/tools'/name,WORK/'tinyproto_diag/tools'/name)
    # The frozen v2 tests resolve WORKSPACE from their original work/ layout
    # and verify every upstream byte identity. Preserve all declared paths.
    security=ROOT/'work/durable_integrity_v2'
    for name in ('protocol.py','test_protocol.py','REPORT.md','upstream_hashes_before.json'):
        copy(security/name,WORK/'durable_integrity_v2'/name)
    for source in sorted((security/'specs').rglob('*')):
        if source.is_file():
            copy(source,WORK/'durable_integrity_v2/specs'/source.relative_to(security/'specs'))
    upstream=json.loads((security/'upstream_hashes_before.json').read_text())
    for relative,expected in upstream.items():
        path=Path(relative)
        if path.is_absolute() or '..' in path.parts or not path.parts or path.parts[0]!='work':
            raise ValueError('upstream source must retain a safe work-relative layout: '+relative)
        source=ROOT/path
        if hashlib.sha256(source.read_bytes()).hexdigest()!=expected:
            raise ValueError('frozen upstream hash mismatch: '+relative)
        target=OUT/path
        if not any(row['path']==str(path) for row in RECORDS):
            copy(source,target)
        if hashlib.sha256(target.read_bytes()).hexdigest()!=expected:
            raise ValueError('portable upstream hash mismatch: '+relative)
    identities={}
    for name in ('fedavg_update_diagnostic.py','sparse_update_benchmark.py'):
        p=ROOT/'work/tinyproto_diag/tools'/name
        identities[name]=hashlib.sha256(p.read_bytes()).hexdigest()
    (OUT/'expected_project_sources.json').write_text(json.dumps(identities,indent=2)+'\n')
    copy(ROOT/'work/feedback_portable_launch.py',OUT/'launch.py')
    copy(ROOT/'work/portable_final_stage.py',OUT/'final_stage.py')
    (OUT/'README.md').write_text('''# 当前有界开发实验的复算入口

本包保存源码和确切的共同warmup输入。实验仍为探索性，未达到核心贡献目标。
原始结果在上级results/对应组件；文献证据位于上级literature/，本包不改原结果。
支持partial_fusion、receiver_feedback、normalized_sparse、shadow_training、mixed_precision、head_control、control_reference、balanced_quant、prior_bias、calibrated_lowbit_confirm、calibration_baselines、calibrated_fashion、prior_vector、predictive_pfl_confirm、calibrated_topk_reference和fedrod_linear_reference。
repro_source_manifest.json逐项记录原始路径、复制后路径及哈希；训练和codec源码保持原样。
另收录predictive_final_test与predictive_manuscript，包括.tex/.bib稿件源；draft不等于finalTEST已运行。
final_stage.py提供单独明确路由：manuscript inventory、final-test CPU fixture及root新增driver。
实际driver若尚不存在，明确报错；不自动选择脚本、不生成root-lock、不补--execute。

依赖Python、PyTorch、torchvision、NumPy。当前验证环境为/opt/anaconda3/envs/Tinyproto/bin/python。
原始CIFAR数据缓存保留在既有实验工程，避免复制大数据。工程须含tools/和dataset/Cifar10/rawdata/。
Fashion入口显式指定--fashion-raw，只要求两个官方TRAIN gzip，正式新路由不读取TEST字节。
calibration_baselines只读取显式--input-root下保存的数组，及prior_bias的NumPy fixture，无需--project-root。

从本目录执行，先检查路径（不训练）：

```bash
python launch.py --project-root "/Users/wind/Desktop/Codex/paper/02_实验工程" --entry receiver_feedback --dry-run
```

复算已完成诊断或有限闭环，请分别使用新的输出目录：

```bash
python launch.py --project-root "/实际实验工程" --entry partial_fusion -- --out /新诊断输出 --backend mps
python launch.py --project-root "/实际实验工程" --entry receiver_feedback -- --out /新闭环输出 --backend mps --rounds 20 --seed 20261144
python launch.py --project-root "/实际实验工程" --entry mixed_precision --dry-run -- --out /新量化输出 --backend mps --rounds 20 --seed 20261181 --methods cal_uniform3,cal_mixed_mse,cal_mixed_sensitive
python launch.py --project-root "/实际实验工程" --entry control_reference --dry-run -- --out /新控制参考输出 --backend mps --rounds 20 --seed 20261191
python launch.py --project-root "/实际实验工程" --entry calibrated_lowbit_confirm --operation train --dry-run -- --out /新CIFAR确认输出 --backend mps --rounds 20 --partition-seed 20261203 --seed 20261211
python launch.py --project-root "/实际实验工程" --entry calibrated_lowbit_confirm --operation evaluate --dry-run -- --model-root /同一便携位置新训练输出 --out /新CIFAR评估输出
python launch.py --project-root "/实际实验工程" --entry calibrated_fashion --fashion-raw /实际FashionMNIST/raw --operation train --dry-run -- --out /新Fashion输出 --backend mps --rounds 20 --partition-seed 20261207 --seed 20261221
python launch.py --project-root "/实际实验工程" --entry calibrated_fashion --fashion-raw /实际FashionMNIST/raw --operation evaluate --dry-run -- --model-root /同一便携位置新Fashion训练输出 --out /新Fashion评估输出
python launch.py --entry calibration_baselines -- --input-root /已完成的prior_bias或confirm评估数组目录 --out /新强校准对照输出
python launch.py --project-root "/实际实验工程" --entry prior_bias --probe-model-root /原冻结balanced模型目录 --probe-baseline-root /原完成prior_personalization目录 --dry-run -- --out /新bias诊断输出
```

mixed_precision的run.py、codec.py和EXPERIMENT_PLAN.md保留在同一目录；启动器优先提供入口所在目录的导入路径。
上例mixed_precision仅检查路由与参数转发；实际复算时移除--dry-run，并始终使用新的--out。

MPS不可用时明确报错，若主动改为cpu将形成另一数值后端，不能合并为原MPS重复。
receiver_feedback读取随包提供的同分区10轮warmup；更改seed只改变后继训练随机性，不重建warmup。
索引划分、源代码/运行时身份及完整数据成员会写到新输出配置。移动路径会改变运行签名，
因此新位置复算应使用新out，不能当原路径checkpoint继续。先核对两helper的固定源码哈希。
calibrated_lowbit_confirm与Fashion的evaluate严格检查完整来源/配置，应使用同一便携位置、同一工程新训练的checkpoint；不能给原路径checkpoint换来源。
prior_bias保留原始基线输入字典中的绝对模型路径校验，显式提供的两个原结果目录必须仍满足该关系；不自动重写旧路径或旧结果。
强校准probe需要完整4组旧模型或3组confirm模型、每组20个client保存数组和DONE result.json；本源码包不复制这些数组/模型。

CPU fixture入口（合成数据与NumPy校验，不是正式ML质量结果）：

```bash
python -B launch.py --project-root "/实际实验工程" --entry calibrated_lowbit_confirm --operation fixture
python -B launch.py --entry prior_bias --operation fixture
python -B launch.py --entry calibration_baselines --operation fixture
python -B launch.py --project-root "/实际实验工程" --entry calibrated_fashion --fashion-raw /实际FashionMNIST/raw --operation fixture
```

Fashion fixture额外只读校验真实TRAIN gzip与分区，真实数据CNN forward/backward为0；正式训练仍需单独明确调用train入口。

## 新prospective路线：必须显式operation

新PLAN、registration、assessment锁以及solver/evaluator/assess保持原SHA。新NLL研究问题与旧accuracy STOP分开。
主路线把冻结CIFAR/Fashion训练与原评估作为明确操作；launcher先校验新PLAN及规定partition/seed/20轮，
不改冻结训练source或其原签名规则。新位置使用fresh out；saved-array适配与assessment分别绑定新PLAN/registration。
下面都是dry-run，验证路由，不读图像/新数组或启动训练：

```bash
python launch.py --project-root /实际实验工程 --entry predictive_pfl_confirm --operation train-cifar --dry-run -- --partition-seed 20261217 --seed 20261231 --rounds 20 --backend mps --out /新CIFARprospective输出
python launch.py --project-root /实际实验工程 --entry predictive_pfl_confirm --operation evaluate-cifar --dry-run -- --model-root /同位置新CIFAR模型 --out /新原评估输出
python launch.py --project-root /实际实验工程 --entry predictive_pfl_confirm --operation train-fashion --fashion-raw /实际FashionMNIST/raw --dry-run -- --partition-seed 20261219 --seed 20261241 --rounds 20 --backend mps --out /新Fashionprospective输出
python launch.py --entry predictive_pfl_confirm --operation adapt --dry-run -- --input-root /已完成新原评估目录 --out /新20维适配目录
python launch.py --entry predictive_pfl_confirm --operation assess --dry-run -- --task cifar --seed 20261231 --evaluation-root /已完成新原评估目录 --adaptation-root /已完成新20维适配目录 --out /新阶段判定目录
python launch.py --entry prior_vector --operation saved-evaluate --dry-run -- --input-root /已完成原评估目录 --out /新通用适配目录 --protocol-plan work/predictive_pfl_confirm/PLAN.md --protocol-registration work/predictive_pfl_confirm/PREREGISTRATION.json
python launch.py --project-root /实际实验工程 --entry calibrated_topk_reference --operation prospective-train --dry-run -- --partition-seed 20261217 --seed 20261231 --rounds 20 --backend mps --out /新TopK输出
python launch.py --project-root /实际实验工程 --entry calibrated_topk_reference --operation prospective-evaluate --dry-run -- --model-root /同位置新TopK模型 --out /新TopK评估
python launch.py --entry fedrod_linear_reference --operation prospective --dry-run -- --partition-seed 20261217 --seed 20261231 --backend cpu --out /新FedRoD设计预检
```

TopK与predictive adapt若未显式传入protocol参数，launcher补充并显示本包的固定PLAN/registration路径。
FedRoD prospective本身默认DESIGN_ONLY；只有另加原入口--execute-formal才启动固定20轮，launcher不会补这个开关。
其formal路线另需--project-root绑定原TRAIN cache。三个新方法入口都没有隐式默认operation。
FedRoD SOURCE_LOCK及两份精确paper来源证据按原work相对布局复制，供原校验使用。
predictive assess沿用冻结旧source/registration检查；不能重写adapted输入的绝对来源关系或用原位置checkpoint换身份。

新增CPU检查（仅合成toy，无真实prospective数组）：

```bash
python -B launch.py --entry prior_vector --operation fixture
python -B launch.py --entry prior_vector --operation saved-fixture
python -B launch.py --entry calibrated_topk_reference --operation prospective-fixture
python -B launch.py --entry fedrod_linear_reference --operation prospective-fixture
```

11维prior-anchored BCTS是单独机制control，不是新candidate/gate。原主solver/evaluator不变。
portable路径改变后，先运行以下纯synthetic fixture以生成当前路径的BCTS_CONTROL_FIXTURES.json，
再显式给出同一原评估和对应已完成20维适配目录；control只拟合11维，复用20维状态，不重拟合主方法。

```bash
python -B launch.py --entry prior_vector --operation bcts-fixture
python -B launch.py --entry prior_vector --operation bcts-control --dry-run -- --input-root /同位置fresh原评估 --vector-root /对应已完成20维适配 --out /新11维control --protocol-plan work/predictive_pfl_confirm/PLAN.md --protocol-registration work/predictive_pfl_confirm/PREREGISTRATION.json
```

BCTS_CONTROL_PLAN.md及control/probe/fixture源码原样复制；现有fixture JSON是来源证据，
不以旧路径fixture冒充便携路径可直接续用。44B/80B状态、FP64评分和不同维度ridge解释见其独立说明。

最终阶段增量（只列清单或dry-run）：

```bash
python -B final_stage.py --component predictive_manuscript --operation inventory
python -B final_stage.py --component predictive_final_test --operation fixture --dry-run
python -B final_stage.py --component predictive_final_test --operation driver --driver-file driver.py --dry-run -- --root-lock /root审阅后生成的锁 --execute
```

driver-file必须是predictive_final_test组件内显式的Python文件且已存在。--dry-run不导入driver、不读TEST。
真正finalTEST由root自行决定；root-lock绑定本包路径下实际PLAN/data/scorer指纹，移动后使用fresh输出，
不可改原签名/旧结果或把synthetic fixtures当真实finalTEST。HELPER_REVIEW.md列明类列序和sumCE/TRAIN-state来源等driver责任。

字节含义必须区分：部分融合旧诊断为解析公式；后继闭环生成真实应用序列化字节计长度，
训练聚合仍使用等价原张量（独立decoder fixture已核查），没有实际网络、链或认证测量。
共同warmup开销按公式追记，不是历史网络抓包。以上入口仅面向TRAIN划分的开发集。

本地持久化协议v2保持work/下的完整相对源码布局，包括specs、固定v1五文件、冻结量化codec和旧toy。
从本repro目录执行以下命令可运行62项synthetic protocol tests，不需要实验工程或ML训练：

```bash
python -B work/durable_integrity_v2/test_protocol.py --output-dir work/durable_integrity_v2/新测试目录
```

新测试目录必须位于work/durable_integrity_v2/内，并且此前不存在synthetic_cases或同名_private_test_keys。
随机测试keys在旁边新生成，发布清单排除*_private_test_keys及local_test_keys.json；原始来源的keys不随包复制。
SQLite协议测试不表示真实网络、链共识或ML质量验证。
''',encoding='utf-8')
    manifest={'scope':'runtime sources, accompanying Markdown, exact partial-fusion warmup, fixed balanced config TRAIN identity reference and declared frozen security references; private test keys excluded',
        'entries':list(ENTRIES),
        'external_runtime_inputs':{
            'image_routes':'explicit project helpers and official CIFAR TRAIN cache; Fashion requires explicit two-TRAIN-gzip raw directory',
            'prior_bias':'explicit original frozen model root and completed prior_personalization arrays; original absolute-path identity relationship retained',
            'calibration_baselines':'explicit completed old-four or confirm-three saved-array result directory; no images or project needed',
            'prior_vector':'explicit complete saved-array source and protocol PLAN/registration; no decision in generic evaluator',
            'prospective':'copied immutable PLAN/registration/assessment and FedRoD source locks; every new route requires explicit operation and fresh outputs',
            'checkpoint_portability':'source paths are signed; use fresh portable output, never rewrite or reuse an original-location checkpoint'},
        'security_entry':'work/durable_integrity_v2/test_protocol.py',
        'prospective_source_pins':PROSPECTIVE_PINS,
        'fedrod_locked_reference_hashes':fedrod_lock,
        'security_upstream_hashes':upstream,
        'records':RECORDS,'excluded_from_runtime_layout':EXCLUDED,
        'files':len(RECORDS),'bytes':sum(row['bytes'] for row in RECORDS)}
    (OUT/'repro_source_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'output':str(OUT),'files':sum(p.is_file() for p in OUT.rglob('*')),
        'source_manifest':'repro_source_manifest.json','verified_source_copies':len(RECORDS)}))


if __name__=='__main__':
    main()
