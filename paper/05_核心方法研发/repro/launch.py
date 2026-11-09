"""Portable entry for bounded development experiments; sources unchanged."""
import argparse
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
DEFAULT_FILES={name:'run.py' for name in (
    'receiver_feedback','normalized_sparse','shadow_training','mixed_precision',
    'head_control','control_reference','balanced_quant')}
DEFAULT_FILES.update({'partial_fusion':'probe_fusion.py','prior_bias':'probe.py',
    'calibrated_lowbit_confirm':'train.py','calibration_baselines':'probe.py',
    'calibrated_fashion':'train.py'})
EXPLICIT_ROUTES={
    'prior_vector':{'saved-evaluate':('prior_vector','evaluate_saved.py'),
        'fixture':('prior_vector','local_fixture_check.py'),
        'saved-fixture':('prior_vector','evaluate_saved_fixture.py'),
        'bcts-control':('prior_vector','bcts_control_probe.py'),
        'bcts-fixture':('prior_vector','bcts_control_fixture.py')},
    'predictive_pfl_confirm':{
        'train-cifar':('calibrated_lowbit_confirm','train.py'),
        'evaluate-cifar':('calibrated_lowbit_confirm','evaluate.py'),
        'train-fashion':('calibrated_fashion','train.py'),
        'evaluate-fashion':('calibrated_fashion','evaluate.py'),
        'adapt':('prior_vector','evaluate_saved.py'),
        'assess':('predictive_pfl_confirm','assess.py')},
    'calibrated_topk_reference':{
        'prospective-train':('calibrated_topk_reference','prospective.py'),
        'prospective-evaluate':('calibrated_topk_reference','prospective_evaluate.py'),
        'prospective-fixture':('calibrated_topk_reference','prospective_fixture_check.py')},
    'fedrod_linear_reference':{
        'prospective':('fedrod_linear_reference','prospective.py'),
        'prospective-fixture':('fedrod_linear_reference','prospective_fixture_check.py')}}
PROSPECTIVE_PINS={
    'work/predictive_pfl_confirm/PLAN.md':'703d5956c9edd24d5a03501a75f678a43222d19bd3dd1ede515ed77545b0830b',
    'work/predictive_pfl_confirm/PREREGISTRATION.json':'1eb7d284c43992a3cca58e31448a4b595b88e67abdf167c24a605342bf2f0568',
    'work/prior_vector/solver.py':'9c3196eb3e6ff7f5f32073e3bc4aae8d986ac7e44bdf0d719258343847d4dbae',
    'work/prior_vector/evaluate_saved.py':'7b45a055df8c3e03421fd9edd6749c1e2bfba1203864e0b41695413db1fb8b05',
    'work/predictive_pfl_confirm/assess.py':'bdb0bac2d4809d100e3112e1516e14d8f295c8dbe77fa4da34ce0543706b7b17'}
TASKS={'cifar':(20261217,(20261231,20261232,20261233)),
       'fashion':(20261219,(20261241,20261242,20261243))}


def option_value(forward,name):
    value=None
    for index,argument in enumerate(forward):
        if argument==name:
            if index+1>=len(forward):
                raise ValueError('missing forwarded value: '+name)
            value=forward[index+1]
        elif argument.startswith(name+'='):
            value=argument.split('=',1)[1]
    return value


def prospective_metadata(parser,args,forward):
    if args.entry not in ('predictive_pfl_confirm','calibrated_topk_reference','fedrod_linear_reference'):
        return None
    for relative,expected in PROSPECTIVE_PINS.items():
        path=HERE/relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
            parser.error('prospective frozen source differs: '+str(path))
    plan=HERE/'work/predictive_pfl_confirm/PLAN.md'
    registration=HERE/'work/predictive_pfl_confirm/PREREGISTRATION.json'
    # Only entry points whose frozen parser accepts these options receive them.
    needs_plan=(args.entry=='calibrated_topk_reference' and args.operation!='prospective-fixture') or (
        args.entry=='predictive_pfl_confirm' and args.operation=='adapt')
    if needs_plan and option_value(forward,'--protocol-plan') is None:
        forward+=['--protocol-plan',str(plan)]
    if args.entry=='predictive_pfl_confirm' and args.operation=='adapt' and option_value(forward,'--protocol-registration') is None:
        forward+=['--protocol-registration',str(registration)]
    chosen_plan=option_value(forward,'--protocol-plan')
    if chosen_plan is not None and hashlib.sha256(Path(chosen_plan).expanduser().read_bytes()).hexdigest()!=PROSPECTIVE_PINS['work/predictive_pfl_confirm/PLAN.md']:
        parser.error('prospective operation requires the exact frozen protocol PLAN')
    chosen_registration=option_value(forward,'--protocol-registration')
    if chosen_registration is not None and hashlib.sha256(Path(chosen_registration).expanduser().read_bytes()).hexdigest()!=PROSPECTIVE_PINS['work/predictive_pfl_confirm/PREREGISTRATION.json']:
        parser.error('prospective operation requires the exact frozen registration')
    if args.entry=='predictive_pfl_confirm':
        task=('fashion' if args.operation.endswith('fashion') else 'cifar'
            if args.operation.endswith('cifar') else option_value(forward,'--task') or 'source_result')
    else:
        task='cifar'
    training=(args.entry=='predictive_pfl_confirm' and args.operation.startswith('train-')) or (
        args.entry=='calibrated_topk_reference' and args.operation=='prospective-train') or (
        args.entry=='fedrod_linear_reference' and args.operation=='prospective')
    if training:
        partition,seeds=TASKS[task]
        for name in ('--seed','--partition-seed'):
            if option_value(forward,name) is None:
                parser.error('explicit '+name+' required for prospective portable operation')
        if (int(option_value(forward,'--seed')) not in seeds
                or int(option_value(forward,'--partition-seed'))!=partition
                or int(option_value(forward,'--rounds') or '20')!=20):
            parser.error('prospective operation requires its registered partition/seed/20 rounds')
    return {'plan':str(plan),'plan_sha256':PROSPECTIVE_PINS['work/predictive_pfl_confirm/PLAN.md'],
        'registration':str(registration),'task_route':task,
        'binding':'launcher preflight plus copied immutable protocol; frozen training source and its source-signature rules are unchanged',
        'previous_accuracy_STOP':'unchanged'}


def main():
    p=argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('--project-root',type=Path,
        help='required for image/model routes; saved-array calibration and NumPy fixtures need no project')
    p.add_argument('--entry',choices=tuple(DEFAULT_FILES)+tuple(EXPLICIT_ROUTES),required=True)
    p.add_argument('--operation',choices=('run','train','evaluate','probe','fixture','saved-evaluate',
        'saved-fixture','bcts-control','bcts-fixture','train-cifar','evaluate-cifar','train-fashion','evaluate-fashion',
        'adapt','assess','prospective-train','prospective-evaluate','prospective-fixture','prospective'),
        help='default is the frozen entry; fixture runs local_fixture_check.py')
    p.add_argument('--fashion-raw',type=Path,
        help='FashionMNIST raw directory; only the two official TRAIN gzip files are required')
    p.add_argument('--probe-model-root',type=Path,
        help='prior_bias probe only: original frozen balanced model root')
    p.add_argument('--probe-baseline-root',type=Path,
        help='prior_bias probe only: original completed prior_personalization saved-array root')
    p.add_argument('--dry-run',action='store_true')
    args,forward=p.parse_known_args()
    if forward and forward[0]=='--':
        forward=forward[1:]
    if args.entry in EXPLICIT_ROUTES:
        if args.operation not in EXPLICIT_ROUTES[args.entry]:
            p.error('explicit supported --operation required for '+args.entry+': '+', '.join(EXPLICIT_ROUTES[args.entry]))
        folder,file=EXPLICIT_ROUTES[args.entry][args.operation]
    else:
        folder=args.entry
        files={'run':'run.py','train':'train.py','evaluate':'evaluate.py',
            'probe':'probe.py','fixture':'local_fixture_check.py'}
        if args.operation is not None and args.operation not in files:
            p.error('unsupported operation for '+args.entry)
        file=DEFAULT_FILES[args.entry] if args.operation is None else files[args.operation]
    entry=HERE/'work'/folder/file
    if not entry.is_file():
        p.error('missing portable entry: '+str(entry))
    protocol=prospective_metadata(p,args,forward)
    array_only=args.entry in ('calibration_baselines','prior_vector') or (
        args.entry=='prior_bias' and args.operation=='fixture')
    array_only=array_only or (args.entry=='predictive_pfl_confirm' and args.operation in ('adapt','assess'))
    array_only=array_only or (args.entry in ('calibrated_topk_reference','fedrod_linear_reference') and args.operation=='prospective-fixture')
    array_only=array_only or (args.entry=='fedrod_linear_reference' and '--execute-formal' not in forward)
    uses_fashion=args.entry=='calibrated_fashion' or (
        args.entry=='predictive_pfl_confirm' and args.operation.endswith('fashion'))
    project=None if args.project_root is None else args.project_root.expanduser().resolve()
    if not array_only:
        if project is None:
            p.error('--project-root is required for this image/model route')
        expected=json.loads((HERE/'expected_project_sources.json').read_text())
        for name,value in expected.items():
            source=project/'tools'/name
            if not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest()!=value:
                p.error('missing or different frozen project helper: '+str(source))
        if not uses_fashion and args.operation!='fixture' and not (project/'dataset/Cifar10/rawdata').is_dir():
            p.error('missing official CIFAR raw cache in selected project')
    fashion_raw=None
    if uses_fashion:
        if args.fashion_raw is None:
            p.error('--fashion-raw is required; Fashion data are not copied into this source bundle')
        fashion_raw=args.fashion_raw.expanduser().resolve()
        for name in ('train-images-idx3-ubyte.gz','train-labels-idx1-ubyte.gz'):
            if not (fashion_raw/name).is_file():
                p.error('missing official Fashion TRAIN file: '+str(fashion_raw/name))
    if args.entry=='prior_bias' and args.operation!='fixture':
        if args.probe_model_root is None or args.probe_baseline_root is None:
            p.error('prior_bias needs --probe-model-root and --probe-baseline-root; frozen results are not runtime sources')
    if args.dry_run:
        print(json.dumps({'project':None if project is None else str(project),
            'entry':str(entry),'fashion_raw':None if fashion_raw is None else str(fashion_raw),
            'array_only':array_only,'source_mutations':False,
            'operation':args.operation,'prospective_protocol':protocol,
            'training_started':False,'forwarded':forward},indent=2))
        return
    if args.entry=='predictive_pfl_confirm' and args.operation.startswith('evaluate-'):
        model_root=option_value(forward,'--model-root')
        if model_root is None:
            p.error('prospective evaluation requires explicit --model-root')
        config=json.loads((Path(model_root)/'ce_uniform3/config.json').read_text())
        task='fashion' if uses_fashion else 'cifar'
        partition,seeds=TASKS[task]
        if config['partition_seed']!=partition or config['seed'] not in seeds or config['rounds']!=20:
            p.error('prospective evaluation requires complete registered partition/seed/20-round models')
    if not array_only:
        scripts=HERE/'work/method_followup'
        sys.path.insert(0,str(scripts))
        spec=importlib.util.spec_from_file_location('_portable_project_binding',scripts/'launch.py')
        launcher=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(launcher)
        launcher.bind_project(project)
        partitions=importlib.import_module('run_partitions')
        partitions.RAW=project/'dataset/Cifar10/rawdata'
    # The frozen mixed-precision runner imports its sibling codec by name.
    # Preserve that source layout and resolve the entry's own modules first.
    sys.path.insert(0,str(entry.parent))
    if fashion_raw is not None:
        adapter=importlib.import_module('adapter')
        if Path(adapter.__file__).resolve()!=(entry.parent/'adapter.py').resolve():
            raise RuntimeError('wrong Fashion adapter imported; use a fresh process')
        adapter.fashion.RAW=fashion_raw
    spec=importlib.util.spec_from_file_location('_bounded_reproduction_entry',entry)
    module=importlib.util.module_from_spec(spec)
    sys.argv=[str(entry),*forward]
    spec.loader.exec_module(module)
    if args.entry=='prior_bias' and args.operation!='fixture':
        module.MODEL_ROOT=args.probe_model_root.expanduser().resolve()
        module.BASELINE_ROOT=args.probe_baseline_root.expanduser().resolve()
    if hasattr(module,'main'):
        module.main()
    elif args.operation not in ('fixture','saved-fixture','bcts-fixture','prospective-fixture'):
        raise RuntimeError('portable entry has no main function')


if __name__=='__main__':
    main()
