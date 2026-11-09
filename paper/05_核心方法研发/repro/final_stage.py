"""Explicit final-stage route; never supplies root authorization or chooses a script."""
from pathlib import Path
import argparse
import importlib.util
import json
import sys

HERE=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument('--component',choices=('predictive_final_test','predictive_manuscript'),required=True)
    parser.add_argument('--operation',choices=('inventory','fixture','driver'),required=True)
    parser.add_argument('--driver-file',help='explicit Python filename inside predictive_final_test; required for driver')
    parser.add_argument('--dry-run',action='store_true')
    args,forward=parser.parse_known_args()
    if forward and forward[0]=='--':forward=forward[1:]
    folder=HERE/'work'/args.component
    manifest=json.loads((HERE/'repro_source_manifest.json').read_text())
    records=[row for row in manifest['records'] if row['path'].startswith('work/'+args.component+'/')]
    if args.operation=='inventory':
        print(json.dumps({'component':args.component,'source_records':records,
            'test_bytes_read':False,'training_started':False,'scope':'source inventory; manuscript and fixtures are not actual final TEST results'},indent=2))
        return
    if args.component!='predictive_final_test':
        parser.error('manuscript route supports inventory only')
    if args.operation=='fixture':
        script=folder/'local_fixture_check.py'
    else:
        if args.driver_file is None:
            parser.error('driver requires explicit --driver-file; no automatic script selection')
        filename=Path(args.driver_file)
        if filename.name!=args.driver_file or filename.suffix!='.py':
            parser.error('driver-file must be a Python basename inside the final-test component')
        script=folder/filename
        if script.name in ('data.py','scorer.py','local_fixture_check.py'):
            parser.error('select an actual root driver, not an imported helper or fixture')
    if not script.is_file():
        parser.error('explicit final-stage entry does not yet exist: '+str(script))
    if args.dry_run:
        print(json.dumps({'component':args.component,'operation':args.operation,'entry':str(script),
            'forwarded':forward,'test_bytes_read':False,'training_started':False,
            'root_lock_created':False,'execute_added':False},indent=2))
        return
    sys.path.insert(0,str(folder))
    sys.argv=[str(script),*forward]
    spec=importlib.util.spec_from_file_location('_explicit_portable_final_stage',script)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if hasattr(module,'main'):
        module.main()
    elif args.operation!='fixture':
        raise RuntimeError('explicit final driver must expose main()')


if __name__=='__main__':main()
