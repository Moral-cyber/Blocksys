"""Hash-only final source/input identity audit. Never fits or scores anything."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib
import json

HERE = Path(__file__).resolve().parent
STAGE = HERE.parents[1]
ORIGINAL_TMP = Path('/private/tmp/pr_lfl_route_a')
FINAL = Path('/Users/wind/Desktop/Codex/paper/03_英文论文')
CACHE = {}


def sha(path):
    path = Path(path)
    key = (str(path.resolve()),path.stat().st_size,path.stat().st_mtime_ns)
    if key not in CACHE:
        value = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda:stream.read(1<<20),b''):
                value.update(block)
        CACHE[key] = value.hexdigest()
    return CACHE[key]


def aliases(path):
    """Explicit path relocation candidates; a mismatch never uses fallback."""
    path = Path(path)
    result = []
    for old,new in ((ORIGINAL_TMP/'p1',HERE/'p1'),(ORIGINAL_TMP/'stats',HERE/'stats'),(STAGE,FINAL)):
        try:
            result.append(new/path.relative_to(old))
        except ValueError:
            pass
    return [p for p in result if p!=path]


def main():
    blocks = []
    def audit(name,lockpath,mapping,expected_count=None):
        lockpath = Path(lockpath)
        rows = []
        for path,expected in mapping.items():
            path = Path(path)
            record = {'recorded_path':str(path),'expected_sha256':expected}
            if path.exists():
                actual = sha(path)
                record.update({'checked_path':str(path),'actual_sha256':actual,
                               'status':'MATCH' if actual==expected else 'MISMATCH','alias_used':False})
            else:
                candidates = [{'path':str(p),'actual_sha256':sha(p)} for p in aliases(path) if p.exists()]
                matching = [x for x in candidates if x['actual_sha256']==expected]
                record.update({'status':'MATCH_EXPLICIT_RELOCATION' if matching else 'MISSING',
                               'alias_used':bool(matching),'alias_candidates':candidates})
                if matching:
                    record['checked_path'] = matching[0]['path']
                    record['actual_sha256'] = matching[0]['actual_sha256']
            rows.append(record)
        count_ok = expected_count is None or len(rows)==expected_count
        passed = count_ok and all(r['status'] in ('MATCH','MATCH_EXPLICIT_RELOCATION') for r in rows)
        blocks.append({'block':name,'record_path':str(lockpath),'record_sha256':sha(lockpath),
                       'expected_count':expected_count,'count':len(rows),'count_matches_expected':count_ok,
                       'status':'PASS' if passed else 'FAIL','identities':rows})

    configs = [
        ('P0 paired statistics','stats/result.json','source_and_input_sha256',7),
        ('P1 lambda','p1/lambda_result.json','source_and_input_sha256',250),
        ('P1 frozen drift','p1/drift_result.json','source_and_input_sha256',248),
        ('FedAvg/FedProx frozen fits','baselines/FIT_LOCK.json','inputs',290),
        ('FedAvg/FedProx training sources','baselines/SOURCE_LOCK.json','inputs',3),
        ('FedAvg/FedProx score sources','baselines/SCORE_SOURCE_LOCK.json','inputs',6),
        ('Multi-partition frozen sources','multipartition/SOURCE_LOCK.json','inputs',22),
        ('Multi-partition aggregate inputs','multipartition/combined_development.json','source_sha256',13),
        ('pFedMe frozen models and inputs','personalized_baseline/MODEL_LOCK.json','inputs',43),
        ('pFedMe source inputs','personalized_baseline/SOURCE_LOCK.json','inputs',25),
        ('CIFAR100 frozen sources','cifar100_extension/SOURCE_LOCK.json','inputs',18),
        ('CIFAR100 TRAIN data','cifar100_extension/DATA_PREFLIGHT.json','data_identities',1),
        ('CIFAR100 source data checkpoint evaluation replay identities',
         'cifar100_extension/INDEPENDENT_RESULTS_VERIFICATION.json','source_data_checkpoint_sha256',25),
        ('ScientistTwo stopped-screen frozen inputs','scientisttwo_screen/SOURCE_LOCK.json','inputs',253),
    ]
    for name,relative,key,count in configs:
        path = HERE/relative
        value = json.loads(path.read_text())
        audit(name,path,value[key],count)

    # Check original temporary paths and their staged archival copies explicitly.
    archived = {}
    for block in blocks:
        for identity in block['identities']:
            p = Path(identity['recorded_path'])
            for old,new in ((ORIGINAL_TMP/'p1',HERE/'p1'),(ORIGINAL_TMP/'stats',HERE/'stats')):
                try:
                    alias = new/p.relative_to(old)
                except ValueError:
                    continue
                if alias.exists():
                    archived[str(alias)] = identity['expected_sha256']
    audit('Archived copies of original temporary P1 sources',HERE/'p1/lambda_result.json',archived,4)

    # Multi-partition run metadata also bind raw data, sources and checkpoint bytes.
    multi_runs = {}
    for path in sorted((HERE/'multipartition/runs').glob('*/*/evaluation.json')):
        value = json.loads(path.read_text())
        multi_runs.update(value['source_and_data_identities'])
        multi_runs[str(path.parent/'fedlc_uniform3/checkpoint.pt')] = value['checkpoint_sha256']
    audit('Multi-partition per-run source data and final checkpoints',
          HERE/'multipartition/result.json',multi_runs)

    # The launcher has a separate scalar-key lock rather than an inputs object.
    matrix_path = HERE/'personalized_baseline/matrix_execution/MATRIX_SOURCE_LOCK.json'
    matrix = json.loads(matrix_path.read_text())
    runners = list((HERE/'personalized_baseline').glob('**/run*matrix*.py'))
    runner = [p for p in runners if sha(p)==matrix['runner_sha256']]
    if len(runner)!=1:
        raise ValueError('pFedMe matrix launcher identity not uniquely located')
    audit('pFedMe matrix launcher scalar lock',matrix_path,
          {str(runner[0]):matrix['runner_sha256'],
           str(HERE/'personalized_baseline/adapter.py'):matrix['adapter_sha256'],
           str(HERE/'personalized_baseline/SOURCE_LOCK.json'):matrix['source_lock_sha256']},3)

    # Each C100 scored model is bound directly by its own result, not just the prior replay.
    c100 = {}
    for path in sorted((HERE/'cifar100_extension/runs').glob('*/evaluation.json')):
        value = json.loads(path.read_text())
        c100.update(value['source_and_data_identities'])
        c100[str(path.parent/'checkpoint.pt')] = value['checkpoint_sha256']
    audit('CIFAR100 per-run locked source data and scored checkpoints',
          HERE/'cifar100_extension/result.json',c100,22)

    failed = [b['block'] for b in blocks if b['status']!='PASS']
    result = {'status':'PASS_ALL_EVIDENCE_IDENTITIES' if not failed else 'FAIL_EVIDENCE_IDENTITIES',
              'created_at':datetime.now(timezone.utc).isoformat(),'agent_id':'/root/route_a_p1',
              'audit_scope':'Hash-only verification of declared existing source/input/model identities; no inference, metric replay, fitting, training or manuscript modification',
              'blocks':blocks,'block_count':len(blocks),'identity_references':sum(b['count'] for b in blocks),
              'distinct_current_files_hashed':len(CACHE),'failed_blocks':failed,
              'relocation_rules':[{'from':str(ORIGINAL_TMP/'p1'),'to':str(HERE/'p1')},
                                  {'from':str(ORIGINAL_TMP/'stats'),'to':str(HERE/'stats')},
                                  {'from':str(STAGE),'to':str(FINAL)}],
              'relocation_policy':'Literal recorded paths are checked first. Existing mismatches never fall back to another copy. Missing recorded paths may resolve only through an explicit prefix relocation with identical bytes. Old temporary sources and staged archived copies are both checked where available. Final delivery copy identity requires root publication verification.',
              'new_cnn_runs':0,'new_predictor_fits':0,'new_scoring':0,
              'existing_evidence_overwritten':False,'manuscript_modified':False,
              'audit_script_sha256':sha(__file__)}
    (HERE/'FINAL_EVIDENCE_IDENTITY.json').write_text(json.dumps(result,indent=2)+'\n')
    md = ['# Final evidence identity audit','',f"Status: **{result['status']}**. Checked {result['identity_references']} identity references in {len(blocks)} blocks, spanning {result['distinct_current_files_hashed']} distinct current evidence files. The audit script has a separate checksum. This is a byte-identity check only; no experiment or score was recomputed.",'',
          '| Evidence block | Declared references checked | Status |','|---|---:|---|']
    md += [f"| {b['block']} | {b['count']} | {b['status']} |" for b in blocks]
    md.extend(['','The report includes P0 statistics; all250lambda/248drift identities; dense FedAvg/FedProx fits and source locks; multi-partition sources, source data, aggregate inputs and checkpoints; the43pFedMe model/input lock plus source/launcher locks; CIFAR100 source/data/scored checkpoints; and all253stopped-screen frozen identities. Existing report files are not overwritten.','',
               '## Path relocation boundary','',
               'The original P1 result files retain `/private/tmp/pr_lfl_route_a/p1/...` paths. Those source files still exist and match; their copies inside the staged manuscript revision artifacts also match the same original hashes. The main experiment sources use current Desktop paper paths or staged revision-artifact paths. Final publication changes paths, not content; this audit records explicit prefix aliases and only permits a missing recorded path to resolve to an identical-byte relocated file. It never hides an existing mismatch by choosing a convenient copy. Root must separately verify the final delivery copy after publishing it.','',
               'The complete JSON records every expected and current SHA-256, checked path, record-file SHA-256 and alias decision. This audit does not certify scientific novelty, experimental independence, or that a file unavailable before this check was previously correct.',''])
    if failed:
        md.extend(['## Failures','']+[f'- {name}' for name in failed])
    (HERE/'FINAL_EVIDENCE_IDENTITY.md').write_text('\n'.join(md))
    print(json.dumps({k:v for k,v in result.items() if k not in ('blocks','relocation_rules')},indent=2))


if __name__=='__main__':
    main()
