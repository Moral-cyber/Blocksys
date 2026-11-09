"""Read-only current English checks; JSON stdout only, no generated artifacts.

--build-dir selects temporary main.pdf/log/aux/bbl; default uses the delivered
PDF and checks/main.log. This checker never compiles, trains, infers or packages.
"""
from pathlib import Path
import argparse
import hashlib
import json
import re
import sys

sys.dont_write_bytecode = True
METHOD = Path(__file__).resolve().parents[2]
PAPER = METHOD.parent
ENGLISH = PAPER / '03_英文论文'
LEGACY_WORK = '/Users/wind/Documents/Codex/2026-11-02/zhe/work/'
TABLES = ('main', 'quality', 'reference', 'training', 'seeds')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def no_comments(text):
    return re.sub(r'(?<!\\)%[^\n]*', '', text)


def expand(relative, stack=()):
    path = (ENGLISH / relative).resolve()
    if not path.is_relative_to(ENGLISH.resolve()) or path in stack:
        raise ValueError('invalid or cyclic manuscript input: ' + str(path))
    def include(match):
        name = match[1]
        return expand(name if Path(name).suffix else name + '.tex', (*stack, path))
    return re.sub(r'\\input\{([^}]+)\}', include, no_comments(path.read_text()))


def resolve_source(source):
    """Map old result identities without altering their recorded paths or hashes."""
    if source.startswith(LEGACY_WORK):
        return METHOD / 'results' / source[len(LEGACY_WORK):]
    path = Path(source)
    return path if path.is_absolute() else METHOD / path


def check_hashes(sources):
    failures = []
    for source, expected in sources.items():
        path = resolve_source(source)
        if not path.is_file():
            failures.append({'source': source, 'resolved_path': str(path), 'error': 'missing'})
        elif sha(path) != expected:
            failures.append({'source': source, 'resolved_path': str(path), 'error': 'sha256 mismatch'})
    return {'checked': len(sources), 'failures': failures,
            'path_mapping': LEGACY_WORK + '* -> ' + str(METHOD / 'results') + '/*'}


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--build-dir', type=Path)
    parser.add_argument('--expected-cited', type=int, default=20, help='expected citations; 0 disables count assertion')
    parser.add_argument('--expected-bib', type=int, default=22, help='expected bibliography entries; 0 disables count assertion')
    args = parser.parse_args()
    build = args.build_dir.resolve() if args.build_dir else ENGLISH
    checks, failures, unavailable = {}, [], []

    def record(name, operation):
        try:
            checks[name] = {'status': 'PASS', **operation()}
        except Exception as error:
            checks[name] = {'status': 'FAIL', 'error': str(error), 'error_type': type(error).__name__}
            failures.append(name)

    def require(condition, message):
        if not condition:
            raise ValueError(message)

    def sources():
        alltex = expand('main.tex')
        require(re.search(r'\\documentclass(?:\[[^]]*\])?\{svproc\}', alltex), 'svproc class missing')
        require('\\bibliographystyle{spmpsci_unsrt}' in alltex, 'required BST not selected')
        require(not re.search(r'\bTODO\b|\bTBD\b|PLACEHOLDER', alltex), 'unfinished source marker')
        return {'expanded_source_sha256': hashlib.sha256(alltex.encode()).hexdigest()}
    record('manuscript_source', sources)

    def template():
        expected = read_json(ENGLISH / 'checks/TEMPLATE_SHA256.json')
        original = PAPER / '01_论文材料/会议模板_BlockSys2026_用于2027'
        for name in ('svproc.cls', 'spmpsci_unsrt.bst'):
            require(sha(ENGLISH / name) == expected[name] == sha(original / name), name + ' identity mismatch')
        return {'sha256': expected}
    record('template_identity', template)

    def citations():
        first = []
        for group in re.findall(r'\\cite\w*\*?(?:\[[^]]*\])*\{([^}]+)\}', expand('main.tex')):
            for key in group.split(','):
                if key.strip() not in first:
                    first.append(key.strip())
        keys = re.findall(r'@(?!(?:comment|string|preamble)\b)\w+\s*\{\s*([^,\s]+)',
                          no_comments((ENGLISH / 'references.bib').read_text()), re.I)
        require(len(keys) == len(set(keys)), 'duplicate bibliography keys')
        require(set(first) <= set(keys), 'cited keys missing from references.bib')
        require(not args.expected_cited or len(first) == args.expected_cited,
                f'cited count {len(first)} differs from expected {args.expected_cited}')
        require(not args.expected_bib or len(keys) == args.expected_bib,
                f'bibliography count {len(keys)} differs from expected {args.expected_bib}')
        artifacts = {}
        aux, bbl = build / 'main.aux', build / 'main.bbl'
        if aux.exists():
            numbered = re.findall(r'\\bibcite\{([^}]+)\}\{(\d+)\}', aux.read_text())
            require(numbered == [(key, str(i + 1)) for i, key in enumerate(first)], 'aux numbering differs from first citation order')
            artifacts['aux'] = str(aux)
        if bbl.exists():
            items = re.findall(r'\\bibitem(?:\[[^]]*\])?\{([^}]+)\}', bbl.read_text())
            require(items == first, 'bbl order differs from first citation order')
            artifacts['bbl'] = str(bbl)
        if not artifacts:
            unavailable.append('citation_numbering: no main.aux/main.bbl; provide --build-dir')
        return {'cited_count': len(first), 'bibliography_count': len(keys), 'first_citation_order': first,
                'numbering_artifacts': artifacts, 'numbering_status': 'PASS' if artifacts else 'UNAVAILABLE'}
    record('citations', citations)

    def tables():
        previous = read_json(ENGLISH / 'checks/FINAL_CHECKS.json')['tables']
        actual = {name: sha(ENGLISH / f'tables/{name}.tex') for name in TABLES}
        require(all(actual[name] == previous[name] for name in TABLES), 'main table hash differs from saved FINAL_CHECKS')
        return {'sha256': actual, 'comparison': 'saved historical FINAL_CHECKS table hashes only'}
    record('main_table_identity', tables)

    def numbers():
        data = read_json(ENGLISH / 'evidence/PAPER_NUMBERS.json')
        require(data.get('status') == 'DONE', 'PAPER_NUMBERS is not DONE')
        result = check_hashes(data['source_sha256'])
        require(not result['failures'], json.dumps(result['failures'], ensure_ascii=False))
        return {**result, 'scope': data.get('scope'), 'numeric_scope': 'fixed official TEST evidence; no metrics recomputed by this checker'}
    record('official_test_source_identity', numbers)

    def diagnostic(name, field, expected_status):
        path = METHOD / f'results/{name}/result.json'
        data = read_json(path)
        require(data.get('status') == expected_status, name + ' status mismatch')
        result = check_hashes(data[field])
        require(not result['failures'], json.dumps(result['failures'], ensure_ascii=False))
        if name == 'core_contribution_probe':
            require(data['input_bytes_unchanged'] and not data['forbidden_open_attempts'], 'probe input/audit failure')
            require(all(data[key] == 0 for key in ('new_cnn_forwards', 'new_cnn_backwards', 'new_training_seeds', 'official_test_reads')), 'probe exceeds recorded scope')
            require(data['declared_plan_sha256'] == sha(METHOD / 'design/core_contribution/PLAN.md'), 'probe plan hash mismatch')
            scope = {'analysis_type': data['analysis_type'], 'new_vector_fits': data['new_vector_fits'],
                     'tasks': list(data['tasks']), 'official_test_reads': 0,
                     'interpretation': data['interpretation_scope']}
        else:
            require(data['checks_passed'] == len(data['checks']), 'smoke check count mismatch')
            require(data['profile']['rounds'] == len(data['rounds']) == 2, 'smoke round scope mismatch')
            require(data['total_gradient_evaluations'] == sum(r['gradient_evaluations'] for r in data['rounds']), 'smoke gradient count mismatch')
            require(all(r['protocol_direct_model_max_abs_error'] == r['protocol_direct_residual_max_abs_error'] == 0 for r in data['rounds']), 'smoke direct/protocol mismatch')
            require(not data['audit']['forbidden_test_open_attempts'] and data['audit']['network_attempts'] == 0, 'smoke TEST/network audit failure')
            scope = {'scope': data['scope'], 'clients': data['profile']['client_ids'], 'rounds': len(data['rounds']),
                     'checks_passed': data['checks_passed'], 'gradient_evaluations': data['total_gradient_evaluations'],
                     'payload_bytes': data['total_protocol_payload_bytes'], 'application_bytes': data['total_protocol_application_bytes'],
                     'limitations': data['limitations']}
        return {**result, 'result_sha256': sha(path), 'actual_numeric_scope': scope,
                'numeric_validation': 'recorded counts and audit fields only; experiment not rerun'}
    record('core_probe_identity', lambda: diagnostic('core_contribution_probe', 'inputs_and_sources_sha256', 'DONE'))
    record('learning_smoke_identity', lambda: diagnostic('learning_protocol_smoke', 'source_and_input_sha256', 'PASS'))

    def chain_integration():
        path = METHOD / 'results/chain_integration/result.json'
        data = read_json(path)
        require(data.get('status') == 'PASS', 'chain integration status mismatch')
        sources = check_hashes(data['source_and_input_sha256'])
        require(sources['checked'] == 14 and not sources['failures'],
                'chain pinned sources: ' + json.dumps(sources, ensure_ascii=False))
        # The imported training helper was omitted from the chain run's source
        # list. Cross-check its current bytes against the earlier smoke record;
        # this is a delivery-time check, never a claim of pre-run chain pinning.
        helper = METHOD / 'scripts/learning_protocol_smoke.py'
        smoke = read_json(METHOD / 'results/learning_protocol_smoke/result.json')
        require(smoke['source_and_input_sha256'].get(str(helper)) == sha(helper),
                'chain training helper differs from earlier smoke source identity')
        require(data['checks_passed'] == len(data['checks']) == 128, 'chain assertion count mismatch')
        require(data['profile']['client_ids'] == [0, 1] and
                data['profile']['rounds'] == len(data['rounds']) == 2,
                'chain fixed diagnostic scope mismatch')
        require(data['total_gradient_evaluations'] ==
                sum(row['gradient_evaluations'] for row in data['rounds']) == 22,
                'chain gradient count mismatch')
        require(data['counters'] == {'protocol_encode': 4, 'protocol_decode': 8,
                'reference_encode': 4, 'reference_decode': 4}, 'chain codec count mismatch')
        clients = [client for row in data['rounds'] for client in row['clients']]
        require(len(clients) == 4 and len({row['payload_sha256'] for row in clients}) == 4,
                'chain requires four different freshly encoded payloads')
        require(len({row['chain']['tx_hash'] for row in clients}) == 4,
                'chain registration transaction identities not distinct')
        for expected_round, row in enumerate(data['rounds'], 1):
            require(row['round'] == expected_round and
                    row['protocol_direct_model_max_abs_error'] ==
                    row['protocol_direct_residual_max_abs_error'] == 0,
                    'chain round/direct-reference mismatch')
            require([client['client'] for client in row['clients']] == [0, 1],
                    'chain per-round clients mismatch')
            for client in row['clients']:
                receipt = client['chain']
                require(client['EF_generation'] == expected_round and client['payload_bytes'] == 54067,
                        'chain residual generation or payload size mismatch')
                require(receipt['status'] == 1 and receipt['canonical'] is True and
                        receipt['chain_id'] == data['chain']['chain_id'] and
                        receipt['contract_address'] == data['chain']['contract_address'] and
                        receipt['uploader'] == data['chain']['client_addresses']['c' + str(client['client'])],
                        'chain receipt identity/status mismatch')
                require(receipt['confirmation_depth'] >= 2 and
                        receipt['head_block_number'] - receipt['block_number'] >= 2 and
                        receipt['confirmations'] == receipt['confirmation_depth'] + 1,
                        'chain canonical-depth rule mismatch')
                readback = receipt['readback']
                require(readback['round_id'] == expected_round and
                        all(readback[key] == receipt[key] for key in readback),
                        'chain recorded readback mismatch')
        require(data['fresh_training_payload_bytes'] == sum(row['payload_bytes'] for row in clients) == 216268 and
                data['fresh_training_envelope_and_tag_bytes'] ==
                sum(row['application_envelope_and_tag_bytes'] for row in clients) == 217632,
                'chain fresh application byte totals mismatch')
        network = data['network']
        require(len(network['client_HTTP_requests']) == len(network['server_HTTP_responses']) == 25 and
                len(network['bridge_HTTP_requests']) == 23 and
                sum(row['label'] == 'chain_register' for row in network['bridge_HTTP_requests']) == 4,
                'chain HTTP request counts mismatch')
        require(network['total_client_request_body_bytes'] ==
                sum(row['request_body_bytes'] for row in network['client_HTTP_requests']) ==
                sum(row['request_body_bytes_received'] for row in network['server_HTTP_responses']),
                'chain HTTP request body totals mismatch')
        require(network['total_server_response_body_bytes'] ==
                sum(row['response_body_bytes_received'] for row in network['client_HTTP_requests']) ==
                sum(row['response_body_bytes_sent'] for row in network['server_HTTP_responses']),
                'chain HTTP response body totals mismatch')
        bridge = data['chain']['bridge_stats']
        require(bridge['registrations'] == bridge['record_count'] == 4 and
                bridge['rpc_requests'] > 0 and bridge['rpc_request_body_bytes'] > 0 and
                bridge['rpc_response_body_bytes'] > 0, 'chain bridge/RPC scope mismatch')
        audit = data['audit']
        require(not audit['forbidden_test_open_attempts'] and not audit['non_loopback_attempts'] and
                sorted(audit['official_train_files_opened']) == [f'data_batch_{i}' for i in range(1, 6)],
                'chain TEST/network/TRAIN audit mismatch')
        require(data['temporary_databases_and_chain_state_removed'] and
                not data['HMAC_keys_persisted_or_reported'] and not data['private_keys_reported'] and
                len(data['sqlite_integrity']) == 3 and set(data['sqlite_integrity'].values()) == {'ok'},
                'chain storage/key scope mismatch')
        limits = ' '.join(data['limitations'])
        require(all(text in limits for text in ('Single-node Ganache', 'remain trusted',
                'Permanent dropout advancement is not implemented', 'no process kill or OS crash')),
                'chain diagnostic trust/failure scope missing')
        return {**sources, 'result_sha256': sha(path), 'checks_passed': data['checks_passed'],
                'gradient_evaluations': 22, 'rounds': 2, 'fresh_payloads': 4, 'registration_transactions': 4,
                'learning_helper_cross_check': {'sha256': sha(helper),
                    'basis': 'earlier smoke result and current bytes; delivery-time check, not chain pre-run pinning'},
                'rpc_counter_scope': 'recorded bridge snapshot; not an independently replayed RPC trace',
                'scope': data['scope'], 'limitations': data['limitations'],
                'numeric_validation': 'recorded hashes/counts/audits only; training/network/ledger not rerun'}
    record('chain_integration_identity', chain_integration)

    def inference_replay():
        path = METHOD / 'results/replay_saved_inference/result.json'
        data = read_json(path)
        require(data.get('status') == 'PASS' and not data['failures'], 'inference replay did not pass')
        sources = check_hashes(data['source_and_input_sha256_before_after_equal'])
        require(sources['checked'] == 291 and not sources['failures'],
                'replay pinned sources: ' + json.dumps(sources, ensure_ascii=False))
        counts = data['counts']
        require(counts['models'] == len(data['runs']) == 6 and
                counts['client_seed_combinations'] == 120 and
                counts['cnn_forward_images'] == 60000 and counts['cnn_forward_batches'] == 240,
                'replay model/client/forward scope mismatch')
        require(all(counts[key] == 0 for key in ('new_training_rounds', 'new_training_seeds',
                'new_fit_calls', 'new_backward_calls')), 'replay exceeded inference-only scope')
        require(data['maximum_raw_abs_error'] == 0 and data['all_raw_logits_bitwise_equal'] is True and
                data['raw_argmax_disagreements'] == data['final_argmax_disagreements'] == 0,
                'replay raw logits or decisions differ')
        tolerance = data['tolerances_declared_before_replay']['metric_atol']
        require(0 < tolerance <= 2e-6 and data['maximum_main_summary_abs_error'] <= tolerance,
                'replay metric tolerance not met')
        clients = [client for run in data['runs'] for client in run['clients']]
        require(len(clients) == 120 and sum(row['test_rows'] for row in clients) == 60000 and
                all(row['status'] == 'PASS' and row['raw_bitwise_equal'] and
                    row['raw_max_abs_error'] == row['raw_argmax_disagreements'] ==
                    row['final_argmax_disagreements'] == 0 and
                    max(abs(value) for value in row['final_metric_differences'].values()) <= tolerance
                    for row in clients), 'replay per-client checks mismatch')
        require(max(abs(value) for run in data['runs'] for value in run['summary_differences'].values()) ==
                data['maximum_main_summary_abs_error'], 'replay summary maximum mismatch')
        require(all(run['method'] == 'fedlc_uniform3' and run['replay_backend'] == 'CPU' and
                    run['raw_logit_shape'] == [10000, 10] and len(run['clients']) == 20 for run in data['runs']),
                'replay method/backend/client scope mismatch')
        limits = ' '.join(data['limitations'])
        require('reference methods and adapted Linear heads are not rerun' in limits and
                'not external cross-model review' in limits, 'replay scope limitations missing')
        return {**sources, 'result_sha256': sha(path), 'counts': counts,
                'maximum_raw_abs_error': data['maximum_raw_abs_error'],
                'maximum_main_summary_abs_error': data['maximum_main_summary_abs_error'],
                'scope': data['scope'], 'limitations': data['limitations'],
                'numeric_validation': 'recorded hashes/counts/tolerances only; inference not rerun'}
    record('saved_inference_replay_identity', inference_replay)

    def compile_log():
        path = build / 'main.log' if args.build_dir else ENGLISH / 'checks/main.log'
        log = path.read_text(errors='replace')
        bad = [line for line in log.splitlines() if re.search(r'Overfull|Warning|undefined|Missing character|^!', line, re.I)]
        require(not bad, json.dumps(bad))
        return {'path': str(path), 'sha256': sha(path), 'errors_warnings_overfull': bad,
                'underfull_notices': [line for line in log.splitlines() if 'Underfull' in line]}
    record('compile_log', compile_log)

    def pdf():
        from pypdf import PdfReader
        path = build / 'main.pdf'
        reader = PdfReader(path)
        pages = len(reader.pages)
        sizes = [[float(page.mediabox.width), float(page.mediabox.height)] for page in reader.pages]
        chars = [len((page.extract_text() or '').strip()) for page in reader.pages]
        result = {'path': str(path), 'sha256': sha(path), 'pages': pages, 'local_page_limit': 14,
                  'page_sizes_points': sizes, 'extracted_characters_per_page': chars}
        checks['pdf_observations'] = result
        require(1 <= pages <= 14, f'PDF has {pages} pages; existing local target is 1..14')
        require(all(abs(w - 612) < .1 and abs(h - 792) < .1 for w, h in sizes), 'PDF is not Letter')
        require(all(count > 100 for count in chars), 'PDF page text missing or too short')
        return result
    record('pdf', pdf)

    def abstract():
        text = no_comments((ENGLISH / 'sections/abstract.tex').read_text()).split('\\keywords')[0]
        words = len(text.replace('\\begin{abstract}', '').strip().split())
        require(70 <= words <= 150, f'abstract has {words} words; expected 70..150')
        return {'words': words, 'count_method': 'whitespace-delimited abstract before keywords'}
    record('abstract', abstract)

    visual = ENGLISH / 'checks/VISUAL_CHECK.json'
    visual_record = read_json(visual) if visual.exists() else {}
    observed_pdf = checks.get('pdf_observations', {})
    report = {'status': 'FAIL' if failures else ('INCOMPLETE' if unavailable else 'PASS'),
              'language': 'english', 'manuscript': str(ENGLISH), 'build_dir': str(build),
              'checks': checks, 'failed_checks': failures, 'unavailable_checks': unavailable,
              'visual': {'inspection_performed_by_this_checker': False,
                         'recorded_status': visual_record.get('status', 'UNAVAILABLE'),
                         'recorded_pdf_matches_checked_pdf': visual_record.get('pdf_sha256') == observed_pdf.get('sha256') if observed_pdf else False,
                         'required_for_this_build_check': False},
              'external_cross_model_review': 'REVIEW_UNAVAILABLE',
              'operations': {'files_written': 0, 'compilation': False, 'training': False, 'inference': False},
              'scope': 'local source/hash/typesetting checks; not visual approval, independent network inference, or deployment validation'}
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return 1 if failures else (2 if unavailable else 0)


if __name__ == '__main__':
    raise SystemExit(main())
