"""Full-shape protocol integration around one frozen, training-generated wire.

Only the 54067-byte first wire comes from a historical real training run.
Both client updates and all server/EF states below are synthetic fixtures.
The encoder deliberately REPLAYS that wire; it does not quantize these updates.
No ML training, dataset access, network, blockchain, or benchmark is performed.
"""
from __future__ import annotations
import argparse
import ast
from dataclasses import replace
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import secrets
import sqlite3
import struct
import subprocess
import sys
import time

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[1]
V2 = WORKSPACE / 'work/durable_integrity_v2'
FROZEN_CODEC = WORKSPACE / 'work/mixed_precision/codec.py'
TRAIN_ROOT = WORKSPACE / 'work/balanced_quant/development_p20261193_s20261201/fedlc_uniform3'
MODEL_SOURCE = WORKSPACE / 'work/tinyproto_diag/tools/fedavg_update_diagnostic.py'
CODEC_NAME = 'frozen-mixed-precision-v1'
ACTIVITY = 'fullwire-synthetic-integration'
WEIGHTS = {'c1': .25, 'c2': .75}
P = 136874


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2))


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def deny_data_network(event, args):
    if event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):
        value = os.fsdecode(args[0])
        if '/dataset/' in value or value.endswith(('.pt', '.pth', '.h5', '.hdf5')):
            raise AssertionError('official data / checkpoints are forbidden in this integration fixture')
    if event in ('socket.connect', 'socket.getaddrinfo'):
        raise AssertionError('network is forbidden in this integration fixture')


sys.addaudithook(deny_data_network)
import numpy as np
import torch
from torch import nn
torch.set_num_threads(1)


def initialize(out):
    protocol = load_module('_full_payload_durable_v2', V2 / 'protocol.py')
    specpath = V2 / 'specs' / (CODEC_NAME + '.json')
    spec = json.loads(specpath.read_text())
    assert sha(FROZEN_CODEC) == spec['reference_source_sha256'], 'codec source differs from pinned spec'
    codec_module = load_module('_full_payload_frozen_codec', FROZEN_CODEC)
    source_conf = json.loads((TRAIN_ROOT / 'config.json').read_text())
    assert source_conf['codec_sha256'] == sha(FROZEN_CODEC)
    assert source_conf['source_sha256'] == sha(WORKSPACE / 'work/balanced_quant/run.py')
    assert source_conf['plan_sha256'] == sha(WORKSPACE / 'work/balanced_quant/PLAN.md')
    assert source_conf['parameter_count'] == P
    wire = (TRAIN_ROOT / 'first_payload.bin').read_bytes()
    assert len(wire) == 54067
    q = codec_module.decode(wire).numpy().astype(np.float64)
    assert q.shape == (P,) and np.isfinite(q).all()
    # Execute ONLY the standalone class AST; do not import source dataset code.
    tree = ast.parse(MODEL_SOURCE.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'SmallCNN')
    namespace = {'nn': nn}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(MODEL_SOURCE), 'exec'), namespace)
    model = namespace['SmallCNN']()
    with torch.no_grad():
        for value in model.parameters():
            value.zero_()
    layout_rows = []
    offset = 0
    for name, value in model.named_parameters():
        layout_rows.append({'name': name, 'shape': list(value.shape), 'dtype': 'float32',
                            'offset': offset, 'numel': value.numel()})
        offset += value.numel()
    assert offset == P
    layout_doc = {'model': 'SmallCNN', 'coordinate_order': 'named_parameters insertion order; row-major flatten',
                  'parameters': layout_rows, 'parameter_count': P}
    layout = hashlib.sha256(json.dumps(layout_doc, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    key_path = out.parent / (out.name + '_private_test_keys') / 'keys.json'
    keys = {client: bytes.fromhex(value) for client, value in json.loads(key_path.read_text()).items()}
    counter = {'encode_calls': 0, 'decode_calls': 0}
    def replay_encoder(values):
        assert len(values) == P
        counter['encode_calls'] += 1
        return wire
    def real_decode(payload):
        counter['decode_calls'] += 1
        return codec_module.decode(payload).tolist()
    codec = protocol.Codec(CODEC_NAME, protocol.frozen_spec_identity(CODEC_NAME), replay_encoder, real_decode)
    server = protocol.LocalServer(out / 'server.sqlite', keys, {codec.name: codec})
    def client(name, layout_override=None):
        return protocol.ClientStore(out / (name + '.sqlite'), activity=ACTIVITY, client=name, key=keys[name],
                                    layout=layout_override or layout, initial_residual=[0.] * P)
    return protocol, codec_module, wire, q, model, layout_doc, layout, keys, counter, codec, server, client


def envelope_path(out, client):
    return out / ('envelope_' + client + '.bin')


def load_envelope(protocol, out, client):
    raw = envelope_path(out, client).read_bytes()
    return protocol.Envelope(raw[:-32], raw[-32:])


def synthetic_update(q, client):
    i = np.arange(P, dtype=np.int64)
    # Distinct deterministic full-shape perturbations, NEVER historical deltas.
    perturbation = ((i % 11) - 5) * (2. ** -12) if client == 'c1' else ((i % 13) - 6) * (2. ** -11)
    return q + perturbation


def vector_sha(value):
    return hashlib.sha256(np.asarray(value, dtype='<f8').tobytes()).hexdigest()


def phase(out, name):
    (pr, cm, payload, q, model, layout_doc, layout, keys, counter, codec, server, client) = initialize(out)
    checks = []
    def check(label, condition):
        assert bool(condition), label
        checks.append({'name': label, 'pass': True})
    def equal(label, a, b):
        check(label, np.array_equal(np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)))
    def rejects(label, call, expected, decode_delta=None, encode_delta=None):
        old = dict(counter)
        try:
            call()
        except expected as err:
            check(label, True)
            detail = {'exception_type': type(err).__name__, 'reason': str(err)}
        else:
            raise AssertionError(label + ': accepted unexpectedly')
        if decode_delta is not None:
            check(label + '_decode_count', counter['decode_calls'] - old['decode_calls'] == decode_delta)
        if encode_delta is not None:
            check(label + '_encode_count', counter['encode_calls'] - old['encode_calls'] == encode_delta)
        return detail
    result = {'phase': name, 'pid': os.getpid(), 'scope': 'simulation_only; synthetic integration around a real training-generated first wire',
              'checks': checks, 'counter': counter, 'official_data_loaded': False, 'ml_training_performed': False,
              'network_used': False}
    base0 = pr.sm3(pr.vector_bytes([0.] * P))
    envs = {} if name == 'prepare' else {c: load_envelope(pr, out, c) for c in WEIGHTS}
    if name == 'prepare':
        base = server.create_activity(ACTIVITY, [0.] * P, layout=layout, codec=codec.name, fixed_weights=WEIGHTS)
        check('zero_model_hash', base == base0)
        check('open_round_base', server.open_round(ACTIVITY, 1) == base)
        packet_sizes = {}
        for c in WEIGHTS:
            cl = client(c)
            update = synthetic_update(q, c)
            env = cl.prepare(round_id=1, record_id='record-' + c, nonce='nonce-' + c, base_model_hash=base,
                             codec=codec, update=update.tolist())
            envs[c] = env
            check('real_first_wire_' + c, env.packet.payload == payload)
            check('full_dimension_' + c, len(codec.decode(env.packet.payload)) == P)
            check('hmac_32_bytes_' + c, len(env.tag) == 32)
            equal('prepare_keeps_committed_residual_zero_' + c, cl.snapshot()['residual'], np.zeros(P))
            with cl.connection() as conn:
                row = conn.execute('SELECT * FROM pending').fetchone()
                equal('rnext_exact_update_minus_realdecode_' + c, pr.bytes_vector(row['rnext']), update - q)
                check('rnext_nonzero_' + c, any(x != 0 for x in pr.bytes_vector(row['rnext'])))
            check('unknown_status_no_EF_commit_' + c, not cl.commit_if_incorporated('record-' + c, server.query))
            # The test serializes the existing Envelope as wire || fixed 32B tag.
            # This is an application-level byte count, not a network framing claim.
            envelope_path(out, c).write_bytes(env.wire + env.tag)
            fields = [env.packet.activity, env.packet.client, env.packet.record_id, env.packet.nonce,
                      env.packet.base_model_hash, env.packet.layout, env.packet.codec, env.packet.codec_spec_sha256]
            metadata = 16 + sum(2 + len(v.encode('utf8')) for v in fields) + 8
            check('application_envelope_exact_length_' + c, len(env.wire) == len(payload) + metadata)
            packet_sizes[c] = {'payload_bytes': len(payload), 'durint02_metadata_bytes': metadata,
                               'wire_bytes': len(env.wire), 'hmac_sm3_tag_bytes': len(env.tag),
                               'application_wire_plus_tag_bytes': len(env.wire) + len(env.tag),
                               'saved_envelope_bytes': envelope_path(out, c).stat().st_size}
            old = counter['encode_calls']
            def forbidden_encoder(_):
                raise AssertionError('retry must not invoke encoder')
            retry_codec = pr.Codec(codec.name, codec.spec_sha256, forbidden_encoder, codec.decode)
            retry = client(c).prepare(round_id=1, record_id='record-' + c, nonce='nonce-' + c,
                                     base_model_hash=base, codec=retry_codec, update=[9999.] * P)
            check('pending_reopen_exact_bytes_no_reencode_' + c, retry == env and counter['encode_calls'] == old)
            rejects('pending_spec_drift_' + c,
                    lambda c=c: client(c).prepare(round_id=1, record_id='record-' + c, nonce='nonce-' + c,
                      base_model_hash=base, codec=pr.Codec(codec.name, '0'*64, forbidden_encoder, codec.decode),
                      update=[9999.] * P), pr.StateError, decode_delta=0, encode_delta=0)
        write_json(out / 'layout.json', {'layout_sha256': layout, **layout_doc})
        result.update(packet_sizes=packet_sizes, codec_spec_sha256=codec.spec_sha256, layout_sha256=layout,
                      model_parameter_count=P, q_sha256_f64=vector_sha(q), base_zero_sm3=base0,
                      synthetic_updates={c: {'rule': 'q + deterministic dyadic coordinate perturbation',
                        'sha256_f64': vector_sha(synthetic_update(q,c)),
                        'rnext_sha256_f64': vector_sha(synthetic_update(q,c)-q),
                        'rnext_nonzero_count': int(np.count_nonzero(synthetic_update(q,c)-q))} for c in WEIGHTS})
        check('encoder_called_once_per_original_client', counter['encode_calls'] == 2)
    elif name == 'register':
        for c, env in envs.items():
            check('registered_' + c, server.register(env).state == pr.REGISTERED)
            check('registered_no_EF_commit_' + c, not client(c).commit_if_incorporated('record-' + c, server.query))
            check('registered_generation_zero_' + c, client(c).snapshot()['generation'] == 0)
        check('register_does_not_decode', counter['decode_calls'] == 0)
    elif name == 'accept':
        for c, env in envs.items():
            check('accepted_' + c, server.accept(env).state == pr.ACCEPTED)
            check('accepted_no_EF_commit_' + c, not client(c).commit_if_incorporated('record-' + c, server.query))
            equal('accepted_EF_still_zero_' + c, client(c).snapshot()['residual'], np.zeros(P))
        check('one_real_decode_per_accept', counter['decode_calls'] == 2)
        equal('accepted_not_incorporated_model_zero', server.snapshot(ACTIVITY)['model'], np.zeros(P))
    elif name == 'crash_before_sql_commit':
        def hard_crash():
            os._exit(91)
        server.finalize(ACTIVITY, 1, after_model_write=hard_crash)
        raise AssertionError('crash callback not invoked')
    elif name == 'recover_rollback':
        snap = server.snapshot(ACTIVITY)
        check('process_crash_atomic_rollback_count', snap['finalized_count'] == 0)
        equal('process_crash_atomic_model_rollback', snap['model'], np.zeros(P))
        equal('process_crash_atomic_accumulator_rollback', snap['accumulator'], np.zeros(P))
        for c in WEIGHTS:
            check('crash_keeps_accepted_' + c, server.query('record-' + c).state == pr.ACCEPTED)
            check('crash_keeps_pending_' + c, client(c).pending() == envs[c])
            check('crash_keeps_generation_zero_' + c, client(c).snapshot()['generation'] == 0)
    elif name == 'finalize_then_lose_ack':
        got = server.finalize(ACTIVITY, 1)
        assert got['newly_finalized']
        os._exit(92)  # Commit succeeded, response is deliberately lost.
    elif name == 'recover_finalized':
        expected = np.array([math.fsum([WEIGHTS['c1']*float(x), WEIGHTS['c2']*float(x)]) for x in q])
        snap = server.snapshot(ACTIVITY)
        equal('fullshape_weighted_model_exact', snap['model'], expected)
        equal('fullshape_weighted_accumulator_exact', snap['accumulator'], expected)
        check('finalized_count_one_after_lost_ack', snap['finalized_count'] == 1)
        check('finalize_retry_is_not_new', not server.finalize(ACTIVITY, 1)['newly_finalized'])
        offset = 0
        with torch.no_grad():
            for value in model.parameters():
                n = value.numel()
                value.copy_(torch.tensor(snap['model'][offset:offset+n], dtype=torch.float32).view_as(value))
                offset += n
        app_vector = torch.cat([v.detach().flatten() for v in model.parameters()]).numpy()
        equal('same_layout_zero_SmallCNN_receives_q_exactly', app_vector, q)
        for c in WEIGHTS:
            check('incorporated_' + c, server.query('record-' + c).state == pr.INCORPORATED)
            check('ack_loss_client_generation_still_zero_' + c, client(c).snapshot()['generation'] == 0)
            check('ack_loss_client_pending_bytes_survive_' + c, client(c).pending() == envs[c])
        result.update(aggregate_model_sha256_f64=vector_sha(snap['model']),
                      expected_model_sha256_f64=vector_sha(expected),
                      max_abs_aggregate_error=float(np.max(np.abs(np.asarray(snap['model'])-expected))),
                      model_storage_numeric_semantics='v2 SQLite vectors FP64; reconstructed SmallCNN parameters FP32',
                      weight_oracle_limitation='both clients replay identical q; tests full-shape fixed-roster path, not differential weights on unequal updates')
    elif name == 'client_commit_then_lose_ack':
        assert client('c1').commit_if_incorporated('record-c1', server.query)
        os._exit(93)
    elif name == 'recover_clients_and_replay':
        check('c1_saved_commit_survived_process_exit', client('c1').snapshot()['generation'] == 1)
        check('c1_commit_retry_not_new', not client('c1').commit_if_incorporated('record-c1', server.query))
        check('c2_commit_after_server_ack_loss', client('c2').commit_if_incorporated('record-c2', server.query))
        before = counter['decode_calls']
        for repeat in range(3):
            for c, env in envs.items():
                check('replay_register_' + c + '_' + str(repeat), server.register(env).state == pr.INCORPORATED)
                check('replay_accept_' + c + '_' + str(repeat), server.accept(env).state == pr.INCORPORATED)
                check('EF_commit_idempotent_' + c + '_' + str(repeat), not client(c).commit_if_incorporated('record-'+c,server.query))
            check('finalize_replay_' + str(repeat), not server.finalize(ACTIVITY,1)['newly_finalized'])
        check('incorporated_replay_never_decodes_again', counter['decode_calls'] == before)
        for c in WEIGHTS:
            snap = client(c).snapshot()
            check('exactly_one_EF_generation_' + c, snap['generation'] == 1)
            equal('committed_residual_exact_' + c, snap['residual'], synthetic_update(q,c)-q)
            check('no_outstanding_pending_' + c, client(c).pending() is None)
            rejects('wrong_client_layout_reopen_' + c, lambda c=c: client(c,'0'*64), pr.StateError,decode_delta=0)
        check('exactly_one_server_finalization',server.snapshot(ACTIVITY)['finalized_count']==1)
        result['client_residual_sha256_f64']={c:vector_sha(client(c).snapshot()['residual']) for c in WEIGHTS}
    elif name == 'tamper_and_context':
        original = envs['c1']
        details = []
        packet = original.packet
        payload_start = len(original.wire)-len(packet.payload)
        offsets = {'payload_magic':0,'payload_version':8,'payload_numel':12,'payload_reserved':24,
                   'block_bitwidth':64,'block_scale_byte0':65,'block_scale_byte1':66,
                   'block_scale_byte2':67,'block_scale_byte3':68,'packed_code_byte0':69,
                   'packed_code_byte1':70,'packed_code_last':len(payload)-1}
        for label, index in offsets.items():
            changed=bytearray(original.wire); changed[payload_start+index]^=1
            detail=rejects('unsigned_single_byte_'+label,lambda changed=changed:server.register(pr.Envelope(bytes(changed),original.tag)),pr.AuthenticationError,decode_delta=0)
            details.append({'kind':'one-byte payload tamper with original tag','field':label,'payload_byte_offset':index,**detail})
        tag=bytearray(original.tag);tag[0]^=1
        details.append({'kind':'one-byte tag tamper','tag_byte_offset':0,
          **rejects('tag_byte_tamper',lambda:server.register(pr.Envelope(original.wire,bytes(tag))),pr.AuthenticationError,decode_delta=0)})
        offset=16
        fieldnames=['activity','client','record_id','nonce','base_model_hash','layout','codec','codec_spec_sha256']
        for field in fieldnames:
            n=struct.unpack_from('<H',original.wire,offset)[0];offset+=2
            changed=bytearray(original.wire)
            changed[offset]=ord('0') if changed[offset]!=ord('0') else ord('1')
            details.append({'kind':'one-byte context tamper with original tag','field':field,'wire_byte_offset':offset,
              **rejects('unsigned_meta_'+field,lambda changed=changed:server.register(pr.Envelope(bytes(changed),original.tag)),pr.AuthenticationError,decode_delta=0)})
            offset+=n
        base=server.open_round(ACTIVITY,2)
        def signed(label, **kw):
            p=replace(packet,round_id=2,record_id='negative-'+label,nonce='negative-'+label,base_model_hash=base,**kw)
            return pr.Envelope.sign(p,keys['c1'])
        context_cases={'base_model_hash':{'base_model_hash':('0' if base[0]!='0' else '1')+base[1:]},
                       'layout_hash':{'layout':'0'*64}, 'codec_name_version':{'codec':'frozen-mixed-precision-v2'},
                       'codec_spec_sha256':{'codec_spec_sha256':'0'*64}}
        for label, kw in context_cases.items():
            # Avoid duplicate keyword for the deliberately changed base.
            if label=='base_model_hash':
                bad=pr.Envelope.sign(replace(packet,round_id=2,record_id='negative-'+label,nonce='negative-'+label,**kw),keys['c1'])
            else:bad=signed(label,**kw)
            server.register(bad)
            details.append({'kind':'validly authenticated context mismatch','field':label,
              **rejects('signed_context_'+label,lambda bad=bad:server.accept(bad),pr.StateError,decode_delta=0)})
            check('signed_context_stays_REGISTERED_'+label,server.query(bad.packet.record_id).state==pr.REGISTERED)
        structural={
          'negative_scale':(68,payload[68]|128),
          'unused_all_ones_code':(69,payload[69]|7),
          'invalid_bitwidth':(64,5),
          'reserved_nonzero':(24,1),
          'codec_wire_version':(8,2),
        }
        check('first_real_scale_positive',struct.unpack_from('<f',payload,65)[0]>0)
        for label,(index,value) in structural.items():
            altered=bytearray(payload);altered[index]=value
            check('structural_case_changes_exactly_one_byte_'+label,sum(a!=b for a,b in zip(payload,altered))==1)
            bad=signed(label,payload=bytes(altered));server.register(bad)
            details.append({'kind':'valid tag; structural decoder rejection','field':label,'payload_byte_offset':index,
              **rejects('signed_bad_payload_'+label,lambda bad=bad:server.accept(bad),(ValueError,pr.ProtocolError),decode_delta=1)})
            check('decoder_rejection_stays_REGISTERED_'+label,server.query(bad.packet.record_id).state==pr.REGISTERED)
        drift_codec=pr.Codec(codec.name,'0'*64,codec.encode,codec.decode)
        drift=pr.LocalServer(out/'server.sqlite',keys,{codec.name:drift_codec})
        details.append({'kind':'reopened server registry spec drift before idempotent accept',
          **rejects('registry_spec_drift_on_incorporated_packet',lambda:drift.accept(original),pr.StateError,decode_delta=0)})
        wire=bytearray(original.wire);wire[7]=ord('3')
        details.append({'kind':'unsupported application envelope version',
          **rejects('application_version_rejected',lambda:server.register(pr.Envelope(bytes(wire),original.tag)),pr.ProtocolError,decode_delta=0)})
        equal('tamper_leaves_full_model_unchanged',server.snapshot(ACTIVITY)['model'],q)
        check('tamper_leaves_finalized_count_one',server.snapshot(ACTIVITY)['finalized_count']==1)
        for c in WEIGHTS:
            equal('tamper_leaves_EF_unchanged_'+c,client(c).snapshot()['residual'],synthetic_update(q,c)-q)
            check('tamper_leaves_EF_generation_one_'+c,client(c).snapshot()['generation']==1)
        result['rejections']=details
    else:
        raise ValueError('unknown phase')
    result['check_count']=len(checks)
    write_json(out/('phase_'+name+'.json'),result)
    print(json.dumps({'phase':name,'checks':len(checks),'pass':True,'pid':os.getpid()}),flush=True)


def protected_paths():
    return sorted(set(list(V2.glob('*.py'))+list((V2/'specs').glob('*.json'))+[
      FROZEN_CODEC,MODEL_SOURCE,WORKSPACE/'work/balanced_quant/run.py',WORKSPACE/'work/balanced_quant/PLAN.md',
      TRAIN_ROOT/'config.json',TRAIN_ROOT/'first_payload.bin']))


def driver(out):
    began=time.perf_counter()
    out.mkdir(parents=True,exist_ok=False)
    private=out.parent/(out.name+'_private_test_keys')
    private.mkdir(exist_ok=False,mode=0o700)
    keypath=private/'keys.json'
    descriptor=os.open(keypath,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(descriptor,'w') as handle:
        json.dump({c:secrets.token_bytes(32).hex() for c in WEIGHTS},handle)
    before={str(p.relative_to(WORKSPACE)):sha(p) for p in protected_paths()}
    write_json(out/'protected_before.json',before)
    phases=['prepare','register','accept','crash_before_sql_commit','recover_rollback',
            'finalize_then_lose_ack','recover_finalized','client_commit_then_lose_ack',
            'recover_clients_and_replay','tamper_and_context']
    expected_exit={'crash_before_sql_commit':91,'finalize_then_lose_ack':92,'client_commit_then_lose_ack':93}
    execution=[]
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1')
    for name in phases:
        t=time.perf_counter()
        got=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--output-dir',str(out),'--phase',name],
                           capture_output=True,text=True,env=env,timeout=120)
        (out/('process_'+name+'.log')).write_text(got.stdout+got.stderr)
        execution.append({'phase':name,'returncode':got.returncode,'expected_returncode':expected_exit.get(name,0),
                          'seconds':time.perf_counter()-t})
        print(json.dumps(execution[-1]),flush=True)
        if got.returncode!=expected_exit.get(name,0):
            raise RuntimeError('integration phase failed: '+name+'; inspect its local log')
    after={str(p.relative_to(WORKSPACE)):sha(p) for p in protected_paths()}
    assert before==after,'protected inputs changed'
    snapshots={p.stem[6:]:json.loads(p.read_text()) for p in sorted(out.glob('phase_*.json'))}
    allchecks=[x for p in snapshots.values() for x in p['checks']]
    assert all(x['pass'] for x in allchecks)
    dbs=[]
    for p in sorted(out.glob('*.sqlite')):
        conn=sqlite3.connect(p)
        integrity=conn.execute('PRAGMA integrity_check').fetchall()
        checkpoint=conn.execute('PRAGMA wal_checkpoint(TRUNCATE)').fetchall()
        page_size=conn.execute('PRAGMA page_size').fetchone()[0]
        page_count=conn.execute('PRAGMA page_count').fetchone()[0]
        tables={row[0]:conn.execute('SELECT count(*) FROM '+row[0]).fetchone()[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        conn.close()
        assert integrity==[('ok',)]
        dbs.append({'path':str(p.relative_to(WORKSPACE)),'bytes_after_wal_checkpoint':p.stat().st_size,
                    'page_size':page_size,'page_count':page_count,'integrity_check':'ok','tables':tables,
                    'wal_bytes':Path(str(p)+'-wal').stat().st_size if Path(str(p)+'-wal').exists() else 0,
                    'shm_bytes':Path(str(p)+'-shm').stat().st_size if Path(str(p)+'-shm').exists() else 0,
                    'wal_checkpoint_result':checkpoint})
    keymode=oct(keypath.stat().st_mode&0o777)
    assert keymode=='0o600'
    assert not any('private_test_keys' in str(p) or p.name=='keys.json' for p in out.rglob('*'))
    prepare=snapshots['prepare']
    report={'status':'PASS','scope':'simulation_only','review_status':'LOCAL_CHECK_ONLY / REVIEW_UNAVAILABLE',
      'claim':'full-shape v2 protocol integration around a real historical training-generated first wire; NOT real FL training end-to-end',
      'real_input':{'path':str((TRAIN_ROOT/'first_payload.bin').relative_to(WORKSPACE)),
                   'sha256':sha(TRAIN_ROOT/'first_payload.bin'),'payload_bytes':54067,'decoded_dimension':P,
                   'provenance':'existing fedlc_uniform3 development artifact; no historical delta/checkpoint loaded',
                   'codec_source_sha256':sha(FROZEN_CODEC),'codec_spec_sha256':prepare['codec_spec_sha256'],
                   'training_config_signature':json.loads((TRAIN_ROOT/'config.json').read_text())['signature']},
      'synthetic_boundary':{'two_clients_both_replay_identical_real_wire':True,'weights':WEIGHTS,
          'model_initialization':'all-zero SmallCNN; exact source class AST, no dataset-module import, no forward or backward',
          'update_rule':'c1 q+((j%11)-5)*2^-12; c2 q+((j%13)-6)*2^-11; initial committed EF residual zero',
          'encoder':'test-only replay stub returns frozen first wire for both synthetic effective updates; no recompression/ML validity claim',
          'weight_oracle_limitation':snapshots['recover_finalized']['weight_oracle_limitation']},
      'checks':{'pass':len(allchecks),'fail':0,'by_phase':{name:p['check_count'] for name,p in snapshots.items()},
                'distinct_normal_subprocess_pids':len({p['pid'] for p in snapshots.values()}),
                'expected_hard_process_exits':expected_exit,
                'tamper_rejection_cases':len(snapshots['tamper_and_context']['rejections']),
                'max_abs_aggregate_error':snapshots['recover_finalized']['max_abs_aggregate_error'],
                'sqlite_integrity_ok':len(dbs),'protected_inputs_unchanged':len(before)},
      'packet_sizes':prepare['packet_sizes'],'layout':{'sha256':prepare['layout_sha256'],'manifest':'layout.json'},
      'sqlite_files':dbs,'sqlite_total_bytes':sum(d['bytes_after_wal_checkpoint']+d['wal_bytes']+d['shm_bytes'] for d in dbs),
      'private_key_handling':{'key_count':2,'key_bytes_each':32,'random_generation':'secrets.token_bytes',
          'directory_outside_result_tree':str(private.relative_to(WORKSPACE)),'keyfile_count':1,'keyfile_mode':keymode,
          'private_key_files_in_result_tree':0,'values_printed':False},
      'source_hashes_before':before,'source_hashes_after':after,'integration_source_sha256':sha(__file__),
      'phases':execution,'phase_evidence':snapshots,
      'security_boundaries':['Trusted key-holding server, SQLite state and status-query/Receipt remain assumptions; HMAC is not public verifiability or poisoning resistance.',
        'Codec name and pinned spec/source identify intended semantics; they do not prove actual decoder execution. Trusted matching decode implementation remains required.',
        'Single-host SQLite registration and atomicity only; no actual blockchain, consensus, network authentication channel, distributed atomicity or smart-contract deployment.',
        'Unauthenticated raw payload corruption is rejected by HMAC; validly tagged structurally invalid payloads rejected by decoder. A key holder can still sign semantically valid malicious updates.'],
      'resources':{'wall_seconds':time.perf_counter()-began,'python':sys.executable,'numpy':np.__version__,'torch':torch.__version__,
                   'official_data_loaded':False,'ml_training_performed':False,'network_used':False}}
    write_json(out/'RESULT.json',report)
    write_json(HERE/'RESULT.json',report)
    text=(f"本地完整维度集成：{len(allchecks)}/{len(allchecks)}检查通过，simulation_only；仅真实训练生成首包接入，非真实联邦训练端到端。\n\n"
      f"冻结fedlc_uniform3首包54067B，真实解码136874维；核对codec源码及spec固定哈希，以原SmallCNN布局构造全零模型。两客户端均重放同包q，更新人为设q＋不同确定扰动，初始EF为零；并未复现历史delta或重新编码。固定权重0.25/0.75聚合与独立预期逐位一致，误差0；因q相同，不区分不同更新上的权重正确性。\n\n"
      f"REGISTERED/ACCEPTED均不提交EF；INCORPORATED后只提交一次。实际子进程验证SQL提交前崩溃回滚、服务端提交后丢ack、客户端提交后丢ack及重开；pending逐字节重传不调用encoder，3次重放不再解码/累计。{report['checks']['tamper_rejection_cases']}项载荷scale/code/元数据/tag及base/layout/codec版本、spec漂移拒绝，模型/EF保持原值。\n\n"
      f"每客户端应用封装={prepare['packet_sizes']['c1']['wire_bytes']}B wire＋32B HMAC-SM3={prepare['packet_sizes']['c1']['application_wire_plus_tag_bytes']}B；其中原压缩包54067B。3个SQLite文件实测合计{report['sqlite_total_bytes']}B，完整性均ok；逐库大小见JSON。{len(before)}个受保护输入前后哈希一致。随机测试密钥600权限、独立private_test_keys目录，结果树无key。\n\n"
      "边界沿用v2：信任持钥服务器、数据库、查询/Receipt及真实decoder；spec/source相同不证明执行，HMAC不证明更新诚实。没有真实链、网络或分布式原子性；无数据/TEST读取、无新训练。LOCAL_CHECK_ONLY / REVIEW_UNAVAILABLE。")
    assert len(text)<=900,len(text)
    (HERE/'REPORT.md').write_text(text+'\n')
    print(json.dumps({'status':'PASS','checks':len(allchecks),'tamper_cases':report['checks']['tamper_rejection_cases'],
                      'sqlite_total_bytes':report['sqlite_total_bytes'],'report_characters':len(text)}),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=HERE/'results')
    parser.add_argument('--phase')
    args=parser.parse_args()
    if args.phase:phase(args.output_dir.resolve(),args.phase)
    else:driver(args.output_dir.resolve())


if __name__=='__main__':main()
