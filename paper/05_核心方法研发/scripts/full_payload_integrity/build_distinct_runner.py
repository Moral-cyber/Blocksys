from pathlib import Path
root=Path(__file__).resolve().parent
source=(root/'run_integration.py').read_text()
s=source

def change(old,new):
 global s
 assert old in s,old[:140]
 s=s.replace(old,new)

change('Only the 54067-byte first wire comes from a historical real training run.\nBoth client updates and all server/EF states below are synthetic fixtures.\nThe encoder deliberately REPLAYS that wire; it does not quantize these updates.',
'''Two distinct 54067-byte first wires come from historical CE and FedLC runs.
Both client updates and all server/EF states below are synthetic fixtures.
Each encoder deliberately REPLAYS its assigned wire; it does not quantize these updates.
This independent runner preserves the original single-wire runner and results.''')
change("TRAIN_ROOT = WORKSPACE / 'work/balanced_quant/development_p20261193_s20261201/fedlc_uniform3'", """TRAIN_ROOTS = {
    'c1': WORKSPACE / 'work/balanced_quant/development_p20261193_s20261201/ce_uniform3',
    'c2': WORKSPACE / 'work/balanced_quant/development_p20261193_s20261201/fedlc_uniform3',
}""")
start=s.index("    source_conf = json.loads((TRAIN_ROOT / 'config.json').read_text())")
end=s.index('    # Execute ONLY the standalone class AST;',start)
s=s[:start]+'''    payloads, qs = {}, {}
    for c, source_root in TRAIN_ROOTS.items():
        source_conf = json.loads((source_root / 'config.json').read_text())
        assert source_conf['codec_sha256'] == sha(FROZEN_CODEC)
        assert source_conf['source_sha256'] == sha(WORKSPACE / 'work/balanced_quant/run.py')
        assert source_conf['plan_sha256'] == sha(WORKSPACE / 'work/balanced_quant/PLAN.md')
        assert source_conf['parameter_count'] == P
        payloads[c] = (source_root / 'first_payload.bin').read_bytes()
        assert len(payloads[c]) == 54067
        qs[c] = codec_module.decode(payloads[c]).numpy().astype(np.float64)
        assert qs[c].shape == (P,) and np.isfinite(qs[c]).all()
    assert payloads['c1'] != payloads['c2'], 'input wires must differ'
    assert not np.array_equal(qs['c1'], qs['c2']), 'decoded client updates must differ'
    wire, q = payloads['c1'], qs['c1']
'''+s[end:]
change('''    def replay_encoder(values):
        assert len(values) == P
        counter['encode_calls'] += 1
        return wire''','''    def replay_encoder_for(c):
        def replay_encoder(values):
            assert len(values) == P
            counter['encode_calls'] += 1
            return payloads[c]
        return replay_encoder''')
change('''    codec = protocol.Codec(CODEC_NAME, protocol.frozen_spec_identity(CODEC_NAME), replay_encoder, real_decode)
    server =''','''    specid = protocol.frozen_spec_identity(CODEC_NAME)
    codecs = {c: protocol.Codec(CODEC_NAME, specid, replay_encoder_for(c), real_decode) for c in WEIGHTS}
    codec = codecs['c1']
    server =''')
change('return protocol, codec_module, wire, q, model, layout_doc, layout, keys, counter, codec, server, client',
       'return protocol, codec_module, wire, q, model, layout_doc, layout, keys, counter, codec, server, client, payloads, qs, codecs')
change('(pr, cm, payload, q, model, layout_doc, layout, keys, counter, codec, server, client) = initialize(out)',
       '(pr, cm, payload, q, model, layout_doc, layout, keys, counter, codec, server, client, payloads, qs, codecs) = initialize(out)')
change('synthetic_update(q, c)', 'synthetic_update(qs[c], c)')
change('synthetic_update(q,c)', 'synthetic_update(qs[c],c)')
# Restore function declaration altered by the preceding literal replacement.
change('def synthetic_update(qs[c], client):','def synthetic_update(q, client):') if 'def synthetic_update(qs[c], client):' in s else None
change('codec=codec, update=update.tolist())','codec=codecs[c], update=update.tolist())')
change("check('real_first_wire_' + c, env.packet.payload == payload)","check('real_first_wire_' + c, env.packet.payload == payloads[c])")
change("pr.bytes_vector(row['rnext']), update - q)","pr.bytes_vector(row['rnext']), update - qs[c])")
change("        packet_sizes = {}\n", """        packet_sizes = {}
        check('two_real_wire_hashes_differ', sha(TRAIN_ROOTS['c1']/'first_payload.bin') != sha(TRAIN_ROOTS['c2']/'first_payload.bin'))
        check('decoded_client_vectors_differ', not np.array_equal(qs['c1'], qs['c2']))
        check('weight_0p25_0p75_is_observable', np.max(np.abs(.25*qs['c1']+.75*qs['c2']-(qs['c1']+qs['c2'])/2))>0)
""")
change("q_sha256_f64=vector_sha(q)","q_sha256_f64_by_client={c:vector_sha(qs[c]) for c in WEIGHTS}")
change("synthetic_update(qs[c],c)-q", "synthetic_update(qs[c],c)-qs[c]")
change("expected = np.array([math.fsum([WEIGHTS['c1']*float(x), WEIGHTS['c2']*float(x)]) for x in q])",
       "expected = WEIGHTS['c1'] * qs['c1'] + WEIGHTS['c2'] * qs['c2']  # independent NumPy FP64 oracle")
change("equal('same_layout_zero_SmallCNN_receives_q_exactly', app_vector, q)",
       "equal('same_layout_SmallCNN_receives_FP32_rounded_weighted_oracle', app_vector, expected.astype(np.float32))")
change("weight_oracle_limitation='both clients replay identical q; tests full-shape fixed-roster path, not differential weights on unequal updates'",
       "weight_oracle_limitation='distinct decoded client vectors and non-equal weights; independent NumPy FP64 oracle; reconstructed application model uses separately checked FP32 rounding',\n                      unequal_q_coordinates=int(np.count_nonzero(qs['c1']!=qs['c2'])),\n                      equal_weight_wrong_oracle_max_abs_error=float(np.max(np.abs(expected-(qs['c1']+qs['c2'])/2)))")
change("equal('tamper_leaves_full_model_unchanged',server.snapshot(ACTIVITY)['model'],q)",
       "equal('tamper_leaves_full_model_unchanged',server.snapshot(ACTIVITY)['model'],WEIGHTS['c1']*qs['c1']+WEIGHTS['c2']*qs['c2'])")
change("TRAIN_ROOT/'config.json',TRAIN_ROOT/'first_payload.bin'", "*(p/name for p in TRAIN_ROOTS.values() for name in ('config.json','first_payload.bin')),\n      HERE/'run_integration.py'")
start=s.index("      'real_input':{")
end=s.index("      'synthetic_boundary':",start)
s=s[:start]+'''      'real_inputs':{c:{'path':str((source_root/'first_payload.bin').relative_to(WORKSPACE)),
                   'sha256':sha(source_root/'first_payload.bin'),'payload_bytes':54067,'decoded_dimension':P,
                   'provenance':'existing '+source_root.name+' development artifact; no historical delta/checkpoint loaded',
                   'codec_source_sha256':sha(FROZEN_CODEC),'codec_spec_sha256':prepare['codec_spec_sha256'],
                   'training_config_signature':json.loads((source_root/'config.json').read_text())['signature'],
                   'decoded_sha256_f64':prepare['q_sha256_f64_by_client'][c]} for c,source_root in TRAIN_ROOTS.items()},
'''+s[end:]
change("'two_clients_both_replay_identical_real_wire':True", "'two_clients_replay_distinct_real_wires':True")
change("'update_rule':'c1 q+((j%11)-5)*2^-12; c2 q+((j%13)-6)*2^-11; initial committed EF residual zero'", "'update_rule':'c1 q_CE+((j%11)-5)*2^-12; c2 q_FedLC+((j%13)-6)*2^-11; initial committed EF residual zero'")
change("'encoder':'test-only replay stub returns frozen first wire for both synthetic effective updates; no recompression/ML validity claim'", "'encoder':'client-specific test replay stubs return assigned distinct frozen real wires; no recompression/ML validity claim'")
change("'max_abs_aggregate_error':snapshots['recover_finalized']['max_abs_aggregate_error'],", "'max_abs_aggregate_error':snapshots['recover_finalized']['max_abs_aggregate_error'],\n                'unequal_q_coordinates':snapshots['recover_finalized']['unequal_q_coordinates'],\n                'equal_weight_wrong_oracle_max_abs_error':snapshots['recover_finalized']['equal_weight_wrong_oracle_max_abs_error'],")
change("'integration_source_sha256':sha(__file__),", "'integration_source_sha256':sha(__file__),\n      'preserved_v1_runner_sha256':sha(HERE/'run_integration.py'),")
change("write_json(HERE/'RESULT.json',report)","write_json(HERE/'RESULT_DISTINCT.json',report)")
start=s.index('    text=(f"本地完整维度集成：')
end=s.index('    assert len(text)<=900,len(text)',start)
s=s[:start]+'''    text=(f"不同真实首包的完整维度集成：{len(allchecks)}/{len(allchecks)}检查通过，simulation_only。保留原同包试验。\\n\\n"
      f"c1用CE uniform3、c2用FedLC uniform3既有首包，各54067B、解码136874维；两wire与解码哈希不同，{report['checks']['unequal_q_coordinates']}个坐标不同。固定权重0.25/0.75，以NumPy FP64独立计算0.25q_CE＋0.75q_FedLC；SQLite模型及累计量逐位一致，最大误差0。错误等权预期最大相差{report['checks']['equal_weight_wrong_oracle_max_abs_error']:.9g}。按原布局回填SmallCNN的FP32舍入也独立核对。\\n\\n"
      "初始模型/EF为零；两客户端更新人为设各自q＋不同确定扰动，encoder仅重放对应真实wire，不是历史训练delta或新量化。REGISTERED/ACCEPTED不提交EF，INCORPORATED后残差只提交一次。实际子进程验证SQL提交前崩溃回滚、服务端及客户端提交后丢ack重开；pending逐字节重传不重编码，3次重放不再次聚合或提交EF。\\n\\n"
      f"{report['checks']['tamper_rejection_cases']}项scale/code/tag/元数据及base/layout/codec/spec拒绝均通过。每客户端封装54373B wire＋32B HMAC-SM3＝54405B；3个SQLite文件实测合计{report['sqlite_total_bytes']}B，完整性均ok。{len(before)}个受保护输入前后哈希不变；随机测试key600权限、结果树无key。\\n\\n"
      "这是真实训练首包的合成协议集成，非真实联邦训练端到端。信任持钥服务器、数据库、查询/Receipt及decoder；spec/source身份不证明执行，HMAC不证明更新诚实。无真实链/网络/分布式原子性，无数据/TEST读取、无新训练。LOCAL_CHECK_ONLY / REVIEW_UNAVAILABLE。")
'''+s[end:]
change("(HERE/'REPORT.md').write_text(text+'\\n')", "(HERE/'REPORT_DISTINCT.md').write_text(text+'\\n')")
change("default=HERE/'results')", "default=HERE/'results_distinct')")
assert 'TRAIN_ROOT/' not in s and 'TRAIN_ROOT /' not in s
assert s.count('synthetic_update(q,') == 1  # declaration only
assert 'same q' not in s
(root/'run_distinct_integration.py').write_text(s)
compile(s,str(root/'run_distinct_integration.py'),'exec')
print('new_runner_written',len(s))
