from pathlib import Path
import json,csv,hashlib,datetime
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def stat(x):
 x=np.asarray(x,dtype=float);return {'n':len(x),'mean':float(x.mean()),'median':float(np.median(x)),'min':float(x.min()),'max':float(x.max()),'sample_sd':float(x.std(ddof=1)) if len(x)>1 else None,'values':x.tolist()}
rows=[];receiptrows=[];inputs={};primaries=[]
for n,cnt,rnd in [('c2_r2_measured',2,2),('c5_r3_measured',5,3)]:
 path=P/'e3'/n/'result.json';r=json.loads(path.read_text());inputs[str(path)]=sha(path);assert r['status']=='PASS'
 for f,h in r['source_and_input_sha256'].items():assert sha(f)==h,f
 clients=[c for rd in r['rounds'] for c in rd['clients']];assert len(clients)==cnt*rnd;assert r['chain']['bridge_stats']['record_count']==cnt*rnd
 assert all(rd['protocol_direct_model_max_abs_error']==rd['protocol_direct_residual_max_abs_error']==0 for rd in r['rounds'])
 assert not r['audit']['forbidden_test_open_attempts'];first={}
 for v in r['receipt_verifications']:
  if v['state']=='INCORPORATED' and v['caller']=='commit':first.setdefault(v['record_id'],v)
 assert len(first)==cnt*rnd
 for rd in r['rounds']:
  for c in rd['clients']:
   chain=c['chain'];key=f"r{rd['round']}-c{c['client']}";verify=first[key]
   assert c['EF_generation']==rd['round'];assert chain['status']==1 and chain['canonical'] and chain['confirmation_depth']>=2
   receiptrows.append({'configuration':n,'round':rd['round'],'client':c['client'],'registration':chain,'incorporation_receipt':verify['raw_receipt'],'verification_ms':verify['verification_ms'],'payload_bytes':c['payload_bytes'],'application_envelope_and_tag_bytes':c['application_envelope_and_tag_bytes']})
 metrics=r['network']['bridge_HTTP_requests'];register=[m['seconds']*1000 for m in metrics if m['label']=='chain_register'];readback=[m['seconds']*1000 for m in metrics if m['label']=='chain_recovery_readback' and m['status']==200]
 vals=[v['verification_ms'] for v in first.values()];gas=[c['chain']['gas_used'] for c in clients]
 row={'configuration':n,'clients':cnt,'rounds':rnd,'registrations':len(clients),'gas_registration_total':sum(gas),'gas_registration_per_upload':stat(gas),'gas_deployment':r['chain']['bridge_stats']['deployment']['gas_used'],'incorporation_verification_ms':stat(vals),'registration_wall_ms':stat(register),'ledger_readback_wall_ms':stat(readback),'payload_bytes_each':54067,'message_bytes_each':54408,'context_and_tag_bytes_each':341,'payload_bytes_total':r['fresh_training_payload_bytes'],'message_bytes_total':r['fresh_training_envelope_and_tag_bytes'],'calldata_bytes':stat([c['chain']['calldata_bytes'] for c in clients]),'signed_transaction_bytes':stat([c['chain']['signed_transaction_bytes'] for c in clients]),'serialized_RPC_receipt_bytes':stat([c['chain']['receipt_json_bytes'] for c in clients]),'raw_result':str(path),'all_protocol_direct_model_and_EF_equal':True,'fresh_gradients':r['total_gradient_evaluations'],'checks_passed':r['checks_passed'],'client_sizes':r['profile']['client_sizes'],'elapsed_seconds':r['elapsed_seconds'],'recovery_summary':{'drop_after_chain_before_local_register':'PASS','drop_after_finalize_before_reply':'PASS','service_database_reopen_exact_retry':'PASS','duplicate_upload_finalize_EF_commit':'PASS'},'timing_boundaries':r['timing_boundaries']}
 rows.append(row);primaries.append(vals)
with (P/'e3_receipts.jsonl').open('w') as f:
 for row in receiptrows:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
columns=['clients','rounds','registrations','payload_bytes_each','message_bytes_each','payload_bytes_total','message_bytes_total','gas_registration_total','gas_deployment','mean_registration_gas','mean_receipt_verify_ms','median_receipt_verify_ms','mean_ledger_readback_ms']
with (P/'e3_cost_table.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=columns);w.writeheader()
 for row in rows:
  w.writerow({**{c:row[c] for c in columns if c in row},'mean_registration_gas':row['gas_registration_per_upload']['mean'],'mean_receipt_verify_ms':row['incorporation_verification_ms']['mean'],'median_receipt_verify_ms':row['incorporation_verification_ms']['median'],'mean_ledger_readback_ms':row['ledger_readback_wall_ms']['mean']})
(P/'tables').mkdir(exist_ok=True)
a,b=rows
tex=r'''\begin{table}[!htb]\centering\small
\caption{Anchoring preserves the 54,408 B application message as the activity grows. Registration gas excludes contract deployment; receipt verification measures local authentication and bound chain fields.}\label{tab:chaincost}
\setlength{\tabcolsep}{5pt}
\begin{tabular}{lrr}\toprule
Metric & $2\times2$ & $5\times3$\\\midrule
Fresh uploads & 4 & 15\\
Codec payload / upload (B) & 54,067 & 54,067\\
Authenticated message / upload (B) & 54,408 & 54,408\\
Total fresh message bytes & 217,632 & 816,120\\
Registration gas, total & GAS0 & GAS1\\
Registration gas / upload, mean & GM0 & GM1\\
Receipt verification, mean (ms) & V0 & V1\\
Ledger readback, mean (ms) & R0 & R1\\\bottomrule
\end{tabular}
\end{table}
'''
for j,row in enumerate(rows):
 for key,value in [('GAS',f"{row['gas_registration_total']:,}"),('GM',f"{row['gas_registration_per_upload']['mean']:,.1f}"),('DEP',f"{row['gas_deployment']:,}"),('V',f"{row['incorporation_verification_ms']['mean']:.3f}"),('R',f"{row['ledger_readback_wall_ms']['mean']:.3f}")]:tex=tex.replace(key+str(j),value)
(P/'tables/table3_chain_cost.tex').write_text(tex)
with plt.rc_context({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False}):
 fig,axes=plt.subplots(1,3,figsize=(7.3,3.2),layout='constrained');x=np.array([2,5]);tick=['2 clients\n2 rounds','5 clients\n3 rounds']
 axes[0].plot(x,[r['gas_registration_total']/1e6 for r in rows],marker='o',color='#0072B2');axes[0].set(ylabel='Total registration gas (million)',title='(a) Registration');axes[0].set_ylim(bottom=0)
 for j,v in enumerate(primaries):axes[1].scatter(np.repeat(x[j],len(v)),v,s=17,marker='x',color='#777777')
 axes[1].plot(x,[r['incorporation_verification_ms']['mean'] for r in rows],marker='s',color='#A64B00',label='Mean / individual receipts');axes[1].set(ylabel='Local receipt verification (ms)',title='(b) Verification');axes[1].set_ylim(bottom=0)
 axes[2].plot(x,[r['message_bytes_total']/1000 for r in rows],marker='D',color='#226A3C');axes[2].set(ylabel='Total fresh message bytes (kB)',title='(c) Message traffic');axes[2].set_ylim(bottom=0)
 for ax in axes:ax.set_xticks(x,tick);ax.grid(axis='y',alpha=.15);ax.set_xlim(1.5,5.5)
 fig.savefig(P/'figures/figure4_chain_cost.pdf');fig.savefig(P/'figures/figure4_chain_cost.png',dpi=200);plt.close(fig)
result={'status':'COMPLETE','created_at':datetime.datetime.now().astimezone().isoformat(),'settings':{'chain':'local EVM development chain','clients_and_rounds':[[2,2],[5,3]],'runs_per_setting':1,'primary_timing_unit':'first successful commit-path INCORPORATED receipt verification per unique client-round','timing_scope':'local JSON/HMAC/context/embedded-chain validation; excludes retrieval, waiting and SQLite commit','no_general_linear_scalability_claim':True},'rows':rows,'source_and_input_sha256':inputs,'source_lock_sha256':sha(P/'e3/SOURCE_LOCK.json'),'limitations':['Two configurations change clients and rounds together; not an isolated client-count scaling estimate.','Timing samples share one host/process activity, not independent deployment replications.','Development-chain gas is execution gas, not a monetary fee.','Protocol/direct equivalence uses FP32 local CNN/codec and FP64 protocol state, not the main MPS trajectory.'],'output_sha256':{str(f):sha(f) for f in [P/'e3_receipts.jsonl',P/'e3_cost_table.csv',P/'tables/table3_chain_cost.tex',P/'figures/figure4_chain_cost.pdf']}}
write(P/'e3_chain_cost.json',result)
print(json.dumps({'status':'COMPLETE','configurations':[{'clients':r['clients'],'rounds':r['rounds'],'gas':r['gas_registration_total'],'verify_ms':r['incorporation_verification_ms']['mean'],'readback_ms':r['ledger_readback_wall_ms']['mean'],'bytes':r['message_bytes_total']} for r in rows]}))
