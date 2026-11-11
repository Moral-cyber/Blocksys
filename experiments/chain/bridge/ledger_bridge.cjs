// ----------------------------------------------------------------------
// 论文原版桥接脚本（参考副本）
//
// 作用：启动 Ganache、编译并部署 UploadRegistry.sol、代理 JSON-RPC、提供 readback。
// 说明：这是论文实验使用的脚本；本目录的 x2_register_on_chain.mjs 是它的精简替代，
//       只做「读 uploads.jsonl → 部署 → 登记 → 读回比对」。
// 论文对应：§7.1–7.2。
// ----------------------------------------------------------------------

/* Local EVM contract bridge. Random Ganache accounts stay inside the process;
 * no private keys are printed or persisted by this script. JSON-RPC transport
 * uses real loopback HTTP. Two descendant blocks are an explicit development
 * confirmation rule, not a claim of distributed consensus finality. */
'use strict';
const fs = require('node:fs');
const path = require('node:path');
const http = require('node:http');
const crypto = require('node:crypto');
const {createRequire} = require('node:module');
const args = process.argv.slice(2);
function arg(name) { const i=args.indexOf(name); if(i<0||!args[i+1]) throw Error('Missing '+name); return args[i+1]; }
const runtime=path.resolve(arg('--runtime')), workdir=path.resolve(arg('--workdir'));
const dependency=createRequire(path.join(runtime,'package.json'));
const ganache=dependency('ganache'), solc=dependency('solc'), ethers=dependency('ethers');
const sourcePath=path.join(__dirname,'UploadRegistry.sol');
const sha=x=>crypto.createHash('sha256').update(x).digest('hex');
const id=x=>'0x'+sha(Buffer.from(x,'utf8'));
const zero='0x'+'0'.repeat(64), zeroAddress='0x'+'0'.repeat(40);
const counters={rpc_requests:0,rpc_request_body_bytes:0,rpc_response_body_bytes:0,
  bridge_requests:0,bridge_request_body_bytes:0,bridge_response_body_bytes:0,registrations:0};
let ganacheServer, proxy, api, provider, contract, deployment, contractAddress, chainId;
let gate=Promise.resolve();
const serialized=f=>{const next=gate.then(f,f);gate=next.catch(()=>{});return next;};
const cache=new Map();
const versions={ganache:dependency('ganache/package.json').version,solc:solc.version(),ethers:ethers.version,node:process.version};
const json=x=>JSON.stringify(x,(_,v)=>typeof v==='bigint'?v.toString():v);
const listen=s=>new Promise((resolve,reject)=>{s.once('error',reject);s.listen(0,'127.0.0.1',(error)=>error?reject(error):resolve(s.address().port));});
async function body(req){let total=0,chunks=[];for await(const b of req){total+=b.length;if(total>1024*1024)throw Error('request too large');chunks.push(b);}return Buffer.concat(chunks);}
function hex32(v){if(typeof v!=='string'||!/^(?:0x)?[0-9a-fA-F]{64}$/.test(v))throw Error('Expected32-bytehex');return '0x'+v.replace(/^0x/,'').toLowerCase();}
function normalize(d){
 if(typeof d.record_id!=='string'||!d.record_id||!Number.isSafeInteger(d.round_id)||d.round_id<1)throw Error('Invalid record/round');
 const out={record_id:d.record_id,round_id:d.round_id};
 for(const k of ['digest','activity_hash','client_hash','nonce_hash','base_model_hash','layout_hash','codec_spec_sha256'])out[k]=hex32(d[k]);
 return out;
}
function binding(d){return {activity:d.activity_hash,client:d.client_hash,round:d.round_id,nonce:d.nonce_hash,
 baseModel:d.base_model_hash,layout:d.layout_hash,codecSpec:d.codec_spec_sha256,digest:d.digest,uploader:zeroAddress,blockNumber:0};}
function fromBinding(record,row){return {record_id:record,activity_hash:row.activity,client_hash:row.client,
 round_id:Number(row.round),nonce_hash:row.nonce,base_model_hash:row.baseModel,layout_hash:row.layout,
 codec_spec_sha256:row.codecSpec,digest:row.digest};}
function exact(a,b){return Object.keys(a).every(k=>a[k]===b[k]);}
async function receiptFor(record){
 const row=await contract.get(id(record));if(row.uploader===zeroAddress)return null;
 const filter=contract.filters.Registered(id(record));
 const logs=await contract.queryFilter(filter,Number(row.blockNumber),Number(row.blockNumber));
 if(logs.length!==1)throw Error('Registration event identity mismatch');
 const receipt=await provider.getTransactionReceipt(logs[0].transactionHash);
 const canonical=await provider.getBlock(receipt.blockNumber);
 const head=Number(await provider.send('eth_blockNumber',[]));
 const rawHead=parseInt(String(head),10); // ethers returns a hex quantity from send
 const height=Number.isFinite(rawHead)?rawHead:head;
 const depth=height-receipt.blockNumber;
 if(receipt.status!==1||receipt.to.toLowerCase()!==contractAddress.toLowerCase()||
    canonical.hash!==receipt.blockHash||depth<2)throw Error('Unconfirmed/noncanonical registration');
 const data=fromBinding(record,row);
 return {...data,readback:data,tx_hash:receipt.hash,block_number:receipt.blockNumber,
   block_hash:receipt.blockHash,head_block_number:height,confirmation_depth:depth,
   confirmations:depth+1,status:receipt.status,canonical:true,chain_id:chainId,
   contract_address:contractAddress,uploader:row.uploader,gas_used:Number(receipt.gasUsed),
   transaction_count_delta:0};
}
async function register(input){
 const d=normalize(input),previous=await contract.get(id(d.record_id));
 if(previous.uploader!==zeroAddress){
  if(!exact(d,fromBinding(d.record_id,previous)))throw Error('record collision');
  return receiptFor(d.record_id);
 }
 const clientIndex=['c0','c1'].findIndex(c=>id(c)===d.client_hash);if(clientIndex<0)throw Error('unknown client');
 const signer=await provider.getSigner(clientIndex+1);
 const tx=await contract.connect(signer).register(id(d.record_id),binding(d));
 await tx.wait();
 await provider.send('evm_mine',[]);await provider.send('evm_mine',[]);
 const confirmed=await receiptFor(d.record_id);
 if(!exact(d,confirmed.readback))throw Error('On-chain readback mismatch');
 counters.registrations++;cache.set(d.record_id,confirmed);
 return {...confirmed,transaction_count_delta:1};
}
async function adversarial(){
 const first=[...cache.values()][0];if(!first)throw Error('Need existing real registration');
 const n=Number(await contract.recordCount()),base=normalize(first),checks=[];
 async function reverts(name,fn){try{await fn();}catch(e){if(e.code!=='CALL_EXCEPTION')throw e;checks.push(name);return;}throw Error('Expected EVM revert: '+name);}
 const correct=await provider.getSigner(1),wrong=await provider.getSigner(2);
 await reverts('wrong_sender_cannot_claim_c0',()=>contract.connect(wrong).register.staticCall(id('attack-sender'),binding({...base,record_id:'attack-sender',nonce_hash:id('attack-nonce'),round_id:999})));
 await reverts('same_record_changed_digest_reverts',()=>contract.connect(correct).register.staticCall(id(base.record_id),binding({...base,digest:id('different-digest')})));
 await reverts('same_nonce_new_record_reverts',()=>contract.connect(correct).register.staticCall(id('attack-nonce-record'),binding({...base,round_id:999})));
 await reverts('same_round_new_nonce_reverts',()=>contract.connect(correct).register.staticCall(id('attack-round-record'),binding({...base,nonce_hash:id('other-nonce')})));
 const before=await receiptFor(base.record_id);
 await contract.connect(correct).register.staticCall(id(base.record_id),binding(base));
 const after=await receiptFor(base.record_id);
 if(before.tx_hash!==after.tx_hash||Number(await contract.recordCount())!==n)throw Error('Adversarial call changed state');
 checks.push('exact_repeat_simulation_is_idempotent','reverts_leave_registry_unchanged');
 return {checks,checks_passed:checks.length,scope:'Real EVM eth_call reverts; no extra mined attack transactions',record_count:n};
}
async function stats(){return {...counters,chain_id:chainId,contract_address:contractAddress,
 record_count:Number(await contract.recordCount()),versions,deployment,
 source_sha256:sha(fs.readFileSync(sourcePath)),bridge_source_sha256:sha(fs.readFileSync(__filename)),
 confirmation_rule:'canonical receipt.status=1 and current_head - receipt.block_number >= 2; single-node manually mined development chain',
 counting_scope:'HTTP JSON-RPC request/response body bytes at loopback proxy; excludes HTTP headers, TCP/IP, peer gossip and consensus',
 chain_scope:'Ganache single-node local EVM development chain; not a public testnet or multi-validator finality benchmark'};}
async function shutdown(){await new Promise(r=>api.close(r));await new Promise(r=>proxy.close(r));await ganacheServer.close();}
async function main(){
 fs.mkdirSync(workdir,{recursive:true});
 const input={language:'Solidity',sources:{'UploadRegistry.sol':{content:fs.readFileSync(sourcePath,'utf8')}},
 settings:{optimizer:{enabled:true,runs:200},viaIR:true,evmVersion:'shanghai',outputSelection:{'*':{'*':['abi','evm.bytecode.object','evm.deployedBytecode.object']}}}};
 const compiled=JSON.parse(solc.compile(JSON.stringify(input)));
 const errors=(compiled.errors||[]).filter(e=>e.severity==='error');if(errors.length)throw Error(json(errors));
 const artifact=compiled.contracts['UploadRegistry.sol'].UploadRegistry;
 ganacheServer=ganache.server({logging:{quiet:true},chain:{chainId:20261104,hardfork:'shanghai'},
 wallet:{totalAccounts:3},database:{dbPath:path.join(workdir,'evm-db')},miner:{instamine:'eager'}});
 const ganachePort=await listen(ganacheServer);
 const upstream='http://127.0.0.1:'+ganachePort;
 proxy=http.createServer(async(req,res)=>{try{
  const raw=await body(req);counters.rpc_requests++;counters.rpc_request_body_bytes+=raw.length;
  const out=await fetch(upstream,{method:'POST',headers:{'content-type':'application/json'},body:raw});
  const bytes=Buffer.from(await out.arrayBuffer());counters.rpc_response_body_bytes+=bytes.length;
  res.writeHead(out.status,{'content-type':'application/json','content-length':bytes.length});res.end(bytes);
 }catch(e){res.writeHead(500);res.end(json({error:e.message}));}});
 const rpcPort=await listen(proxy),rpcUrl='http://127.0.0.1:'+rpcPort;
 provider=new ethers.JsonRpcProvider(rpcUrl,undefined,{cacheTimeout:-1,batchMaxCount:1,polling:true,pollingInterval:10});
 chainId=Number((await provider.getNetwork()).chainId);
 const deployer=await provider.getSigner(0),clients=[await provider.getSigner(1),await provider.getSigner(2)];
 const addresses=await Promise.all(clients.map(c=>c.getAddress()));
 const factory=new ethers.ContractFactory(artifact.abi,'0x'+artifact.evm.bytecode.object,deployer);
 contract=await factory.deploy(['c0','c1'].map(id),addresses);await contract.waitForDeployment();
 contractAddress=await contract.getAddress();const tx=contract.deploymentTransaction(),r=await tx.wait();
 deployment={tx_hash:r.hash,block_number:r.blockNumber,block_hash:r.blockHash,status:r.status,gas_used:Number(r.gasUsed),
  bytecode_sha256:sha(Buffer.from(artifact.evm.bytecode.object,'hex')),
  deployed_code_sha256:sha(Buffer.from((await provider.getCode(contractAddress)).slice(2),'hex')),
  abi_sha256:sha(Buffer.from(json(artifact.abi))),compiler_settings:input.settings};
 api=http.createServer((req,res)=>serialized(async()=>{try{
  const started=performance.now(),raw=await body(req);counters.bridge_requests++;counters.bridge_request_body_bytes+=raw.length;
  const u=new URL(req.url,'http://127.0.0.1');let result,status=200;
  if(req.method==='POST'&&u.pathname==='/register')result=await register(JSON.parse(raw));
  else if(req.method==='GET'&&u.pathname==='/record'){result=await receiptFor(u.searchParams.get('record_id'));if(!result){result={error:'not found'};status=404;}}
  else if(req.method==='GET'&&u.pathname==='/stats')result=await stats();
  else if(req.method==='POST'&&u.pathname==='/adversarial-checks')result=await adversarial();
  else if(req.method==='POST'&&u.pathname==='/shutdown'){result={status:'STOPPING'};setTimeout(()=>shutdown().then(()=>process.exit(0)),25);}
  else {status=404;result={error:'unknown route'};}
  result={...result,bridge_elapsed_ms:performance.now()-started};const out=Buffer.from(json(result));
  counters.bridge_response_body_bytes+=out.length;
  res.writeHead(status,{'content-type':'application/json','content-length':out.length});res.end(out);
 }catch(e){const out=Buffer.from(json({error:e.shortMessage||e.message}));counters.bridge_response_body_bytes+=out.length;res.writeHead(400,{'content-type':'application/json','content-length':out.length});res.end(out);}}));
 const port=await listen(api);
 console.log(json({status:'READY',url:'http://127.0.0.1:'+port,rpc_url:rpcUrl,chain_id:chainId,contract_address:contractAddress,
   client_addresses:Object.fromEntries(['c0','c1'].map((c,i)=>[c,addresses[i]])),versions,deployment}));
}
main().catch(e=>{console.error(e.stack||e);process.exitCode=1;});
process.on('SIGTERM',()=>{provider?.destroy();shutdown().then(()=>process.exit(0)).catch(()=>process.exit(1));});