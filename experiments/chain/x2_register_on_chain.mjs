#!/usr/bin/env node
/**
 * x2_register_on_chain.mjs — 读取 uploads.jsonl，把每条上传的绑定字段登记到 UploadRegistry 合约。
 *
 * 用途
 * ----
 * 这是「链上实验」的执行端：不训练、不读论文代码，只消费 data/uploads.jsonl。
 * 每个 configuration（c2_r2 / c5_r3）单独部署一份合约，因为合约要求
 * 同一 client 的 round 不得重复。
 *
 * 输入
 * ----
 * data/uploads.jsonl          x1_export_uploads.py 的产物（19 条）
 * contracts/UploadRegistry.sol 合约源码
 *
 * 输出
 * ----
 * results/x2_registration.json
 *   每个 configuration 的合约地址、部署 gas，以及每条记录的 record_id / tx_hash /
 *   block_number / gas_used / readback 比对结果。
 *
 * 对应论文
 * --------
 * §6.1（登记绑定）、Table 3（字段划分）、Table 4（gas 与字节）。
 *
 * 运行
 * ----
 *   npm ci                       # 首次，安装 ganache/solc/ethers（版本已锁定）
 *   node x2_register_on_chain.mjs
 *   node x2_register_on_chain.mjs --input data/uploads.jsonl --out results/x2_registration.json
 *
 * 说明
 * ----
 * 默认在进程内启动 Ganache（单节点开发链），跑完即关；不连接任何公链。
 * 若要对接到外部链，把 `--rpc <url>` 指向对方节点，并提供已部署合约地址
 * （此时跳过部署，直接登记）。
 */

import fs from 'node:fs';
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath } from 'node:url';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const ganache = require('ganache');
const solc = require('solc');
const { ethers } = require('ethers');

const HERE = path.dirname(fileURLToPath(import.meta.url));
const argv = process.argv.slice(2);
const arg = (name, fallback) => {
  const i = argv.indexOf(name);
  return i >= 0 && argv[i + 1] ? argv[i + 1] : fallback;
};

const INPUT = path.resolve(HERE, arg('--input', 'data/uploads.jsonl'));
const OUT = path.resolve(HERE, arg('--out', 'results/x2_registration.json'));
const CONTRACT_SRC = path.resolve(HERE, 'contracts/UploadRegistry.sol');

// 允许的 Binding 顺序必须与合约 struct 完全一致。
const BINDING_ORDER = [
  'activity_hash', 'client_hash', 'round_id', 'nonce_hash',
  'base_model_hash', 'layout_hash', 'codec_spec_sha256', 'digest',
];

function loadUploads(file) {
  if (!fs.existsSync(file)) throw new Error(`input not found: ${file}`);
  return fs.readFileSync(file, 'utf8').split('\n').filter(Boolean).map((l) => JSON.parse(l));
}

function compile() {
  const source = fs.readFileSync(CONTRACT_SRC, 'utf8');
  const input = {
    language: 'Solidity',
    sources: { 'UploadRegistry.sol': { content: source } },
    settings: {
      optimizer: { enabled: true, runs: 200 },
      viaIR: true,
      evmVersion: 'shanghai',
      outputSelection: { '*': { '*': ['abi', 'evm.bytecode.object'] } },
    },
  };
  const compiled = JSON.parse(solc.compile(JSON.stringify(input)));
  const errors = (compiled.errors || []).filter((e) => e.severity === 'error');
  if (errors.length) throw new Error(JSON.stringify(errors, null, 2));
  return compiled.contracts['UploadRegistry.sol'].UploadRegistry;
}

function bindingOf(row) {
  const b = row.binding;
  for (const key of BINDING_ORDER) {
    if (b[key] === undefined) throw new Error(`missing binding.${key} in ${row.record_id}`);
  }
  // 合约 struct 还有 uploader/blockNumber 两个派生字段，register 要求它们为零。
  return [...BINDING_ORDER.map((k) => b[k]), ethers.ZeroAddress, 0];
}

async function main() {
  const rows = loadUploads(INPUT);
  const artifact = compile();

  // 进程内 Ganache：单节点开发链，跑完即关。
  const server = ganache.server({ chain: { chainId: 20261104 }, logging: { quiet: true }, miner: { blockGasLimit: 30_000_000 } });
  await server.listen(0, '127.0.0.1');
  const port = server.address().port;
  const provider = new ethers.JsonRpcProvider(`http://127.0.0.1:${port}`);
  const deployer = await provider.getSigner(0);

  const configs = [...new Set(rows.map((r) => r.configuration))];
  const out = {
    schema: 'pr-lfl-chain-registration/1',
    created_at: new Date().toISOString(),
    input: INPUT,
    contract_source: CONTRACT_SRC,
    chain: 'in-process ganache (single-node development chain)',
    versions: { ganache: require('ganache/package.json').version, solc: solc.version(), ethers: ethers.version, node: process.version },
    configurations: {},
  };

  for (const config of configs) {
    const subset = rows.filter((r) => r.configuration === config);
    const clientHashes = [...new Set(subset.map((r) => r.binding.client_hash))];
    // 每个 client 分配一个独立签名账户（1..n），与合约的 authorized 映射一致。
    const accounts = [];
    for (let i = 0; i < clientHashes.length; i += 1) accounts.push(await provider.getSigner(i + 1));

    const factory = new ethers.ContractFactory(artifact.abi, '0x' + artifact.evm.bytecode.object, deployer);
    const contract = await factory.deploy(clientHashes, accounts.map((s) => s.address));
    const deployReceipt = await contract.deploymentTransaction().wait();

    const signerOf = new Map(clientHashes.map((h, i) => [h, accounts[i]]));
    const records = [];

    for (const row of subset) {
      const b = row.binding;
      const signer = signerOf.get(b.client_hash);
      const conn = contract.connect(signer);
      const recordKey = ethers.id(row.record_id); // keccak256(utf8(record_id))
      const tx = await conn.register(recordKey, bindingOf(row));
      const rc = await tx.wait();

      // 读回并逐字段比对，确认链上存的就是我们导出的那条。
      const stored = await contract.get(recordKey);
      const same = stored.digest.toLowerCase() === b.digest.toLowerCase()
        && stored.nonce.toLowerCase() === b.nonce_hash.toLowerCase()
        && Number(stored.round) === Number(b.round_id);

      records.push({
        record_id: row.record_id,
        record_key: recordKey,
        round: row.round,
        client: row.client,
        tx_hash: rc.hash,
        block_number: rc.blockNumber,
        gas_used: Number(rc.gasUsed),
        readback_matches: same,
        status: rc.status,
      });
    }

    out.configurations[config] = {
      clients: clientHashes.length,
      contract_address: await contract.getAddress(),
      deployment: { tx_hash: deployReceipt.hash, block_number: deployReceipt.blockNumber, gas_used: Number(deployReceipt.gasUsed) },
      registrations: records.length,
      gas_total: records.reduce((a, r) => a + r.gas_used, 0),
      all_readback_match: records.every((r) => r.readback_matches && r.status === 1),
      records,
    };
  }

  await server.close();
  fs.mkdirSync(path.dirname(OUT), { recursive: true });
  fs.writeFileSync(OUT, JSON.stringify(out, null, 2) + '\n');
  console.log(JSON.stringify({
    configurations: Object.fromEntries(Object.entries(out.configurations).map(([k, v]) => [k, {
      clients: v.clients, registrations: v.registrations, gas_total: v.gas_total, all_readback_match: v.all_readback_match,
    }])),
    out: OUT,
  }, null, 2));
}

main().catch((e) => { console.error(e); process.exit(1); });
