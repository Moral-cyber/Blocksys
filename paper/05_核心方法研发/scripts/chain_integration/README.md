# 真实 TRAIN、HTTP 与本机 EVM 链集成验证

当前入口为 `run_learning.py`；`network_service.py` 提供回环 HTTP 服务与认证收据，`ledger_bridge.cjs` 启动 Ganache 并部署 `UploadRegistry.sol`。本次结果保存在项目的 `05_核心方法研发/results/chain_integration/result.json`：**PASS，128 项检查，22 次真实梯度计算，4 个新更新、4 次链登记**，两轮模型和已提交 EF 残差均与直接参考路径逐位一致。总运行约 6.06 秒，属于本机诊断观测。

## 依赖与运行

使用现有 Python 环境 `/opt/anaconda3/envs/Tinyproto/bin/python`（NumPy、PyTorch 与支持 HMAC-SM3 的 hashlib）。本次 Node 为 v24.21.0；Node 依赖固定为 Ganache **7.9.2**、Solidity 编译器 solc **0.8.30**、ethers **6.15.0**，完整解析记录见同目录 `package-lock.json`。运行时位于临时目录，不把 `node_modules` 放入项目。复用已安装的 `/private/tmp/paper-chain-runtime` 即可；需要重装时执行：

```bash
mkdir -p /private/tmp/paper-chain-runtime
cp /Users/wind/Desktop/Codex/paper/05_核心方法研发/scripts/chain_integration/package.json /Users/wind/Desktop/Codex/paper/05_核心方法研发/scripts/chain_integration/package-lock.json /private/tmp/paper-chain-runtime/
npm ci --prefix /private/tmp/paper-chain-runtime
```

入口命令为：

```bash
/opt/anaconda3/envs/Tinyproto/bin/python -B /Users/wind/Desktop/Codex/paper/05_核心方法研发/scripts/chain_integration/run_learning.py --runtime /private/tmp/paper-chain-runtime
```

**入口拒绝覆盖已经存在的成功结果，因此当前项目再次执行上述命令会在训练与节点启动前退出。** 如需独立复跑，先复制项目，保留原有相对目录结构及所需 CIFAR TRAIN 数据；在副本中移走已有的 `05_核心方法研发/results/chain_integration/result.json`，再将命令中的脚本路径改为副本路径。当前入口没有 `--out` 参数。运行需要允许本机 `127.0.0.1` 监听；受限沙箱内的监听失败不能解释成链故障。Runner 自动启动临时端口、临时 SQLite 与 EVM 数据库，完成后关闭服务并删除临时状态；HMAC 密钥不持久化，账户私钥不输出。

## 验证含义

客户端 0/1 沿用原 CIFAR 划分与初始化种子，分别使用 450/159 个 TRAIN 样本，每轮完成一个本地 epoch，共两轮；没有读取官方 TEST，也没有增加主实验训练种子。两条路径共享每个客户端当轮实际算出的训练更新，分别编码，以隔离协议处理影响。两条路径均采用 FP32 本地 CNN/codec、float64 EF/服务器模型与 `math.fsum` 聚合；该逐位一致性对应本诊断的数值配置。

完整顺序是：真实训练更新 → 新编码并持久保存 pending → HTTP 传输 → HMAC 与固定轮次上下文校验 → EVM 登记交易 → 等待包含块之后两个后继块 → 验证成功交易、规范块哈希、固定 chain/contract/uploader 与完整绑定 readback → SQLite accept/finalize → 域分离 HMAC-SM3 收据 → 客户端验证 session、随机 challenge、完整消息上下文、状态和结果模型哈希 → EF 提交。**链已登记不等于模型已纳入聚合；链登记收据不能授权 EF 提交。** 测试覆盖链成功但本地 register 前应答丢失、finalize 后应答丢失、服务/SQLite 重建恢复、精确字节重试、篡改及旧收据拒绝。重试不重新编码、登记交易、聚合或重复提交 EF。

固定全参与缺少任一客户端时，轮次保持 OPEN，保留 pending 与旧 EF，禁止 finalize 或推进下一轮；该客户端回来后完成同一轮。永久缺席时保持安全停止，没有自动改权重或继续训练的活性机制。

## 计量与范围

结果逐次记录应用 HTTP 请求/响应 body、实际发送的 HTTP 请求头/body、服务响应头/body及调用时延；包含反例、查询与重试，不能把这些累计流量当作四次正常上传的净成本。四个新 codec payload 共 **216,268 B**，四个 envelope/tag 共 **217,632 B**。Node 代理另统计实际 JSON-RPC 请求/响应 body；`bridge_elapsed_ms` 与应用 HTTP 时延覆盖其各自调用范围，没有单独测量每条 RPC 的时延。`bridge_stats` 是最终 `/stats` 处理期间的截面，计数不包含该次响应自身及随后关闭操作，也不补计构造该截面时才发起的后续 RPC。全部计量排除 TCP/IP、TLS、重传和节点间共识流量；gas 是开发链执行消耗，未换算实际费用。

Ganache 是**真实执行合约的单节点本机开发链**，两后继块由本机手动挖出；结果不构成公链、PoW、多节点共识或生产最终性证据。服务器、HMAC 密钥、SQLite 聚合权威、链适配器/节点与正确 decoder 仍在信任范围内。登记绑定外部上传标识和摘要，不证明诚实训练、抗投毒或模型内嵌水印。恢复测试重建服务与数据库对象，没有注入操作系统崩溃。这组系统验证服务于论文的轻量化方法链路，不替代主实验的预测质量结果。
