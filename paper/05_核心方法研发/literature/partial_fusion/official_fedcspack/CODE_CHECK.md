# FedCSPACK 作者仓库只读核查

2026-11-03。唯一仓库：[NigeloYang/FedCSPACK](https://github.com/NigeloYang/FedCSPACK)。固定 `main` commit **6bb6d9ef68d8e74873cbd3480337deabbe3a7e76**，提交时间 `2026-01-06T04:48:10Z`。只下载/读6个文件，没有安装依赖、导入模块或训练；本报告是静态源码核查，不等于复现实验。原始GitHub metadata、commit、完整tree与blob响应同目录保存。未审计其他commit/分支。

## 关键结果与论文差异

| 项目 | 固定commit的实际代码 | 与原文的关系 |
|---|---|---|
| 选包阈值/预算 | `system/clients/clientcspack.py` 99行整模cos，103行包cos；106–114行对**所有包cos < 整模cos**的包上传，无排序、固定K或固定上传率 | 符合方法式(5)周围的低于阈值叙述，消除引言“highest similarity”歧义；不能视作固定5%预算top-K |
| KL与权重 | 同文件104行 `F.kl_div(log softmax(local_packet),softmax(global_packet),reduction='sum')`，即 **KL(softmax(global)‖softmax(local))**；109行 `abs(cos+KL)`；cs/kd消融也取abs（111/113行） | 论文式(7)显示local·log(local/global)，未写softmax或abs；代码既明确正值/归一转换，也采用相反KL方向。不得把raw signed参数直接代入论文KL式 |
| 上传绝对模型值还是增量 | `clientbase.py` 87–90行：`delta=local_trained−pre_global`；`clientcspack.py` 59、78、87、114行切包并上传该delta | 代码解决绝对模型值/增量歧义；符号与论文Alg.1第15行所写 `global−local` 相反 |
| server权重/有效位 | `servercspack.py` 100行 `agg_mask=sum(uploaded_masks)`；104–109行仅处理 `val>0`，按 `val/agg_mask[idx]` 归一；122行 `global+=agg_delta` | 不覆盖的包贡献0，原global包保留。虽读取train_samples（90行）并循环为sw（102行），sw没有进入权重公式；实际是mask权重，不是样本量×mask权重 |
| **mask下行/部分融合** | `servercspack.py` 30–38行仅调用 `update_client_params(global_model)`；`clientcspack.py` 128–132行把**每一个参数**的local和pre_global都clone为global，函数不接收mask | **与论文不一致。** Alg.1第8行及“Top-k...”首段描述下发global mask并按有效位置覆盖；此代码没有mask下行或接收端部分覆盖。`agg_mask`仅用于本轮server聚合 |
| 反馈影响下轮上传优先级 | 下一轮重新用global/local模型算cos；没有把 `agg_mask` 或其coverage状态发给client、没有显式mask反馈优先级 | global模型本身会影响新一轮cos，但不能将其说成显式mask反馈闭环 |

以上定位均可用固定commit永久链接核对：

- [clientcspack.py](https://github.com/NigeloYang/FedCSPACK/blob/6bb6d9ef68d8e74873cbd3480337deabbe3a7e76/system/clients/clientcspack.py#L72)
- [clientbase.py](https://github.com/NigeloYang/FedCSPACK/blob/6bb6d9ef68d8e74873cbd3480337deabbe3a7e76/system/clients/clientbase.py#L87)
- [servercspack.py](https://github.com/NigeloYang/FedCSPACK/blob/6bb6d9ef68d8e74873cbd3480337deabbe3a7e76/system/servers/servercspack.py#L30)
- [serverbase.py聚合入口](https://github.com/NigeloYang/FedCSPACK/blob/6bb6d9ef68d8e74873cbd3480337deabbe3a7e76/system/servers/serverbase.py#L162)：继承入口调用上述CSPACK的receive/aggregate覆盖方法。

默认设置：`main.py` 121–124行为 `spars_name=sim, pack_size=1024, spars_type=cskd`。代码只对 `.parameters()` 展平/更新；并非整个包含buffers的state_dict。代码在同进程内传torch对象/list，没有本次可核实的网络序列化或实测传输字节协议；不能由包数量自动推出真实mask/索引字节。

## 对后续有限闭环的含义

当前按固定5%预算、delta能量选位置却上传模型值、按样本量coverage平均、接收端保留/gate融合的实验，都有明确设计含义，但应标为自己的改编组合，不能称此commit的FedCSPACK复现。若比较准确官方实现，须保留其delta、softmax KL方向、abs权重、可变包数与完整global覆盖，另报告其实际通信预算；若刻意补齐论文所述mask下行部分融合，应标为“按论文描述补充的变体”。这项代码差异不自动否决组合研究，只要求baseline命名与预算边界准确。

一次有限反馈优先级检索的来源与近邻（TCS-H）另见 `../feedback_priority_search/SEARCH_CHECK.md`。该近邻已有上轮server位移→client生成共同上传支持→本轮上传的循环；尚需用明确反馈语义、相同接收规则、相同总字节对照证明当前具体组合的增量价值。
