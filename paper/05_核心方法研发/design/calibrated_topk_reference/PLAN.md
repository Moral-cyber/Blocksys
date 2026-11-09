# 同预算TopK+EF参考：固定入口

2026-11-04。只补现有pipeline的强压缩参考，不是新候选，不设GO，不改变旧balanced/prior/bias或confirm各seed判定。

固定CIFAR10官方TRAIN内partition20261203、alpha.1、20clients，seed20261211、20轮、全参与实际样本权。复用冻结confirm.common.prepare的五TRAIN文件严格SHA/labels校验与纯ID分区、SmallCNN136874参数、balanced.train的CE/FedLC-inspired tau1损失；batch64/SGD.03/localepoch1，seed+100r+i，所有真实counts原样报告。

两组ce_topk、fedlc_topk。每client上传cap54808B；64B真实header＋6843个uint32LE index/FP32LE value对，每对8B，实际54808B。全局|delta+r|降序、并列低index优先，发送按index升序；复用冻结guarded_codec.canonical_pairs，新增薄header/严格byte decoder。拒重复/无序/越界/非有限/错误header/保留位/截断/尾字节；不声称认证或网络实测。仅decode的q用于聚合，EF=delta+旧r−q。global下行解析547560B/client；保存首包。

同cap的uniform3实际54067B，TopK多741B/client；20轮各400包分别21626800/21923200B，解析下行均219024000B。与uniform的seed20261211为同初始化比较；seed12/13只作不同seed的预算及结果描述，不冒充配对重复。

source/PLAN、冻结依赖、数据、IDs/counts/weights和runtime进config签名；checkpoint完整绑定config/模型/EF/轮历史，不兼容拒resume。只读五TRAIN，main在prepare前装test_batch字节guard。

最终仅完整final checkpoint由新wrapper调用冻结confirm.evaluate_model：TRAIN counts prior、冻结lambda10 prior-anchor bias/20步Armijo、同协议matched head；holdout只评分，输出保存arrays和实际成本。无新gate，无holdout调参，不改旧源码/阈值。先CPU synthetic toy核真实wire/index/value/bytes、EF/聚合/resume/负例；不运行正式数据训练。外部REVIEW_UNAVAILABLE，交root决定启动。
