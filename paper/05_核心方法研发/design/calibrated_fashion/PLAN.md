# FashionMNIST迁移边界：独立wrapper

2026-11-04固定。FedLC-inspired＋uniform3＋prior-anchor bias迁移第二任务；不保证收益，不改CIFAR或旧STOP。

partition20261207、alpha.1、20clients，seed20261221及后续22/23。SmallCNN136874，gray28在uint8域pad2至32/repeat3/归一化[-1,1]，不称Fashion最优。ce_uniform3、fedlc_uniform3、fedlc_dense各20轮，SGD.03/b64/epoch1，全参与样本权、同初始化/batch流；75%TRAIN按实际计数，globaldev2000。

仅读两个TRAIN gzip，核对官方MD5、原SHA与labels hash；拒绝t10k/test_batch字节读取。复用冻结run_fashion.partition_indices/FashionIndexDataset，不调用旧load_official_train/source_file_manifest。

importlib复用冻结confirm common/train/evaluate、loss/codec/lambda10 solver；仅进程内覆盖prepare/dataset/source/runtime。新旧依赖及路由进签名，严格复载；Fashion/CIFAR不互载。

uniform3真实wire/decode/EF，cap54808B；dense上传/global下行解析547560B。final全部clientholdout评估raw/prior/bias及SGD1epoch head/head+prior；仅TRAIN拟合。bias一次TRAINforward/零backward/40B，强对照梯度成本另计。

继承六gates：FedLC量化+bias比同loss prior准确率≥+.5pp且CE更低；比CE+bias非劣.5pp、CE不差.01；比同模型best head非劣.5pp；比FedLC dense+bias非劣.5pp。通过FOLLOW_UP，失败STOP完整报告；不调lambda/温度/bit或挑seed。保留零anchor、BCTS/vector/FedCal强对照；任务/seeds结果不互代。

先CPUtoy、真实TRAIN只读分区fixture，再报主任务启动。无新架构或网络声明。
