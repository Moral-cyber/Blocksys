# 官方测试集访问范围的精确定义

2026-11-04，核查已安装 torchvision.datasets.CIFAR10 源码。

历史新索引入口 `run_partitions.load_official_train` 调用 `CIFAR10(train=True, download=False)`。
该构造函数的 `_check_integrity()` 遍历 `train_list + test_list`，可为了MD5完整性校验读取
`test_batch` 的原始文件字节。因此“从未打开任何official-test文件字节”不是可成立的描述。

`train=True` 随后仅反序列化五个 `data_batch_1..5`，不会把test样本/标签装入本次训练或
开发预测数组。当前方法设计、go/stop和所有近期诊断均未计算或使用official TEST预测/指标。
这与MD5字节校验不同；文件完整性读取没有把test标签用于选型。

后继需严格限制文件读取的入口，直接只解包五个官方TRAIN文件，并核对历史manifest
记录的SHA256、标签摘要、原始行ID和相同归一化。冻结的历史源码与报告不追改，
其 `official_test_loaded=False` 解释为“没有载入测试记录进行学习/推理”，不解释为
“系统调用层从未读取测试文件字节”。

本说明为访问范围更精确的补充，不把已有最终测试指标用于新方法选型，不把TRAIN holdout
改称最终测试。旧归档可能含更早 `official_test_final` 字段，仍禁止用于当前研发筛选。
