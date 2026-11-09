# 图形加速确认协议与偏离记录

记录时间：2026-11-03T22:29:04.648722+08:00

## 采用原因

此前普通执行上下文不可访问MPS；只读提权检查发现本机MPS可用，50 minibatches约.225秒。新独立加速入口已经过独立CPU路由/状态/后端隔离检查，以及root实际MPS小batch等价和重复性核验。采用加速是运行效率决定，未依据任何确认run的official_test_final选取后端或算法。

## 数值分组

只迁移本地训练，模型增量回CPU；codec、残差、全局模型、聚合、验证和checkpoint均CPU。不同后端浮点轨迹不能假定相同。CPU与MPS结果分开汇总，必须使用summarize_accelerated.py，不得混池。原CPU文件、实验和结果继续保留。

## 开发选择

MPS开发只用旧清洁CIFAR固定数据、training seed20261113、2000张官方训练集holdout、20轮、rho=.05，比较TopK-EF、prefix tau=.9/.99、整体能量约束tau=.9/.99、RegTopK mu1/5/10、FedProx mu.01/.1。预设按每次运行最后5轮验证loss选候选和基线参数。官方test不在开发路径。

## 新正式确认

- 40轮、rho=.05、全部20客户端、SGD .03、batch64、1本地epoch。
- 优先使用新的Cifar10IndexA0p1数据划分，partition seed20261122；training seeds20261141/42/43。
- 比较Dense FedAvg、TopK-EF、开发选定的RegTopK参数、开发选定FedProx参数和开发选定新方法。
- 正式只评估预定末轮官方test，报告全部seed及均值/样本标准差，不选最佳test轮、不回用test改参数。
- FashionMNIST作为第二任务；新方法参数直接迁移，基线可用Fashion开发验证选择其既定参数网格，不能通过官方test选择。
- 再按核心主张补独立划分、近IID及必要消融；结果不支持时继续设计而非改写成有效。

原CPU冻结确认20261123/24/25仍作为独立CPU协议记录，不混同新正式MPS主表；任何未完成运行不进入完整seed统计。此记录属于公开标注的探索开发与硬件可行性修订，并非事前未发生的注册研究。
