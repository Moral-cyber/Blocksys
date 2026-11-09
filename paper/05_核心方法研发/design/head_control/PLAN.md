# 分类头子空间漂移控制：有限开发诊断

2026-11-04，运行前冻结。问题：当前alpha=.1标签异构dense学习损失远大于压缩附加损失；
前一梯度敏感性量化规则已经停止。本候选直接改变本地学习梯度，不能视为旧的强制头坐标上传。

## 精确定义与已有部件

SmallCNN最后Linear(64,10)的650个参数记为h，其他136224参数记为b。
仅h梯度改为g_h+c_h−c_ih，普通SGD步长η=.03；本轮内c固定。
实际K_i个batch后delta_i=local−global，c_ih'=c_ih−c_h−delta_ih/(ηK_i)。
固定全参与、p_i=n_i/Σn_i，c_h=Σp_i c_ih；服务器按已传delta_h及K_i重建c_h'。
这是SCAFFOLD型控制在分类头子空间的限制，delta重建control与SCALLION有直接先例，
不能把零额外control上行当首次提出。无新收敛或梯度无偏保证。

候选上传FP32头delta，body用冻结mixed_precision/codec.py的mixed_mse及标准EF。
主体编码预算=54808−32(wrapper)−2600(head)=52176B；wrapper含版本/参数数/实际步数/长度。
用实际wire解码值聚合。头无EF，body residual=delta_body+旧residual−decoded_body。
所有方法上传不超过54808B/client。按方法分别报告实际字节。
控制状态每client2600B；额外下行控制=16B头+2600B，global模型下行仍为dense解析值。
固定权重和全参与是当前边界，不声称已支持部分参与。

## 四组，fresh seed20261191

1. plain_uniform：普通CE，本地无校正，冻结uniform3全模型EF。
2. exact_head：普通CE，精确头+mixed_mse body，无校正。
3. head_control：同2传输，仅增加头部控制。
4. head_control_dense：同3学习，全部delta精确传输，隔离压缩损失；此组不满足低字节上限。

同分区20261143、CIFAR10官方TRAIN内部37500训练/2000分层globaldev，20clients，
alpha=.1，batch64，localepoch1，20轮，全参与固定样本权，MPS本地/CPU全局；
共同初始化、batch流，不叠加校准CE/额外SGD/额外数据，不读取official TEST。
这是多次开发选择后形成的探索性试验，不能写成独立确认。

## 去留规则

先验证逐batch原始头梯度均值与c_i'公式、server加权控制一致、FP32 EF/真实解码/预算，
不将头校正误施于body。四组都完成20轮，不按中途accuracy挑选。
以round16–20均值：head_control CE低于plain_uniform与exact_head，
accuracy比两者分别至少+0.5pp，才值得独立复核与新seed/新分区确认；否则停止此固定配方，
不调头范围、强度、动量、位宽网格或阈值挽救。dense只诊断压缩代价，不替代强低字节对照。
通过也不表示核心已成立。下一阶段必须有强full-control基线、机制删除、独立分区/多seed和第二任务。

一次实际训练不超过4×20轮，预计本机MPS约5–8分钟。轮次不是独立seed。
不作真实网络、设备能耗、链共识、最终测试或新机制首创声明。
