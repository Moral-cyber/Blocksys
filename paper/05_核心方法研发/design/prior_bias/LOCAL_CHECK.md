# 独立NumPy核查
DONE；PASS 1901/1901；固定候选balanced_uniform3，决定**STOP**；外部`REVIEW_UNAVAILABLE`。
候选宏acc 77.812440%、CE 0.592503798；相对同模型prior +0.315042pp、最好prior +0.214179pp，均未达+.5pp。两项CE及learned门槛通过也不能挽救；保留其他loss fit结果，不换候选。

|模型|fit宏acc%|fit宏CE|weighted acc%|q10%|
|---|---:|---:|---:|---:|
|ce_uniform3|77.94111|0.58738955|75.98816|68.55838|
|fedlc_uniform3|78.08935|0.58715854|76.07617|69.10665|
|balanced_uniform3|77.81244|0.59250380|76.11618|69.48963|
|balanced_dense|77.84216|0.59120716|76.16419|69.48963|

80组TRAIN计数/IDs及复用holdout与config/旧PFL一致，按ID重建50000标签hash吻合；六score共480条client指标、120项汇总独立复算，最大差4.4e-16。目标/梯度/Hessian含有限差分、>=10I及mean-zero检查通过；完整Newton/Armijo轨迹独立重放，FP32 bias与保存FP64转换精确一致。独立重拟合FP64最大差1.6e-08，FP32位级一致78/80，另2组差1ulp且预测不变。
部分fit按FP64目标分辨率停止，最终grad∞最大1.9e-07，不能称全部≤1e−8；等价求和末位线搜索误差容差2e−10，不改效果门槛。
作者记录额外TRAIN forward150008样本/640批，Newton280步、评估377次；bias40B/client。新增CNN backward/新head训练/uplink为0，但不是零计算。缓存10060892B。
仅最终轮PFL开发，37502/12498且含旧dev2000；20client不是20seed，不冒充global或独立test。未读原图/officialtest，未做新CNN forward/训练，未导入作者solver；五个原图文件摘要仅保留作者记录，其余保护输入重验、源文件不变。完整计数、F/g/H、轨迹、六score及gates见LOCAL_CHECK.json。
