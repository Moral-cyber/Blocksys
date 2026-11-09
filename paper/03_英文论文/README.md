# 英文论文当前稿

阅读入口为 `main.pdf`：**Prior as Center, Not Offset: Post-hoc Calibration for Label-Skewed Federated Learning**。

本稿已按更新任务书路线 A 完成修订，核心为冻结共享 logits 上“可纠正的本地类别先验中心”。全文12页，摘要136词，33条被引文献，正文两张表、主表六行、三张矢量图。低比特训练上传54,067 B，十类预测器本地状态80 B。原13组数字、模板、导言区和旧证据保持。

本轮实际完成：加权/尾部指标、固定划分内配对与条件bootstrap、λ扫描、冻结预测器重分配诊断、FedAvg/FedProx十二次训练、pFedMe六次训练与冻结TEST评分、两任务各三划分开发验证、CIFAR-100三种子开发验证。CIFAR-100先验中心使宏NLL 2.4283→2.2896，准确率35.27%→35.12%；不利准确率/ECE、漂移诊断均保留。新增实验的独立复算与来源见 `revision_artifacts/route_a_20261105/`。

Google Cloud AI Research 主导的 ScientistTwo 公开流程已用于真实原生代理审阅和本机MPS实验。安装的 Open-ScientistTwo v0.4.0 是非官方实现，本地 bridge 已记录两组角色输出与五组实验；不是 Google 官方源码或原系统基准复现。新geometry候选未满足双任务继续门槛，已停止，不列为论文新贡献。

只推进轻量化FL与标签分布异构性；没有新增水印、搭链或攻防实验。历史上传实现仅保留附录范围。详细11项验收、最小修正和完成情况见 `checks/REVISION_STATUS.md`；`CHECK_REPORT.md` 与 `CURRENT_REVISION.json` 指向本版。既有 `checks/` 为受保护历史记录，只有新增状态文件反映当前修订。

这份交付形成有数据支持的明确NLL贡献和完整英文论证，不等同于任意数据异构性已解决或CCF录用保证。作者/单位与最终会议track尚未提供，未实际投稿。修订前备份：`../01_论文材料/历史论文版本/route_A_before_20261104T235231+0800/03_英文论文/`。
