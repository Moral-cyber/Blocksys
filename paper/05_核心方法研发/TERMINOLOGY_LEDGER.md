# Terminology ledger for manuscript revision

|Canonical English|中文|Definition and usage|
|---|---|---|
|label-skew heterogeneity|标签分布异构|Client label proportions differ. The current benchmark does not establish arbitrary feature or concept shift.|
|personalized federated learning (PFL)|个性化联邦学习|Client-distribution predictions; distinguish these metrics from a single pooled/global predictor.|
|prior-regularized vector adaptation|先验正则化向量适配|Twenty local FP32 parameters: classwise slopes and additive bias; bias is regularized toward centered, smoothed TRAIN label frequencies. Existing mechanisms are combined, with no atomic first claim.|
|prior-anchor bias|先验锚定偏置|Ten-parameter restricted control with all slopes fixed at one. Preserve earlier accuracy-oriented STOP records.|
|zero-anchor vector scaling|零偏置锚定向量校准|Same twenty parameters and ridge coefficient as the candidate; slope anchor one and bias anchor zero.|
|bias-corrected temperature scaling (BCTS)|带偏置的温度缩放|One positive common slope and ten additive biases. Specify whether bias is anchored at zero or the TRAIN prior.|
|error feedback (EF)|误差反馈|The untransmitted FP32 update residual is carried to a later round; pending residual is committed only after matching incorporation confirmation in the protocol design.|
|uniform3 block quantization|统一三位块量化|Block size256, per-block FP32 scale and bit-width field, packed codes,64B header; actual shared-update wire54067B for136874 parameters.|
|Top-K with EF|Top-K稀疏加误差反馈|Select6843 coordinates at cap54808B including header and index/value pairs. Same cap does not mean identical actual wire to uniform3.|
|negative log likelihood (NLL)|负对数似然／交叉熵|Mean categorical loss in natural-log units. It is the new prospective question's primary endpoint; lower is better.|
|multiclass Brier score|多分类Brier分数|Per-sample sum across ten classes of squared probability errors, averaged over samples; no class-count normalization.|
|top-label expected calibration error (ECE15)|最高概率类别校准误差|Fifteen fixed equal-width confidence bins; macro client ECE and pooled ECE are distinct. NLL improvement does not imply ECE improvement.|
|client-macro metric|客户端宏平均指标|Equal weight over20 clients. Clients are not independent training seeds.|
|TRAIN-internal client holdout|TRAIN内部客户端留出集|Development evaluation drawn from official TRAIN. Different partition seeds still reuse the same source dataset.|
|official TEST|官方测试集|Reserved for final locked comparisons; do not call current development metrics test results.|
|ledger-binding registration interface|链上绑定登记接口|Proposed finalized digest/context binding before admission. Actual execution is local SQLite only; real-chain adapter is pending.|
|authentication tag|认证标签|A symmetric MAC, not a digital signature or public non-repudiation proof.|
|incorporation|纳入聚合|An admitted update is actually included in the recorded aggregate/model transaction. Acceptance alone does not authorize EF commit.|
|codec specification digest|编码规范摘要|Bound in the packet, active configuration and local registry before decode; does not prove a decoder honestly executes that specification.|
|local durable upload integration|本地持久上传集成|SQLite recovery fixtures plus a real-TRAIN two-client/two-round diagnostic; no executed ledger consensus or measured network.|

Use these terms consistently in the English manuscript, equations, figures and checklist. Refer to raw result keys separately when necessary; do not vary scientific terms for stylistic variety.
