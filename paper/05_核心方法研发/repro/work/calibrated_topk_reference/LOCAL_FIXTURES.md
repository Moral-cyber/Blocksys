# TopK+EF同预算参考：CPU toy交付

2026-11-04。**PASS；仅CPU合成向量/图像和临时checkpoint，没有正式TRAIN加载、原图/TEST读取或长训练。** 外部`REVIEW_UNAVAILABLE`。这是压缩参考，不是新候选；旧balanced/prior/bias/confirm各seed判定保持不变。

## 启动入口

正式启动由root决定。本交付没有执行以下命令：

```bash
/opt/anaconda3/envs/Tinyproto/bin/python -B /Users/wind/Documents/Codex/2026-11-02/zhe/work/calibrated_topk_reference/train.py --out /Users/wind/Documents/Codex/2026-11-02/zhe/work/calibrated_topk_reference/development_p20261203_s20261211 --backend mps
/opt/anaconda3/envs/Tinyproto/bin/python -B /Users/wind/Documents/Codex/2026-11-02/zhe/work/calibrated_topk_reference/evaluate.py --model-root /Users/wind/Documents/Codex/2026-11-02/zhe/work/calibrated_topk_reference/development_p20261203_s20261211 --out /Users/wind/Documents/Codex/2026-11-02/zhe/work/calibrated_topk_reference/evaluation_p20261203_s20261211
```

固定两组`ce_topk`、`fedlc_topk`，partition20261203/seed20261211/20轮、batch64、SGD.03、epoch1。CLI拒绝其他seed/partition/轮数；toy仅直接调用函数使用2clients/2rounds。MPS不可用即失败，无自动CPU回退。

## 复用与真实wire

核查的旧`tinyproto_diag/compression/sparse_update_codec.py`只提供Tensor packet与公式字节、没有真实wire且并列TopK未固定，因此没有把它宣称为严格wire实现。已复用冻结`method_followup/guarded_codec.py:185`的`canonical_pairs`，SHA`3dcb6636f12d6dc17fe023a978a9016ce10593ed3da5472b1ad498a33b8c948e`。新增`wire.py`只补稳定选择、64B header与严格byte decoder。

Header为`<8sIIII40s`：magic `TOPKEF01`、version1、numel、K、pair_size8、40个零保留字节。payload是升序uint32LE index/FP32LE value对。固定模型N136874时K6843，真实`len(wire)=64+6843×8=54808`，没有只算公式或遗漏header。|u|并列选低index，再按index升序上wire；显式零和负零保留。未实现认证、网络或链。

同cap54808B下uniform3实际54067B，TopK多741B/client/round。400包TopK上传21923200B、uniform3上传21626800B；两者解析下行均219024000B。因此是相同上限，实发字节不同。新wrapper默认纳入三个已完成uniform seed：只有seed20261211同初始化；seed12/13明确标为不同seed的描述比较。

## CPU toy覆盖

- 满长度合成FP32向量：Python稳定排序独立核选中indices；独立`struct.unpack_from`逐pair解码，不调用新decoder；values逐位相同、实际54808B、重复编码逐字节一致。
- 明确并列、全零和signed-zero；24类错误输入/包全部拒绝，包括重复/无序/越界、NaN/Inf、坏magic/version/count/size/保留位、截断、尾字节及超cap。
- 12张随机uint8图像、2clients（TRAIN3/5、holdout2/2），两组真实SmallCNN CPU各2轮，实际4梯度/组。直接复用冻结balanced.train，记录同batch seed/order、共同初始anchor和client间global不变。
- 独立byte decoder逐轮重构q，EF=`delta+r−q`、固定真实样本权聚合的下一轮anchor/最终模型/最终residual逐位吻合；首包与当时真实wire逐字节一致。
- 在完整round1后中断，resume到round2与连续运行模型/EF逐位一致；完成态resume无训练；11类不兼容config/checkpoint在训练前拒绝。
- 新评估wrapper拒绝不完整checkpoint；完整toy调用冻结`confirm.evaluate_model`，TRAIN-only prior/bias与matched head协议不变，保存IDs互斥、bias重拟合逐位、独立logaddexp holdout指标吻合。bias额外TRAIN forward16样本、CNN backward0；matched head4梯度。输出`REFERENCE_ONLY`、`gates:null`。
- official TEST guard以不存在的dummy路径在open前拒绝；实际official文件读取0。全部源身份前后相同。

机器证据`LOCAL_FIXTURES.json`；toy输出`/var/folders/0t/lt7l05rn1zz3xrpbkj6ym_8m0000gn/T/strict_topk_reference_fixture_4gpkrivq`。可重算toy：

```bash
/opt/anaconda3/envs/Tinyproto/bin/python -B /Users/wind/Documents/Codex/2026-11-02/zhe/work/calibrated_topk_reference/local_fixture_check.py
```

## 冻结身份

| 文件 | SHA256 |
|---|---|
| PLAN.md | `b0daad197d50f71c4d18e3ea3d12878192ab9705a3d6b6179479048e8c6d7130` |
| common.py | `37976f773cbaca44ee444d95091cb77cb916e06a9943837dbbee8586425a4a58` |
| wire.py | `6bf9856e73fba6d601bbcd8ce19cb9ec3bdfe09f17e58ca79e0faef000c86693` |
| train.py | `13d8538e671302e1f8ec891fea3f0b87c6cb6a996774222aef4d36856946affe` |
| evaluate.py | `22cec15467ae06629c361a8f510da45cc1637bdf2ef1242db42169c88dbb3e8c` |
| local_fixture_check.py | `11a6b0341a01bace9fda83748d83456e3051a1f3dae783c4e32466af6a0d441f` |

PLAN1228字符。新common强pin全部四个confirm源，继承其已pin balanced.train/codec/prior solver、数据与模型/helper身份；再把新增五源和pair serializer纳入训练config签名。checkpoint复用严格完整config、CPU FP32有限模型/EF/轮历史验证。新入口没有修改旧source、数据或gates。
