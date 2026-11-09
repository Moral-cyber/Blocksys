"""Read-only, seed-level summary of a frozen confirmatory protocol.

Official final-test metrics are accessed only with --allow-final-test. This
program reports predetermined comparisons; it never selects an algorithm.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

from summarize_accelerated import settings as accelerated_settings


METRICS=("final_test_accuracy","final_test_loss","last5_validation_accuracy",
         "last5_validation_loss","final_validation_accuracy","final_validation_loss")


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def finite_tree(obj):
    if isinstance(obj,float):
        return math.isfinite(obj)
    if isinstance(obj,dict):
        return all(finite_tree(v) for v in obj.values())
    if isinstance(obj,list):
        return all(finite_tree(v) for v in obj)
    return True


def metric(value,*,accuracy=False):
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):
        raise ValueError("missing/nonfinite metric")
    if accuracy and not 0<=value<=1:
        raise ValueError("accuracy outside [0,1]")
    if not accuracy and value<0:
        raise ValueError("negative cross-entropy loss")
    return float(value)


def setting(config,expected_rounds):
    _,identity=accelerated_settings(config)
    manifest=config.get("data_manifest",{})
    identity["confirmatory_partition_identity"]={k:manifest.get(k) for k in (
        "partition_seed","alpha","partition_generator_sha256","train_indices_sha256",
        "holdout_indices_sha256","official_train_labels_sha256","official_train_batch_files",
        "source_gzip_files","download_manifest_sha256")}
    identity["expected_rounds"]=expected_rounds
    return sha(json.dumps(identity,sort_keys=True,separators=(",",":")).encode())[:16],identity


def stats(values):
    values=[float(x) for x in values if x is not None]
    n=len(values)
    mean=statistics.mean(values) if values else None
    sd=statistics.stdev(values) if n>1 else None
    # The declared protocol has three training seeds. This is the conventional
    # two-sided 95% Student-t interval, df=2; no normality claim is inferred.
    half=4.302652729911275*sd/math.sqrt(3) if n==3 else None
    return {"n":n,"mean":mean,"sample_sd":sd,
            "t95_low":mean-half if half is not None else None,
            "t95_high":mean+half if half is not None else None}


def write_csv(path,rows,prefix=()):
    fields=list(prefix)+sorted({k for row in rows for k in row}-set(prefix))
    with path.open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize(input_path,output_path,*,allow_final_test=False,expected_rounds=40):
    input_path,output_path=Path(input_path),Path(output_path)
    paths=([input_path/"result.json"] if (input_path/"config.json").exists() else
           sorted(set(input_path.rglob("result.json"))|{p.with_name("result.json") for p in input_path.rglob("config.json")}))
    if not paths:
        raise ValueError("No run config/result paths found")
    accepted,excluded,identities=[],[],{}
    for result_path in paths:
        directory=result_path.parent
        provenance={"run_id":directory.name,"run_path":str(directory.resolve()),
            "config_path":str((directory/"config.json").resolve()),
            "result_path":str(result_path.resolve())}
        try:
            config_raw=(directory/"config.json").read_bytes()
            provenance["config_sha256"]=sha(config_raw)
            config=json.loads(config_raw)
            if not finite_tree(config):
                raise ValueError("nonfinite configuration metadata")
            result_raw=result_path.read_bytes()
            provenance["result_sha256"]=sha(result_raw)
            result=json.loads(result_raw)
            if result.get("status")!="DONE":
                raise ValueError("run is not DONE")
            if config.get("mode","train")!="train":
                raise ValueError("snapshot is not a confirmatory training trajectory")
            if config.get("rounds")!=expected_rounds:
                raise ValueError(f"configuration does not request exactly {expected_rounds} rounds")
            rows=result.get("rows",[])
            if len(rows)!=expected_rounds or any(not isinstance(r.get("round"),int) or isinstance(r["round"],bool) for r in rows):
                raise ValueError("wrong number or type of rounds")
            if [r["round"] for r in rows]!=list(range(1,expected_rounds+1)):
                raise ValueError("rounds are not a continuous ordered 1..R trajectory")
            if not finite_tree(rows) or not finite_tree(result.get("client_rows",[])):
                raise ValueError("nonfinite round/client diagnostic value")
            for row in rows:
                metric(row.get("validation",{}).get("accuracy"),accuracy=True)
                metric(row.get("validation",{}).get("loss"))
            seed=config.get("seed")
            method=config.get("variant")
            if not isinstance(seed,int) or isinstance(seed,bool) or not isinstance(method,str) or not method:
                raise ValueError("missing seed or method")
            if result.get("seed",seed)!=seed or result.get("variant",method)!=method:
                raise ValueError("config/result seed or method mismatch")
            if config.get("data_manifest",{}).get("test_used_for_selection") is True:
                raise ValueError("manifest declares final test used for selection")
            signature=config.get("run_signature")
            if signature:
                recomputed=sha(json.dumps({k:v for k,v in config.items() if k not in ("out","run_signature")},sort_keys=True).encode())
                if signature!=recomputed:
                    raise ValueError("config run_signature mismatch")
            group,identity=setting(config,expected_rounds)
            identities[group]=identity
            tail=rows[-5:]
            record={**provenance,"settings_id":group,"method":method,"seed":seed,
                "training_backend":config.get("training_backend","legacy_cpu_no_accelerator_metadata"),
                "split_seed":config.get("data_manifest",{}).get("partition_seed",config.get("split_seed")),
                "rounds":expected_rounds,"last5_round_count":len(tail),
                "last5_validation_accuracy":statistics.mean(r["validation"]["accuracy"] for r in tail),
                "last5_validation_loss":statistics.mean(r["validation"]["loss"] for r in tail),
                "final_validation_accuracy":rows[-1]["validation"]["accuracy"],
                "final_validation_loss":rows[-1]["validation"]["loss"],
                "final_test_accuracy":None,"final_test_loss":None}
            if allow_final_test:
                # Do not access this result field anywhere outside this gate.
                test=result.get("official_test_final")
                if config.get("test_final") is not True or not isinstance(test,dict):
                    raise ValueError("explicit final-test config/metrics missing")
                if not finite_tree(test) or test.get("samples")!=10000:
                    raise ValueError("nonfinite final-test result or expected 10,000 examples missing")
                record["final_test_accuracy"]=metric(test.get("accuracy"),accuracy=True)
                record["final_test_loss"]=metric(test.get("loss"))
                record["final_test_samples"]=test["samples"]
            accepted.append(record)
        except (OSError,ValueError,TypeError,KeyError,AttributeError,OverflowError) as exc:
            excluded.append({**provenance,"reason":str(exc)})

    buckets={}
    for row in accepted:
        buckets.setdefault((row["settings_id"],row["method"],row["seed"]),[]).append(row)
    duplicate_keys={k for k,v in buckets.items() if len(v)>1}
    for key in duplicate_keys:
        excluded.extend({**r,"reason":"duplicate same setting/method/seed; all copies excluded"} for r in buckets[key])
    accepted=[r for r in accepted if (r["settings_id"],r["method"],r["seed"]) not in duplicate_keys]
    baseline={(r["settings_id"],r["seed"]):r for r in accepted if r["method"]=="topk_ef"}
    paired=[]
    for row in accepted:
        reference=baseline.get((row["settings_id"],row["seed"]))
        if row["method"]=="topk_ef" or reference is None:
            continue
        contrast={"settings_id":row["settings_id"],"method":row["method"],"seed":row["seed"],
            "training_backend":row["training_backend"],"split_seed":row["split_seed"],
            "run_path":row["run_path"],"baseline_run_path":reference["run_path"],
            "result_sha256":row["result_sha256"],"baseline_result_sha256":reference["result_sha256"]}
        for name in METRICS:
            scale=100 if "accuracy" in name else 1
            contrast[name+"_difference"+("_pp" if scale==100 else "")]=(
                scale*(row[name]-reference[name]) if row[name] is not None and reference[name] is not None else None)
        paired.append(contrast)

    def group_rows(rows,metrics):
        grouped={}
        for row in rows:
            grouped.setdefault((row["settings_id"],row["method"]),[]).append(row)
        out=[]
        for key,bucket in sorted(grouped.items()):
            record={"settings_id":key[0],"method":key[1],"n_training_seeds":len(bucket),
                "training_backend":bucket[0]["training_backend"],"split_seed":bucket[0]["split_seed"],
                "seeds":",".join(str(r["seed"]) for r in sorted(bucket,key=lambda r:r["seed"])),
                "protocol_n3_met":len(bucket)==3}
            for name in metrics:
                for stat,value in stats(row.get(name) for row in bucket).items():
                    record[name+"_"+stat]=value
            out.append(record)
        return out
    groups=group_rows(accepted,METRICS)
    pair_metrics=sorted({k for r in paired for k in r if "_difference" in k})
    pair_groups=group_rows(paired,pair_metrics)
    output_path.mkdir(parents=True,exist_ok=True)
    write_csv(output_path/"per_run.csv",accepted,("settings_id","method","seed"))
    write_csv(output_path/"groups.csv",groups,("settings_id","method","n_training_seeds"))
    write_csv(output_path/"paired_vs_topk.csv",paired,("settings_id","method","seed"))
    write_csv(output_path/"paired_groups.csv",pair_groups,("settings_id","method","n_training_seeds"))
    write_csv(output_path/"excluded.csv",excluded,("run_id","reason"))
    meta={"allow_final_test":allow_final_test,"expected_rounds":expected_rounds,
          "accepted_run_count":len(accepted),"excluded_run_count":len(excluded),
          "independent_unit":"training seed; rounds/clients are not independent repetitions",
          "settings":identities,"summary_source_sha256":sha(Path(__file__).read_bytes()),
          "input_path":str(input_path.resolve()),"official_test_scope":"10,000 official examples, final model only" if allow_final_test else "not accessed",
          "analysis_policy":"No ranking or method selection; predefined Top-K EF paired contrasts only"}
    (output_path/"summary_metadata.json").write_text(json.dumps(meta,indent=2,allow_nan=False),encoding="utf-8")
    lines=["# 冻结方案确认实验汇总","",
        f"预设轮数：{expected_rounds}。接纳 {len(accepted)} 个完整 DONE 运行，排除 {len(excluded)} 个。",
        "官方最终测试指标已显式允许读取。" if allow_final_test else "未启用 --allow-final-test：不访问官方最终测试字段，仅汇总验证轨迹。","",
        "独立重复单位是训练随机种子。每个运行先求最后五轮验证均值，再跨种子统计。不同划分、后端和源码/运行身份不混池、不配对。",
        "n=3 只刻画同一固定划分上的训练随机性，不能代表数据划分不确定性；样本标准差使用 ddof=1。",
        "仅 n=3 时提供常规 df=2 的双侧 95% Student-t 区间；三个样本难以验证分布假设，区间可能较宽甚至超出准确率的自然范围。区间不裁剪，不默认宣称显著，不据此选算法。","",
        "| 设置 | 方法 | 后端 | n | 末轮官方测试准确率 % | 末轮官方测试 loss | 最后五轮验证准确率 % | 最后五轮验证 loss |",
        "|---|---|---|---:|---:|---:|---:|---:|"]
    def display(row,name,scale=1):
        value,sd=row.get(name+"_mean"),row.get(name+"_sample_sd")
        if value is None:return "—"
        return f"{scale*value:.4f}"+(f" ± {scale*sd:.4f}" if sd is not None else "（单种子）")
    for row in groups:
        vals=[row["settings_id"],row["method"],row["training_backend"],row["n_training_seeds"],
              display(row,"final_test_accuracy",100),display(row,"final_test_loss"),
              display(row,"last5_validation_accuracy",100),display(row,"last5_validation_loss")]
        lines.append("| "+" | ".join(map(str,vals))+" |")
    lines += ["","## 相对同种子 Top-K EF 的预设配对差","",
        "准确率差为百分点，正值有利；loss 差为候选减基线，负值有利。所有候选均按名称排列，不根据测试分数排名。","",
        "| 设置 | 方法 | n 对 | 官方测试准确率差 pp | 95% t 区间 pp（n=3） | 官方测试 loss 差 | 最后五轮验证准确率差 pp |",
        "|---|---|---:|---:|---:|---:|---:|"]
    for row in pair_groups:
        name="final_test_accuracy_difference_pp"
        lo,hi=row.get(name+"_t95_low"),row.get(name+"_t95_high")
        ci=f"[{lo:.4f}, {hi:.4f}]" if lo is not None else "—"
        vals=[row["settings_id"],row["method"],row["n_training_seeds"],display(row,name),ci,
              display(row,"final_test_loss_difference"),display(row,"last5_validation_accuracy_difference_pp")]
        lines.append("| "+" | ".join(map(str,vals))+" |")
    lines += ["","原始数值、每个运行的完整路径和 config/result SHA256 保存在 per_run.csv；全部区间在 groups.csv / paired_groups.csv。",
        "排除原因见 excluded.csv。源码哈希只提供来源追踪，不单独证明实验实际执行或数据真实性。","校验范围包括连续轮次、每轮指标和诊断有限性、同种子重复排除及签名一致性；本工具不替代原始数据/代码审计。",""]
    (output_path/"summary.md").write_text("\n".join(lines),encoding="utf-8")
    return meta


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--allow-final-test",action="store_true")
    p.add_argument("--expected-rounds",type=int,default=40)
    args=p.parse_args()
    if not args.input.is_dir() or args.expected_rounds<5:
        p.error("existing input directory and expected-rounds >=5 required")
    meta=summarize(args.input,args.output,allow_final_test=args.allow_final_test,expected_rounds=args.expected_rounds)
    print(json.dumps({k:v for k,v in meta.items() if k!="settings"},indent=2))


if __name__=="__main__":
    main()
