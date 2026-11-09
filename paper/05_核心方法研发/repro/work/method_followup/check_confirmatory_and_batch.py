"""Synthetic fixtures only. No real confirmatory run/test is read here."""

import contextlib
import csv
import io
import json
import math
from pathlib import Path
import sys
import tempfile

from summarize_confirmatory import summarize
from run_batch import run_queue


def main():
    with tempfile.TemporaryDirectory(prefix="confirmatory_synthetic_fixture_only_") as temp:
        root=Path(temp);runs=root/"SYNTHETIC_RUNS_NOT_RESEARCH";runs.mkdir()
        def put(name,method,seed,*,backend="mps",bad=None):
            directory=runs/name;directory.mkdir()
            config={"synthetic_fixture":True,"mode":"train","dataset":"SYNTHETIC_FIXTURE_NOT_REAL_DATA",
                "seed":seed,"variant":method,"rounds":40,"split_seed":101,"lr":.03,"local_epochs":1,
                "training_backend":backend,"test_final":True,
                "accelerated_adapter_source_sha256":"synthetic-code-identity",
                "accelerated_dependency_versions":{"torch":"synthetic version"},
                "data_manifest":{"partition_seed":101,"alpha":.1,"test_used_for_selection":False,
                                 "train_indices_sha256":"synthetic train identity"}}
            accuracy=.4+.02*(seed-1)+(.01 if method=="align" else 0)
            rows=[{"round":r,"validation":{"accuracy":accuracy-.1+.001*r,"loss":2.-.01*r},
                   "cumulative_bytes":r*100,"aggregate_relative_error":.3} for r in range(1,41)]
            result={"status":"DONE","seed":seed,"variant":method,"rows":rows,"client_rows":[{"round":1,"dummy":.2}],
                    "official_test_final":{"accuracy":accuracy,"loss":1.4-.1*(seed-1),"samples":10000}}
            if bad=="unfinished":result["status"]="IN_PROGRESS"
            if bad=="sequence":rows[-1]["round"]=39
            if bad=="nan_round":rows[3]["validation"]["loss"]=float("nan")
            if bad=="nan_client":result["client_rows"][0]["dummy"]=float("inf")
            if bad=="bad_test":result["official_test_final"]["loss"]=float("nan")
            (directory/"config.json").write_text(json.dumps(config))
            (directory/"result.json").write_text(json.dumps(result))
        for seed in (1,2,3):
            for method in ("topk_ef","align"):put(f"synthetic_{method}_{seed}",method,seed)
        put("synthetic_cpu_unpaired","align",1,backend="cpu")
        for seed,bad in enumerate(("unfinished","sequence","nan_round","nan_client","bad_test"),4):
            put("synthetic_invalid_"+bad,"invalid_"+bad,seed,bad=bad)
        val=summarize(runs,root/"val_only",allow_final_test=False)
        final=summarize(runs,root/"final_allowed",allow_final_test=True)
        assert val["accepted_run_count"]==8 and final["accepted_run_count"]==7
        assert final["excluded_run_count"]==5
        pairs=list(csv.DictReader((root/"final_allowed"/"paired_vs_topk.csv").open()))
        assert len(pairs)==3 and all(r["training_backend"]=="mps" for r in pairs)
        assert all(abs(float(r["final_test_accuracy_difference_pp"])-1)<1e-10 for r in pairs)
        groups=list(csv.DictReader((root/"final_allowed"/"groups.csv").open()))
        baseline=[r for r in groups if r["method"]=="topk_ef"][0]
        assert int(baseline["n_training_seeds"])==3
        assert abs(float(baseline["final_test_accuracy_mean"])-.42)<1e-12
        assert abs(float(baseline["final_test_accuracy_sample_sd"])-.02)<1e-12
        assert float(baseline["final_test_accuracy_t95_high"])-float(baseline["final_test_accuracy_t95_low"])>.09
        raw=list(csv.DictReader((root/"val_only"/"per_run.csv").open()))
        assert all(r["final_test_accuracy"]=="" for r in raw)
        assert all(len(r["config_sha256"])==len(r["result_sha256"])==64 for r in raw)

        work=root/"queue_cwd";work.mkdir()
        marker=root/"MUST_NOT_BE_CREATED"
        queue=root/"synthetic_queue.json"
        queue.write_text(json.dumps([
            {"id":"first","argv":[sys.executable,"-c","from pathlib import Path; print('SYNTHETIC FIRST',Path.cwd())"],"cwd":"queue_cwd"},
            {"id":"failure","command":[sys.executable,"-c","import sys;print('SYNTHETIC EXPECTED FAILURE');sys.exit(7)"]},
            {"id":"not_run","argv":[sys.executable,"-c",f"from pathlib import Path;Path({str(marker)!r}).write_text('should not run')"]}]))
        with contextlib.redirect_stdout(io.StringIO()):
            assert run_queue(queue,root/"dry_logs",dry_run=True)==0
            assert not (root/"dry_logs").exists()
            assert run_queue(queue,root/"fail_logs")==7
        status=json.loads((root/"fail_logs"/"batch_status.json").read_text())
        assert [r["status"] for r in status["jobs"]]==["DONE","FAILED","NOT_RUN_AFTER_STOP"]
        assert str(work) in (root/"fail_logs"/"001_first.log").read_text()
        assert not marker.exists()
        try:run_queue(queue,root/"fail_logs")
        except ValueError:pass
        else:raise AssertionError("nonempty logs must not resume/overwrite")
        good=root/"synthetic_success_queue.json"
        good.write_text(json.dumps([[sys.executable,"-c","print('SYNTHETIC SUCCESS')"]]))
        with contextlib.redirect_stdout(io.StringIO()):assert run_queue(good,root/"success_logs",default_cwd=work)==0
        assert json.loads((root/"success_logs"/"batch_status.json").read_text())["status"]=="DONE"
    report={"status":"PASS","fixture_type":"SYNTHETIC FIXTURE ONLY; TEMPORARY INPUTS REMOVED",
        "real_confirmatory_runs_read":False,"real_official_test_results_read":False,
        "checks":["40 continuous DONE finite rounds only","explicit final-test access gate",
            "seed-level sample SD and df2 wide t interval","same backend/source paired differences",
            "config/result paths and SHA provenance","failure-stop finite queue","cwd and logs",
            "dry-run no writes","nonempty log overwrite/resume refused","all-success queue"]}
    Path(__file__).with_name("confirmatory_and_batch_check.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))


if __name__=="__main__":main()
