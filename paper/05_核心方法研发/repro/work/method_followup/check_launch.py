"""Path-routing checks only: no training, real image reads, or Desktop writes."""

import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import launch


def main():
    with tempfile.TemporaryDirectory(prefix="fl_launcher_check_only_") as tmp:
        base=Path(tmp)
        def marker(path):
            (path/"tools").mkdir(parents=True)
            (path/"dataset").mkdir()
            (path/"tools"/"fedavg_update_diagnostic.py").write_text("# marker only, never imported\n")
            return path.resolve()
        a=marker(base/"first")
        b=marker(base/"second")
        assert launch.resolve_project(a,environment={"FL_PROJECT_ROOT":str(b)})[0]==a
        assert launch.resolve_project(environment={"FL_PROJECT_ROOT":str(b)},here=base)[0]==b
        for explicit,environment in ((base/"absent",{"FL_PROJECT_ROOT":str(b)}),
                                     (None,{"FL_PROJECT_ROOT":str(base/"absent")})):
            try:
                launch.resolve_project(explicit,environment=environment,here=base)
            except ValueError:
                pass
            else:
                raise AssertionError("invalid explicit/environment paths must not silently fall back")
        exported=base/"paper"/"06_new_material"/"scripts";exported.mkdir(parents=True)
        relative=marker(base/"paper"/"02_实验工程")
        assert launch.resolve_project(environment={},here=exported)[0]==relative
        forwarded=["--variants","topk_ef,align90","--out",str(base/"never_created"),"--rounds","20"]
        capture=io.StringIO()
        with contextlib.redirect_stdout(capture):
            info=launch.main(["--project-root",str(a),"--dry-run","--entry","run_controls","--",*forwarded])
        assert info["forwarded_args"]==forwarded
        assert not (base/"never_created").exists()
        assert "torch" not in sys.modules and "fedavg_update_diagnostic" not in sys.modules

    actual=(launch.HERE.parent/"tinyproto_diag").resolve()
    environment=dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"]="1"
    checked=[]
    for entry in launch.ENTRIES:
        completed=subprocess.run([sys.executable,str(launch.HERE/"launch.py"),
            "--project-root",str(actual),"--entry",entry,"--","--help"],
            capture_output=True,text=True,env=environment,check=True)
        info,_=json.JSONDecoder().raw_decode(completed.stdout)
        assert info["verified_loader_root"]==str(actual)
        assert info["forwarded_args"]==["--help"]
        if entry=="run_partitions":
            assert info["partition_raw_directory"]==str(actual/"dataset"/"Cifar10"/"rawdata")
        checked.append(entry)
    report={"status":"PASS","entry_help_routes_verified":checked,
        "checks":["explicit path overrides environment", "invalid supplied path has no silent fallback",
                  "environment root", "bounded ../../02_实验工程 export layout",
                  "forwarded training args preserved", "dry-run creates no output and imports no torch/data helper",
                  "actual helper ROOT and model/loader import identities checked for all entrypoints",
                  "partition RAW bound to selected project"],
        "scope":"No training or real image data loaded; no Desktop writes."}
    Path(__file__).with_name("launch_check.json").write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf-8")
    print(json.dumps(report,indent=2,ensure_ascii=False))


if __name__=="__main__":
    main()
