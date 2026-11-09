"""Run one explicit finite JSON command queue, in order, stopping on failure.

No scheduling, recurrence, algorithm selection, flag rewriting or automatic
resume. Commands must be argv arrays and execute with shell=False. Each job
gets a combined stdout/stderr log plus atomic batch state and JSONL events.
"""

from __future__ import annotations

import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time


class Interrupted(Exception):
    def __init__(self,signum):self.signum=signum


def stamp():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path,value):
    tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False),encoding="utf-8")
    tmp.replace(path)


def parse_jobs(queue_path,default_cwd=None):
    queue_path=Path(queue_path).resolve()
    raw=queue_path.read_bytes()
    commands=json.loads(raw)
    if not isinstance(commands,list) or not commands:
        raise ValueError("Queue must be a nonempty finite JSON list")
    jobs=[];ids=set()
    fallback=Path(default_cwd).expanduser().resolve() if default_cwd else queue_path.parent
    for index,item in enumerate(commands,1):
        if isinstance(item,list):
            item={"argv":item}
        if not isinstance(item,dict):raise ValueError("Each item must be an argv array or an object")
        command=item.get("argv",item.get("command"))
        if not isinstance(command,list) or not command or any(not isinstance(x,str) or "\0" in x for x in command):
            raise ValueError("Every command must be a nonempty argv array of strings; shell command strings are unsupported")
        identifier=item.get("id",f"job_{index:03d}")
        if not isinstance(identifier,str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}",identifier) or identifier in ids:
            raise ValueError("IDs must be unique short filenames using letters/digits/._-")
        ids.add(identifier)
        cwd=Path(item["cwd"]).expanduser() if item.get("cwd") else fallback
        if not cwd.is_absolute():cwd=queue_path.parent/cwd
        cwd=cwd.resolve()
        if not cwd.is_dir():raise ValueError(f"Job {identifier} cwd does not exist: {cwd}")
        env=item.get("env",{})
        if not isinstance(env,dict) or any(not isinstance(k,str) or not isinstance(v,str) or not k or "=" in k or "\0" in k+v for k,v in env.items()):
            raise ValueError("Optional env must map valid string keys to string values")
        jobs.append({"id":identifier,"argv":command,"cwd":str(cwd),"env":env,
                     "log_name":f"{index:03d}_{identifier}.log","status":"PENDING"})
    return jobs,hashlib.sha256(raw).hexdigest()


def run_queue(queue_path,logs,*,default_cwd=None,dry_run=False):
    queue_path=Path(queue_path).resolve();logs=Path(logs).expanduser().resolve()
    jobs,source_sha=parse_jobs(queue_path,default_cwd)
    public_jobs=[{k:v for k,v in job.items() if k!="env"} | {"environment_keys":sorted(job["env"])} for job in jobs]
    if dry_run:
        print(json.dumps({"dry_run":True,"queue_path":str(queue_path),"queue_sha256":source_sha,
                          "logs":str(logs),"jobs":public_jobs},indent=2,ensure_ascii=False))
        return 0
    if logs.exists() and any(logs.iterdir()):
        raise ValueError(f"Logs directory is not empty; refusing overwrite or implicit resume: {logs}")
    logs.mkdir(parents=True,exist_ok=True)
    manifest={"queue_path":str(queue_path),"queue_sha256":source_sha,"created_utc":stamp(),
              "runner_source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "jobs":public_jobs,"policy":"finite ordered queue; stop on failure; exact argv; no scheduler/selection/resume"}
    # Exclusive creation prevents concurrent runners from claiming one log dir.
    with (logs/"batch_manifest.json").open("x",encoding="utf-8") as f:
        json.dump(manifest,f,indent=2,ensure_ascii=False)
    state={"status":"RUNNING","started_utc":stamp(),"jobs":public_jobs}
    state_path=logs/"batch_status.json"
    atomic_json(state_path,state)
    def event(value):
        with (logs/"events.jsonl").open("a",encoding="utf-8") as f:
            f.write(json.dumps({"utc":stamp(),**value},ensure_ascii=False)+"\n")
    def interrupt(signum,frame):
        raise Interrupted(signum)
    previous={s:signal.getsignal(s) for s in (signal.SIGINT,signal.SIGTERM)}
    for s in previous:signal.signal(s,interrupt)
    exit_code=0
    proc=None
    def stop_process(process):
        if process is None or process.poll() is not None:return
        try:os.killpg(process.pid,signal.SIGTERM)
        except ProcessLookupError:return
        try:process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:os.killpg(process.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            process.wait()
    try:
        for index,(private,job) in enumerate(zip(jobs,public_jobs)):
            proc=None;began=time.monotonic()
            job.update({"status":"RUNNING","started_utc":stamp(),"log_path":str(logs/private["log_name"])})
            atomic_json(state_path,state);event({"id":job["id"],"status":"RUNNING"})
            try:
                with (logs/private["log_name"]).open("x",encoding="utf-8") as output:
                    proc=subprocess.Popen(private["argv"],cwd=private["cwd"],
                        env={**os.environ,**private["env"]},stdin=subprocess.DEVNULL,
                        stdout=output,stderr=subprocess.STDOUT,shell=False,start_new_session=True)
                    code=proc.wait()
                job["return_code"]=code
                job["status"]="DONE" if code==0 else "FAILED"
                if code!=0:exit_code=code if code>0 else 128-code
            except (Interrupted,KeyboardInterrupt) as exc:
                stop_process(proc)
                signum=exc.signum if isinstance(exc,Interrupted) else signal.SIGINT
                exit_code=128+signum
                job.update({"status":"INTERRUPTED","return_code":exit_code})
            except (OSError,ValueError) as exc:
                exit_code=127
                job.update({"status":"FAILED","return_code":exit_code,"error":str(exc)})
            job.update({"ended_utc":stamp(),"elapsed_seconds":time.monotonic()-began})
            event({"id":job["id"],"status":job["status"],"return_code":job["return_code"]})
            if exit_code:
                for remaining in public_jobs[index+1:]:remaining["status"]="NOT_RUN_AFTER_STOP"
                state["status"]="INTERRUPTED" if job["status"]=="INTERRUPTED" else "FAILED"
                atomic_json(state_path,state)
                break
            atomic_json(state_path,state)
        if exit_code==0:state["status"]="DONE"
    except (Interrupted,KeyboardInterrupt) as exc:
        stop_process(proc)
        signum=exc.signum if isinstance(exc,Interrupted) else signal.SIGINT
        exit_code=128+signum
        state["status"]="INTERRUPTED"
        for job in public_jobs:
            if job["status"]=="RUNNING":job["status"]="INTERRUPTED"
            elif job["status"]=="PENDING":job["status"]="NOT_RUN_AFTER_STOP"
        event({"status":"INTERRUPTED","exit_code":exit_code})
    finally:
        for s,handler in previous.items():signal.signal(s,handler)
        state["ended_utc"]=stamp()
        state["exit_code"]=exit_code
        atomic_json(state_path,state)
    print(json.dumps({"status":state["status"],"exit_code":exit_code,
                      "status_path":str(state_path)},ensure_ascii=False),flush=True)
    return exit_code


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--queue",type=Path,required=True)
    p.add_argument("--logs",type=Path,required=True)
    p.add_argument("--cwd",type=Path)
    p.add_argument("--dry-run",action="store_true")
    args=p.parse_args()
    try:code=run_queue(args.queue,args.logs,default_cwd=args.cwd,dry_run=args.dry_run)
    except (OSError,ValueError) as exc:p.error(str(exc))
    raise SystemExit(code)


if __name__=="__main__":
    main()
