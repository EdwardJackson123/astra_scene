"""Task presentation, persistent drafts and serialized candidate submissions."""
from __future__ import annotations
import argparse,fcntl,json,os,re,subprocess,time,uuid,math
from pathlib import Path
from .io import ROOT,write_json,sha256
from .pipeline import environment

ACTIVE={"queued","starting","running"}
def safe_name(name):
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,120}",name):raise ValueError("无效任务")
    return name
def read(path,default=None):
    try:return json.loads(Path(path).read_text())
    except (OSError,ValueError):return {} if default is None else default
def directory(root):
    p=Path(root)/"workflow";p.mkdir(exist_ok=True);return p
def task_path(root,name):return directory(root)/(safe_name(name)+".json")
def snapshot(root,name):
    root=Path(root);safe_name(name);run=root/"runs"/name;meta=read(task_path(root,name));state=read(run/"status.json")
    if not meta and not state:raise FileNotFoundError("任务不存在")
    if meta.get("state")=="queued":state={"state":"queued","stage":"queued"}
    elif meta.get("state")=="failed":state={"state":"failed","error":meta.get("error","处理未完成")}
    elif not state:state={"state":meta.get("state","starting"),"stage":"starting"}
    # Detect worker loss instead of leaving a permanently spinning task.
    if meta.get("state") in ACTIVE and meta.get("pid") and state.get("state") in ACTIVE:
        try:
            command=Path("/proc")/str(meta["pid"])/"cmdline"
            alive="astra_scene.workflow" in command.read_text() and name in command.read_text()
        except OSError:alive=False
        if not alive and meta.get("child_pid"):
            try:alive=str(root/"runs"/name) in (Path("/proc")/str(meta["child_pid"])/"cmdline").read_text()
            except OSError:pass
        if not alive:
            state={"state":"failed","error":"任务进程已中断。已保存原视频和选区，可重试。"}
    scope=read(run/"manual_scope.json")
    automatic_source=read(run/"automatic_source.json").get("source")
    parent=meta.get("source") or scope.get("source_run")
    kind=meta.get("kind") or ("candidate" if parent else "capture" if (run/"job/ready.json").exists() else "experiment")
    title=meta.get("title") or (({"add":"新增","replace":"替换","delete":"删除"}.get(scope.get("mode"),"更新")+" "+str(len(scope.get("regions",[scope])))+" 个区域") if parent else ("手机视频 · 已有采集" if name=="phone_fresh_001" else "Spirula · 重建效果对比" if name=="spirula_trial_001" else name))
    reconstruction=read(run/"reconstruction.json") or read(run/"reconstruction_used.json")
    backend=("Spirula SfM + 3DGS / SH "+str(reconstruction.get("output_sh_degree","待核对"))+" 阶" if reconstruction else meta.get("backend")) or ("Spirula / 原生 SfM + 3DGS" if name.startswith("spirula") else "VGGT + 位姿校正 / GaME 局部训练")
    if not reconstruction and name.endswith("_capture"):
        owner=read(task_path(root,name[:-8]))
        if owner.get("kind")=="automatic":
            backend="VGGT 定位与几何检查 → Spirula SfM + 3DGS（SH跟随原图）"
    busy=state.get("state") in ACTIVE
    quality=read(run/"artifacts/quality.json")
    if kind=="candidate" and quality.get("SH_degree") is not None:backend+=" / SH "+str(quality["SH_degree"])+" 阶"
    legacy_quality=bool(quality and quality.get("quality_policy_version",1)<2 and kind=="candidate" and scope.get("mode")!="delete")
    if legacy_quality and state.get("state")=="candidate_ready":state={**state,"state":"needs_review"}
    grid=quality.get("grid") or {}
    summary=dict(legacy=legacy_quality,policy_version=quality.get("quality_policy_version",1) if quality else None,
        visual_passed=quality.get("visual_gate_passed"),structural_passed=quality.get("structural_gate_passed"),
        grid_changed=grid.get("changed_cells"),grid_cleared=grid.get("cleared_columns"),floor_verified=(quality.get("floor") or {}).get("verified"),
        field_acceptance=quality.get("field_acceptance"))
    from .progress import details
    progress=details(root,name,state,kind)
    return {**state,"progress":progress,"id":name,"title":title,"kind":kind,"parent":parent,"backend":backend,
        "has_reconstruction":bool(reconstruction),"automatic_source":automatic_source,"created":meta.get("created",run.stat().st_mtime if run.exists() else time.time()),
        "can_select":kind=="capture" and not busy and state.get("state")!="failed" and (run/"job/ready.json").exists(),
        "has_report":not busy and (run/"artifacts/index.html").is_file(),"quality":summary,"can_retry":state.get("state")=="failed" and bool(meta.get("command")),
        "region_count":meta.get("region_count",len(scope.get("regions",[]))),"mode":meta.get("mode",scope.get("mode")),
        "experimental":meta.get("experimental",(name.startswith(("auto_","automatic_")) and name.endswith(("_capture","_candidate"))) or kind=="experiment" or bool(re.search(r"test|quality_|guard_",name+" "+str(parent or "")))),
        "draft":(run/"editor_draft.json").exists(),"published":False}
def all_tasks(root):
    ids={p.parent.name for p in (Path(root)/"runs").glob("*/status.json")}
    ids.update(p.stem for p in directory(root).glob("*.json"))
    out=[]
    for name in ids:
        try:out.append(snapshot(root,name))
        except (ValueError,FileNotFoundError):continue
    return sorted(out,key=lambda x:x["created"],reverse=True)
def enqueue(root,meta,command,dedupe,python):
    root=Path(root);folder=directory(root)
    with (folder/"submit.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        for p in folder.glob("*.json"):
            old=read(p)
            if old.get("dedupe")==dedupe:
                state=snapshot(root,p.stem)
                if state["state"]!="failed":return {"id":p.stem,"reused":True}
        name=meta["id"];meta={**meta,"command":command,"dedupe":dedupe,"state":"queued","created":time.time()}
        write_json(task_path(root,name),meta)
        (root/"logs").mkdir(exist_ok=True)
        try:
            with (root/"logs"/(name+"_worker.log")).open("w") as log:
                proc=subprocess.Popen([python,"-u","-m","astra_scene.workflow","--task",name],cwd=root,
                    env=environment(os.environ.get("ASTRA_GPU","1")),stdin=subprocess.DEVNULL,stdout=log,
                    stderr=subprocess.STDOUT,start_new_session=True)
            meta["pid"]=proc.pid;write_json(task_path(root,name),meta)
        except Exception as e:
            meta.update(state="failed",error=str(e));write_json(task_path(root,name),meta);raise
    return {"id":name,"reused":False}
def execute(name,root=ROOT):
    path=task_path(root,name);folder=directory(root)
    # Parent records PID while holding submit.lock; wait before reading its metadata.
    with (folder/"submit.lock").open("a") as lock:fcntl.flock(lock,fcntl.LOCK_EX);meta=read(path)
    with (folder/"worker.lock").open("a") as serial:
        fcntl.flock(serial,fcntl.LOCK_EX)
        meta.update(state="starting");write_json(path,meta)
        env=environment(os.environ.get("ASTRA_GPU","1"));env["ASTRA_WAIT_GPU"]="1"
        try:
            with (Path(root)/"logs"/(name+".log")).open("w") as log:
                child=subprocess.Popen(meta["command"],cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                meta["child_pid"]=child.pid;write_json(path,meta)
                rc=child.wait()
            actual=read(Path(root)/"runs"/name/"status.json")
            if rc or actual.get("state") in ACTIVE or not actual:
                raise RuntimeError(actual.get("error","处理进程未完整结束，请查看诊断信息"))
            meta["state"]="finished"
        except Exception as e:meta.update(state="failed",error=str(e))
        write_json(path,meta)
def draft_key(source):
    p=source/"job/views.npz";s=p.stat()
    return str(s.st_size)+":"+str(s.st_mtime_ns)
def save_draft(source,payload):
    if len(json.dumps(payload,allow_nan=False))>200000:raise ValueError("草稿太大")
    state=payload.get("workspace",{});regions=state.get("regions",[])
    if not isinstance(regions,list) or len(regions)>8:raise ValueError("最多 8 个区域")
    ids=[]
    for row in regions:
        ids.append(row["id"])
        for field,n in [("center",3),("size",3),("quaternion_xyzw",4)]:
            vals=row.get(field)
            if not isinstance(vals,list) or len(vals)!=n or not all(isinstance(x,(int,float)) and math.isfinite(x) for x in vals):raise ValueError("选区数据无效")
        if any(x<.03 or x>4 for x in row["size"]) or math.prod(row["size"])>12:raise ValueError("选区尺寸超出范围")
        if abs(sum(x*x for x in row["quaternion_xyzw"])-1)>.01:raise ValueError("旋转数据无效")
    if len(set(ids))!=len(ids) or (state.get("activeId") is not None and state["activeId"] not in ids):raise ValueError("选区编号无效")
    if state.get("mode") not in ("add","replace","delete"):raise ValueError("操作类型无效")
    with (source/"editor_draft.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        old=read(source/"editor_draft.json")
        if payload.get("revision",0)!=old.get("revision",0):raise FileExistsError("其他页面已更新草稿。请刷新后继续，当前修改已保留在本机。")
        out=dict(revision=old.get("revision",0)+1,workspace=state,geometry_key=draft_key(source),updated=time.time())
        write_json(source/"editor_draft.json",out);return out
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--task",required=True);execute(p.parse_args().task)
