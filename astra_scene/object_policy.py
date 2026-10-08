"""Object-level candidate policy. Training-view ablation never authorizes deployment."""
import argparse,copy,fcntl,json
from pathlib import Path
import numpy as np
from .io import ROOT,write_json,independent

def accept_trial(records):
    useful=[r for r in records if r["changed_pixels"]>=100]
    gains=[r["before_l1"]-r["after_l1"] for r in useful]
    return bool(len(gains)>=3 and np.mean(gains)>.003 and min(gains)>=-.015 and np.mean(np.array(gains)>0)>=.6)

def chair_bounds(bounds):
    b=np.asarray(bounds,float).copy()
    b[0,:2]-=.1;b[1,:2]+=.1;b[1,2]+=.1
    # Keep the detected bottom, not a scene-specific world floor coordinate.
    return b.tolist()

def run(source,output):
    from .postprocess import load,rows_from_data,render_map
    from .roi import preview,frame_manifest
    from .spirula import fuse
    source=Path(source);output=Path(output);out=output/"artifacts/object_policy";out.mkdir(parents=True,exist_ok=True)
    cfg,d,split,_=load(source);rows=rows_from_data(d)
    completion=json.loads((output/"artifacts/completion/report.json").read_text())
    proposals=json.loads((output/"analysis/evidence.json").read_text())["proposals"]
    advice={p["id"]:p for p in json.loads((output/"analysis/ai.json").read_text())["items"]}
    regs=copy.deepcopy(completion["regions"]);decisions=[]
    if not regs:
        write_json(out/"report.json",dict(regions=[],decisions=[],preview_only=True));return
    positions=independent(rows,split["train_views"],.15)
    ids=[positions[i] for i in np.unique(np.linspace(0,len(positions)-1,min(8,len(positions))).astype(int))]
    crops={x["view"]:x["crop"] for x in frame_manifest(source)["frames"]}
    with (ROOT/"runs/gpu.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        from vps.refinement.gaussian_renderer import GsplatGaussianRenderer
        def renders(path):
            renderer=GsplatGaussianRenderer("cuda")
            result={i:render_map(renderer,path,rows[i])["rgb"] for i in ids}
            del renderer
            import torch
            torch.cuda.empty_cache()
            return result
        current=renders(output/"artifacts/completion/scene_preview.ply")
        number=0
        def trial(candidate,label):
            nonlocal current,regs,number
            number+=1;path=out/f"trial_{number:03d}.ply"
            fuse(source,dict(mode="replace",regions=candidate),path)
            images=renders(path);records=[]
            for i in ids:
                x0,y0,x1,y1=crops[i];a=current[i][y0:y1,x0:x1].astype(float)/255;b=images[i][y0:y1,x0:x1].astype(float)/255
                truth=d["rgb"][i,y0:y1,x0:x1]
                changed=np.abs(a-b).max(2)>2/255
                records.append(dict(view=i,changed_pixels=int(changed.sum()),before_l1=float(np.abs(a-truth)[changed].mean()) if changed.any() else 0,after_l1=float(np.abs(b-truth)[changed].mean()) if changed.any() else 0))
            accepted=accept_trial(records)
            decisions.append(dict(id=label,accepted=accepted,views=records,identity_verified=False))
            if accepted:regs=candidate;current=images
            path.unlink()
            write_json(out/"progress.json",dict(trials=number,last=label,accepted=accepted))
            print("OBJECT_POLICY",label,accepted,flush=True)
        # Only test retracting expanded components, never shrink the original detected object.
        for group in completion["groups"]:
            if not group.get("included") or not group.get("search_boundary_reached"):continue
            candidate=copy.deepcopy(regs)
            target=next((r for r in candidate if r["id"]==group["id"]),None)
            if target is None:continue
            target["bounds"]=group["original_bounds"]
            trial(candidate,group["id"]+"_bounded")
        # Old-position replacement covers movable equipment as well as chairs.
        # These are replacement hypotheses, not AI-authorized obstacle deletion.
        for p in proposals:
            if p.get("evidence_side")!="old" or p.get("source_label","") not in {"chair","office chair","stool","robot","easel","metal frame","equipment cart","cart","trolley","table","box","storage box"}:continue
            a=advice.get(p["id"],{})
            if a.get("decision")=="deprioritize" or a.get("ai_validation_failed"):continue
            reg=dict(id=p["id"],label="旧物体位置检查",bounds=chair_bounds(p["bounds"]),mode="replace")
            trial(copy.deepcopy(regs)+[reg],p["id"])
    selected=[];blocked={}
    for reg in regs:
        try:
            check=preview(source,reg)
            if not check["coverage"]["ready"]:blocked[reg["id"]]=check["coverage"]["message"]
        except ValueError as exc:blocked[reg["id"]]=str(exc)
    selected=regs
    report=dict(regions=selected,blocked=blocked,decisions=decisions,train_views=ids,preview_only=True,deployable=False,
        method="whole_object_completion_with_training_view_ablations_and_old_position_replacement",
        independent_validation=False,original_modified=False,retrained=False)
    write_json(out/"report.json",report)
    page='<meta charset="utf-8"><h1>自动整物体范围选择</h1><p>训练视角用于范围比较；正式候选随后独立执行模型、栅格和定位图库检查。这里不代表验收通过。</p>'
    for decision in decisions:page+=f'<p>{decision["id"]}: '+("采用" if decision["accepted"] else "未采用")+'</p>'
    page+='<p>进入候选 '+str(len(selected))+' 个范围；覆盖不足 '+str(len(blocked))+' 个。</p><a href="report.json">详细审计</a>'
    (out/"index.html").write_text(page)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--source",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args();run(a.source,a.output)
