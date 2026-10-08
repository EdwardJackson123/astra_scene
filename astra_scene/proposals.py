"""Boxer suggestions only: no detection is automatically committed."""
from __future__ import annotations
import argparse,fcntl,json,os,subprocess,time,uuid
from pathlib import Path
import cv2,numpy as np
from .io import ROOT,output_path,write_json
from .postprocess import load
from .roi import frame_manifest
from .pipeline import environment,run_stage,ENGINE,PHONE

ALIASES={"cart":"推车 / 金属架","trolley":"推车 / 金属架","metal frame":"推车 / 金属架","equipment cart":"推车 / 金属架",
 "chair":"椅子","office chair":"办公椅","table":"桌子","robot":"机器人","stool":"凳子","cabinet":"柜子","shelf":"架子",
 "box":"箱子","storage box":"收纳箱","partition":"隔断","monitor":"显示器","whiteboard":"白板","green board":"绿板","trash can":"垃圾桶"}
def category(label):
    if label in ("cart","trolley","metal frame","equipment cart"):return "cart"
    if label in ("chair","office chair"):return "chair"
    return label

def boxes_path(run):
    existing=run/"update/boxer_native_game_v1/boxer_boxes.json"
    if existing.is_file():
        data=json.loads(existing.read_text())
        if not data.get("manual_scope"):return existing
    pointer=run/"region_detection_pointer.json"
    if pointer.exists():
        path=Path(json.loads(pointer.read_text())["output"])
        if not path.resolve().is_relative_to(run.resolve()):raise ValueError("invalid_detection_path")
        target=path/"boxer_native_game_v1/boxer_boxes.json"
        if target.is_file():return target
    return None

def proposals(run,side="current"):
    if side not in ("current","old"):raise ValueError("invalid_proposal_side")
    path=boxes_path(run)
    if path is None:return dict(state="not_prepared",proposals=[])
    cfg,d,split,_=load(run);frames=frame_manifest(run)["frames"];groups=[]
    data=json.loads(path.read_text())
    for box in sorted(data.get(side,[]),key=lambda b:b.get("confidence3d",0),reverse=True):
        center=np.asarray(box["center"]);size=np.asarray(box["size"])
        if not np.isfinite(np.r_[center,size]).all() or (size<=0).any() or len(set(box.get("support_views",[])))<2:continue
        found=None
        for g in groups:
            a=g[0];ratio=size/np.maximum(a["size"],.01)
            if category(box["label"])==category(a["label"]) and np.linalg.norm(center-a["center"])<.5 and ((ratio>.4)&(ratio<2.5)).all():
                found=g;break
        if found is None:groups.append([box])
        else:found.append(box)
    out=run/"region_proposals";out.mkdir(exist_ok=True);result=[]
    for index,group in enumerate(groups):
        box=group[0];pose=np.asarray(box["pose"]);bounds=np.asarray(box["bounds"])
        corners=np.array([[x,y,z] for x in bounds[0:2] for y in bounds[2:4] for z in bounds[4:6]])
        corners=corners@pose[:3,:3].T+pose[:3,3]
        available=[]
        for frame in frames:
            i=frame["view"];p=(corners-d["poses"][i,:3,3])@d["poses"][i,:3,:3]
            if (p[:,2]<=.1).any():continue
            pix=p@d["intrinsics"][i].T;uv=pix[:,:2]/pix[:,2,None]
            low=uv.min(axis=0);high=uv.max(axis=0);crop=np.array(frame["crop"])
            raw_area=np.prod(np.maximum(0,high-low))
            low=np.maximum(low,crop[:2]);high=np.minimum(high,crop[2:])
            area=np.prod(np.maximum(0,high-low))
            if area<900 or area/max(raw_area,1)<.6:continue
            rect=np.r_[(low-crop[:2])/(crop[2:]-crop[:2]),(high-crop[:2])/(crop[2:]-crop[:2])]
            if not ((rect>=0)&(rect<=1)).all():continue
            available.append(dict(view=i,rect=rect.tolist(),area=float(area)))
        if len(available)<2:continue
        available.sort(key=lambda x:x["area"],reverse=True);a=available[0];center=np.asarray(box["center"])
        direction=d["poses"][a["view"],:3,3]-center;direction/=max(np.linalg.norm(direction),1e-8)
        alternatives=[]
        for b in available[1:]:
            delta=d["poses"][b["view"],:3,3]-d["poses"][a["view"],:3,3]
            q=d["poses"][b["view"],:3,3]-center;q/=max(np.linalg.norm(q),1e-8)
            angle=float(np.degrees(np.arccos(np.clip(q@direction,-1,1))))
            if np.linalg.norm(delta)>=.15:alternatives.append((min(angle,90)*np.sqrt(b["area"]),b,angle))
        if not alternatives:continue
        _,b,angle=max(alternatives,key=lambda x:x[0])
        proposal_id=(f"boxer_{index:03d}" if side=="current" else f"old_boxer_{index:03d}")
        frame=next(f for f in frames if f["view"]==a["view"])
        im=cv2.imread(str(run/"selection_frames"/frame["filename"]));h,w=im.shape[:2]
        rect=np.rint(np.asarray(a["rect"])*[w,h,w,h]).astype(int)
        cv2.rectangle(im,tuple(rect[:2]),tuple(rect[2:]),(20,220,100),3)
        cv2.imwrite(str(out/(proposal_id+".jpg")),im)
        result.append(dict(id=proposal_id,evidence_side=side,label=ALIASES.get(box["label"],box["label"]),source_label=box["label"],
            confidence=float(box.get("confidence3d",0)),support_views=len(set(j for g in group for j in g.get("support_views",[]))),
            selections=[dict(view=a["view"],rect=a["rect"]),dict(view=b["view"],rect=b["rect"])],
            bounds=[corners.min(axis=0).tolist(),corners.max(axis=0).tolist()],suggested_angle_deg=angle,thumbnail=proposal_id+".jpg"))
    return dict(state="ready",proposals=result,source=str(path),note="Boxer 仅提供建议；点选、调整并预览确认后才更新。")

def detect(run,out,gpu):
    out=output_path(out);out.mkdir(exist_ok=False);lock=(ROOT/"runs/gpu.lock").open("w")
    try:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        cfg=json.loads((run/"config.json").read_text());env=environment(gpu)
        free,total=map(int,subprocess.check_output(["nvidia-smi","--id",str(gpu),"--query-gpu=memory.free,memory.total","--format=csv,noheader,nounits"],text=True).strip().split(","))
        if free<20000:raise ValueError("GPU_needs_20000_MiB_free")
        env["VPS_GPU_MEMORY_FRACTION"]=str(min(.85,(free-2048)/total));gp=cfg["game_python"]
        for stage in ["render","boxer"]:
            run_stage(out,stage,[gp,PHONE/"run_update.py",run/"job",out,"--root",ENGINE,"--stage",stage,
                 "--boxer-assets",ROOT/"vendor/boxer_assets","--target-labels",cfg["target_labels"]],env)
        write_json(out/"status.json",dict(state="ready",updated=time.time()))
    except BaseException as exc:
        write_json(out/"status.json",dict(state="failed",error=str(exc),updated=time.time()));raise
    finally:lock.close()
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--run",required=True,type=Path);p.add_argument("--output",required=True,type=Path);p.add_argument("--gpu",type=int,default=1)
    a=p.parse_args();detect(output_path(a.run),a.output,a.gpu)
