"""On-demand original GS renders and explicit deletion into isolated candidates."""
import argparse,fcntl,hashlib,json,os,re,time
from pathlib import Path
import cv2,numpy as np
from plyfile import PlyData,PlyElement
from .io import ROOT,write_json,sha256,rigid,output_path
from .postprocess import render_map
from .roi import contains

def safe_id(value):
    if not isinstance(value,str) or not re.fullmatch(r"[a-f0-9]{64}",value):raise ValueError("无效预览编号")
    return value

def render_request(run,payload):
    pose=rigid(payload.get("pose"))
    w=int(payload.get("width",640));h=int(payload.get("height",480));fx=float(payload.get("fx",500))
    if not 160<=w<=1024 or not 160<=h<=1024 or not np.isfinite(fx) or not 50<=fx<=4000:raise ValueError("渲染相机参数无效")
    preview_id=payload.get("preview_id")
    if preview_id:
        scope=json.loads((run/"region_previews"/safe_id(preview_id)/"scope.json").read_text())
        if scope["views_sha256"]!=sha256(run/"job/views.npz"):raise ValueError("选区已过期")
    data=dict(pose=pose.tolist(),width=w,height=h,fx=fx,preview_id=preview_id,
              base_sha256=sha256(json.loads((run/"config.json").read_text())["map_ply"]))
    if preview_id and scope["mode"]!="delete":
        from .spirula import asset
        reconstruction=asset(run)
        if reconstruction:data["reconstruction_key"]=reconstruction["key"]
    key=hashlib.sha256(json.dumps(data,sort_keys=True).encode()).hexdigest()
    return key,data

def deleted_mask(vertex,scope):
    mask=np.zeros(len(vertex),bool)
    for start in range(0,len(vertex),100000):
        v=vertex[start:start+100000];p=np.column_stack([v[n] for n in ("x","y","z")])
        mask[start:start+len(v)]=contains(p,scope)
    return mask

def write_filtered(source,scope,target):
    ply=PlyData.read(str(source),mmap=True);v=ply["vertex"].data;removed=deleted_mask(v,scope)
    if not removed.any():raise ValueError("框内没有可删除的原模型内容")
    if removed.all():raise ValueError("不能删除整个模型")
    PlyData([PlyElement.describe(v[~removed],"vertex")],text=False).write(str(target))
    xyz=np.column_stack([v[n][removed] for n in ("x","y","z")])
    return removed,xyz

def verify_render_proof(run,preview_id,render_id):
    path=run/"render_previews"/safe_id(render_id)
    status=json.loads((path/"status.json").read_text());request=json.loads((path/"request.json").read_text())
    if status.get("state")!="ready" or status.get("deleted_gaussians",0)<=0 or request.get("preview_id")!=preview_id:
        raise ValueError("删除范围已改变，请重新预览删除效果")
    cfg=json.loads((run/"config.json").read_text())
    if status["base_sha256"]!=sha256(cfg["map_ply"]):raise ValueError("原模型已变化，请重新预览")
    return status

def verify_candidate_proof(run,preview_id,render_id,reconstruction_key):
    path=run/"render_previews"/safe_id(render_id)
    state=json.loads((path/"status.json").read_text());data=json.loads((path/"request.json").read_text())
    if state.get("state")!="ready" or not state.get("candidate_preview") or data.get("preview_id")!=preview_id or data.get("reconstruction_key")!=reconstruction_key:
        raise ValueError("更新范围或重建已变化，请重新预览候选")
    cfg=json.loads((run/"config.json").read_text())
    if data["base_sha256"]!=sha256(cfg["map_ply"]):raise ValueError("原模型已变化，请重新预览")
    return state

def create_delete_candidate(source,scope,run):
    cfg=json.loads((run/"config.json").read_text());out=run/"polish";out.mkdir()
    removed,xyz=write_filtered(cfg["map_ply"],scope,out/"game_candidate.ply")
    target=run/"update/boxer_native_game_v1/guarded";target.mkdir(parents=True,exist_ok=True)
    ids=np.flatnonzero(~removed)
    np.savez_compressed(target/"lineage.npz",original_ids=ids,owner=np.full(len(ids),-1,dtype=np.int64))
    np.save(run/"manual_deleted_centers.npy",xyz)
    write_json(run/"deletion_audit.json",dict(deleted_gaussians=int(removed.sum()),surviving_rows_byte_identical=True,
        original_unchanged=True,navigation_policy="clear_only_with_multiview_full_height_free_evidence"))

def render_worker(run,key,gpu):
    out=run/"render_previews"/safe_id(key);candidate=out/"deletion_candidate.ply"
    lock=(ROOT/"runs/gpu.lock").open("a")
    try:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        cfg=json.loads((run/"config.json").read_text());data=json.loads((out/"request.json").read_text())
        before_hash=sha256(cfg["map_ply"])
        scope=None
        if data.get("preview_id"):scope=json.loads((run/"region_previews"/safe_id(data["preview_id"])/"scope.json").read_text())
        count=0;after_path=cfg["map_ply"]
        if scope and scope["mode"]=="delete":
            removed,_=write_filtered(cfg["map_ply"],scope,candidate);count=int(removed.sum());after_path=candidate
        merged=False
        if scope and scope["mode"]!="delete" and data.get("reconstruction_key"):
            from .spirula import fuse
            fuse(run,scope,candidate,expected_key=data["reconstruction_key"]);after_path=candidate;merged=True
        from vps.refinement.gaussian_renderer import GsplatGaussianRenderer
        renderer=GsplatGaussianRenderer("cuda")
        w,h,fx=data["width"],data["height"],data["fx"]
        row=dict(pose=np.asarray(data["pose"]),K=np.array([[fx,0,w/2],[0,fx,h/2],[0,0,1]]),depth=np.zeros((h,w)))
        for name,path in [("before.jpg",cfg["map_ply"]),("after.jpg",after_path)]:
            result=render_map(renderer,path,row)
            cv2.imwrite(str(out/name),cv2.cvtColor(result["rgb"],cv2.COLOR_RGB2BGR))
        if sha256(cfg["map_ply"])!=before_hash:raise ValueError("原模型已变化")
        write_json(out/"status.json",dict(state="ready",id=key,candidate_preview=merged,deleted_gaussians=count,base_sha256=before_hash,preview_id=data.get("preview_id")))
    except Exception as exc:
        write_json(out/"status.json",dict(state="failed",id=key,error="渲染暂不可用："+str(exc)))
        raise
    finally:
        lock.close()
        if candidate.exists():candidate.unlink()

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--run",type=Path,required=True);p.add_argument("--id",required=True);p.add_argument("--gpu",type=int,default=1)
    a=p.parse_args();render_worker(output_path(a.run),a.id,a.gpu)
