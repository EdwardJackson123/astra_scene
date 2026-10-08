"""User-defined 3D scope from two or more training-view rectangles."""
from __future__ import annotations
import os
import argparse,fcntl,hashlib,json,math,os,shutil,time,uuid
from pathlib import Path
import cv2,numpy as np
from .io import ROOT,output_path,write_json,sha256,independent
from .geometry import backproject,project,evidence,voxel_fuse
from .postprocess import load,rows_from_data
from .pipeline import environment,run_stage,ENGINE,PHONE

def box_geometry(scope):
    """Return center, local dimensions and local-to-world rotation, including legacy AABBs."""
    obb=scope.get("oriented_box")
    if obb:
        from scipy.spatial.transform import Rotation
        return np.asarray(obb["center"],float),np.asarray(obb["size"],float),Rotation.from_quat(obb["quaternion_xyzw"]).as_matrix()
    b=np.asarray(scope["bounds"],float)
    return (b[0]+b[1])/2,b[1]-b[0],np.eye(3)

def box_corners(scope):
    center,size,rotation=box_geometry(scope)
    signs=np.array([[x,y,z] for x in (-.5,.5) for y in (-.5,.5) for z in (-.5,.5)])
    return (signs*size)@rotation.T+center

def contains(xyz,scope):
    if "regions" in scope:
        return np.logical_or.reduce([contains(xyz,s) for s in scope["regions"]])
    center,size,rotation=box_geometry(scope)
    local=(np.asarray(xyz)-center)@rotation
    return (np.abs(local)<=size/2+1e-5).all(axis=1)

def validate_oriented_box(payload):
    obb=payload.get("oriented_box")
    if not isinstance(obb,dict):raise ValueError("三维框参数无效")
    center=np.asarray(obb.get("center"),float);size=np.asarray(obb.get("size"),float)
    quat=np.asarray(obb.get("quaternion_xyzw"),float)
    if center.shape!=(3,) or size.shape!=(3,) or quat.shape!=(4,) or not np.isfinite(np.r_[center,size,quat]).all():
        raise ValueError("位置、尺寸和旋转必须是有限数值")
    if (size<.03).any() or (size>4).any() or np.prod(size)>12:
        raise ValueError("每边须为 0.03–4 米，总体积不超过 12 立方米")
    if abs(np.linalg.norm(quat)-1)>1e-4:raise ValueError("旋转参数不是有效的单位四元数")
    if payload.get("mode","add") not in ("add","replace","delete"):raise ValueError("未知更新模式")
    return dict(center=center.tolist(),size=size.tolist(),quaternion_xyzw=(quat/np.linalg.norm(quat)).tolist())

def scope_mask(row,scope):
    if "regions" in scope:
        return np.logical_or.reduce([scope_mask(row,s) for s in scope["regions"]])
    # Conservative convex projection of the actual oriented box.
    corners=box_corners(scope)
    camera=(corners-row["pose"][:3,3])@row["pose"][:3,:3]
    if np.any(camera[:,2]<=.05):return np.ones(row["depth"].shape,bool)
    pix=camera@row["K"].T;uv=pix[:,:2]/pix[:,2,None];h,w=row["depth"].shape
    a=np.floor(uv.min(axis=0)).astype(int);b=np.ceil(uv.max(axis=0)).astype(int)
    mask=np.zeros((h,w),bool)
    x0,y0=np.clip(a,[0,0],[w,h]);x1,y1=np.clip(b+1,[0,0],[w,h])
    if x1>x0 and y1>y0:
        # Convex projected box, not the larger world-axis enclosure.
        hull=cv2.convexHull(np.rint(np.clip(uv,-1e7,1e7)).astype(np.int32))
        cv2.fillConvexPoly(mask.view(np.uint8),hull,1)
    return mask

def frame_manifest(run):
    cfg,d,split,report=load(run);path=run/"selection_frames";path.mkdir(exist_ok=True)
    capture=json.loads(Path(report["capture"]).read_text())
    from PIL import Image
    from vps.dynamic.image_mask import content_mask
    records=[]
    for i in split["train_views"]:
        h,w=d["depths"][i].shape
        with Image.open(capture["frames"][i]["image"]) as im:mask=content_mask(*im.size,h,w)
        y,x=np.where(mask);crop=[int(x.min()),int(y.min()),int(x.max()+1),int(y.max()+1)]
        target=path/f"frame_{i:03d}.jpg"
        if not target.exists():
            image=np.uint8(np.clip(d["rgb"][i,crop[1]:crop[3],crop[0]:crop[2]]*255,0,255))
            cv2.imwrite(str(target),cv2.cvtColor(image,cv2.COLOR_RGB2BGR))
        records.append(dict(view=i,crop=crop,width=crop[2]-crop[0],height=crop[3]-crop[1],filename=target.name,
                            camera_position=d["poses"][i,:3,3].tolist()))
    return dict(frames=records,total_views=len(d["poses"]),registered_training_views=len(records),
                note="只能用训练视角定义选区；测试视角不用于选区生成。")

def validate_selections(payload,frames):
    selections=payload.get("selections")
    if not isinstance(selections,list) or not 2<=len(selections)<=4:raise ValueError("请在至少两个不同视角框选同一目标")
    known={r["view"]:r for r in frames};result=[];seen=set()
    for s in selections:
        i=s.get("view");rect=np.asarray(s.get("rect"),float)
        if i not in known or i in seen:raise ValueError("请选择不同的有效训练视角")
        if rect.shape!=(4,) or not np.isfinite(rect).all() or (rect<0).any() or (rect>1).any() or rect[2]-rect[0]<.02 or rect[3]-rect[1]<.02:
            raise ValueError("选框过小或坐标无效")
        seen.add(i);f=known[i];x0,y0,x1,y1=f["crop"]
        pixels=rect*np.array([x1-x0,y1-y0,x1-x0,y1-y0])+[x0,y0,x0,y0]
        result.append(dict(view=int(i),rect=rect.tolist(),pixels=pixels.tolist()))
    padding=float(payload.get("padding_m",.02))
    if not np.isfinite(padding) or not 0<=padding<=.15:raise ValueError("边界余量必须在 0–15 厘米")
    if payload.get("mode","add") not in ("add","replace","delete"):raise ValueError("未知更新模式")
    return result,padding

def validate_bounds(payload):
    bounds=np.asarray(payload.get("bounds"),float)
    if bounds.shape!=(2,3) or not np.isfinite(bounds).all():
        raise ValueError("三维范围坐标无效")
    size=bounds[1]-bounds[0]
    if (size<.03).any() or (size>4).any() or np.prod(size)>12:
        raise ValueError("请把范围缩小到一个物体：每边 3 厘米至 4 米，总体积不超过 12 立方米")
    if payload.get("mode","add") not in ("add","replace","delete"):raise ValueError("未知更新模式")
    return bounds

def preview(run,payload):
    # Browser cancellation does not stop Flask work; serialize identical/revised previews.
    with (run/"region_preview.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        return _preview_regions(run,payload) if "regions" in payload else _preview(run,payload)

def preview_context(run):
    cfg,d,split,_=load(run)
    return cfg,d,split,rows_from_data(d),frame_manifest(run)["frames"]

def _preview_regions(run,payload):
    requested=payload.get("regions")
    if not isinstance(requested,list) or not 1<=len(requested)<=32:raise ValueError("请选择 1–32 个区域")
    mode=payload.get("mode","add")
    if mode not in ("add","replace","delete"):raise ValueError("未知更新模式")
    context=preview_context(run);regions=[]
    for index,item in enumerate(requested):
        if not isinstance(item,dict):raise ValueError("选区参数无效")
        try:regions.append(_preview(run,{**item,"mode":mode},context))
        except (ValueError,TypeError,KeyError) as exc:
            raise ValueError("区域 "+str(index+1)+"（"+str(item.get("label","未命名"))[:64]+"）："+str(exc)) from exc
    ready=all(s["coverage"]["ready"] for s in regions)
    supports=sorted(set(i for s in regions for i in s["coverage"]["supported_views"]))
    bounds=np.asarray([s["bounds"] for s in regions])
    scope=dict(schema="astra.manual_regions.v3",regions=regions,bounds=[bounds[:,0].min(0).tolist(),bounds[:,1].max(0).tolist()],
        mode=mode,label=str(len(regions))+" 个区域",manual_scope=True,margin_m=0,padding_m=0,source_run=run.name,
        views_sha256=regions[0]["views_sha256"],selection_method="3d_multiple_oriented_boxes",
        selections=regions[0]["selections"],
        coverage=dict(ready=ready,supported_views=supports,points=sum(s["coverage"]["points"] for s in regions),
            max_angle_deg=min(s["coverage"]["max_angle_deg"] for s in regions),unseen_surfaces_verified=False,
            message=str(sum(s["coverage"]["ready"] for s in regions))+"/"+str(len(regions))+" 个区域通过覆盖检查",
            direction_hint="；".join(s["label"]+"："+s["coverage"]["direction_hint"] for s in regions)))
    _,d,_,rows,frames=context
    return write_preview(run,scope,d,rows,frames)

def preview_deletion(run,payload,context):
    from plyfile import PlyData
    cfg,d,split,rows,frames=context
    if "oriented_box" in payload:region=dict(oriented_box=validate_oriented_box(payload))
    else:region=dict(bounds=validate_bounds(payload).tolist())
    corners=box_corners(region)
    vertex=PlyData.read(cfg["map_ply"],mmap=True)["vertex"].data
    count=0;sample=[]
    for start in range(0,len(vertex),100000):
        v=vertex[start:start+100000];xyz=np.column_stack([v[k] for k in ("x","y","z")])
        inside=contains(xyz,region);count+=int(inside.sum());sample.append(xyz[inside][::10])
    if count==0:raise ValueError("框内没有原模型内容，请显示原场景后重新选择")
    xyz=np.concatenate(sample);xyz=xyz[::max(1,len(xyz)//30000)]
    views=[]
    for f in frames:
        ix,_,_,_=project(xyz,rows[f["view"]])
        if len(ix)>=10:views.append(f["view"])
    if not views:raise ValueError("这块原模型没有可用的拍摄方向，请换一个区域")
    scope=dict(schema="astra.manual_region.v2",**region,
        region_id=str(payload.get("id",""))[:64],mode="delete",label=str(payload.get("label","删除区域"))[:64],
        padding_m=0,margin_m=0,manual_scope=True,source_run=run.name,views_sha256=sha256(run/"job/views.npz"),
        selection_method="manual_delete_oriented_box",selections=[dict(view=views[0])],
        coverage=dict(ready=True,supported_views=views,max_angle_deg=0,points=count,unseen_surfaces_verified=False,
            message="选中 "+str(count)+" 个原模型高斯；请先核对删除预览",
            direction_hint="删除将联动生成模型、图库和栅格候选。是否可开放通行，按现场视频的腾空证据另行检查。"))
    scope["bounds"]=[corners.min(0).tolist(),corners.max(0).tolist()]
    return write_preview(run,scope,d,rows,frames)

def _preview(run,payload,context=None):
    cfg,d,split,rows,frames=context if context is not None else preview_context(run)
    if payload.get("mode")=="delete":
        return preview_deletion(run,payload,(cfg,d,split,rows,frames))
    oriented="oriented_box" in payload
    direct=oriented or "bounds" in payload
    region=None
    if direct:
        if oriented:
            obb=validate_oriented_box(payload);region=dict(oriented_box=obb)
            corners=box_corners(region);low=corners.min(axis=0);high=corners.max(axis=0)
        else:low,high=validate_bounds(payload);region=dict(bounds=[low,high])
        padding=0.
        selections=[dict(view=int(split["train_views"][0]))]
    else:
        selections,padding=validate_selections(payload,frames)
        if len(independent(rows,[s["view"] for s in selections],.15))<2:
            raise ValueError("所选相机位置太近，请换到另一个方向的视角")
    selected=[s["view"] for s in selections]
    pts=[];colors=[]
    for i in split["train_views"]:
        p,c=backproject(rows[i],stride=3)
        keep=contains(p,region) if direct else np.ones(len(p),bool)
        for s in ([] if direct else selections):
            ix,u,v,z=project(p,rows[s["view"]]);a,b,x,y=s["pixels"];inside=np.zeros(len(p),bool)
            ok=(u>=a)&(u<=x)&(v>=b)&(v<=y)
            # Occluded geometry from a different object must not enlarge the box.
            rr=rows[s["view"]]
            ok &= rr["valid"][v,u] & (np.abs(rr["depth"][v,u]-z)<=.12)
            inside[ix[ok]]=True;keep &= inside
        pts.append(p[keep]);colors.append(c[keep])
    if sum(map(len,pts))<150:raise ValueError("范围内没有足够的有效三维表面；请框住目标并补拍侧面")
    xyz,rgb=voxel_fuse(np.concatenate(pts),np.concatenate(colors),.025)
    positions=independent(rows,split["train_views"],.15)
    support,free,_=evidence(xyz,rows,positions,.08)
    xyz=xyz[(support>=3)&(free<=1)]
    if len(xyz)<80:raise ValueError("至少三个分开的位置观测不足，不能可靠更新这块区域")
    if not direct:
        low=np.percentile(xyz,1,axis=0)-padding;high=np.percentile(xyz,99,axis=0)+padding
    if not oriented and (np.any(high-low<.03) or np.any(high-low>4) or np.prod(high-low)>12):
        raise ValueError("选区尺寸不合理；请缩小到一个物体或局部区域")
    scope=dict(schema="astra.manual_region.v1",bounds=[low.tolist(),high.tolist()],selections=selections,
               padding_m=padding,mode=payload.get("mode","add"),label=str(payload.get("label","手动区域"))[:64],manual_scope=True,margin_m=0,
               source_run=run.name,selection_method="3d_box" if direct else "multiview_rectangles",views_sha256=sha256(run/"job/views.npz"))
    if oriented:
        scope["schema"]="astra.manual_region.v2";scope["oriented_box"]=obb
        scope["selection_method"]="3d_oriented_box"
    xyz=xyz[contains(xyz,scope)]
    center=(low+high)/2;good=[];angle=0
    for i in positions:
        surface,_,_=evidence(xyz,rows,[i],.08)
        if (surface>0).sum()>=80:good.append(i)
    if direct and good:
        selections[:]=[dict(view=int(i)) for i in good[:2]]
        selected=[s["view"] for s in selections]
    directions=np.array([rows[i]["pose"][:3,3]-center for i in good])
    if len(directions)>=2:
        directions/=np.maximum(np.linalg.norm(directions,axis=1,keepdims=True),1e-8)
        angle=float(np.degrees(np.arccos(np.clip((directions@directions.T).min(),-1,1))))
    # Selection must have validation and test coverage before optimization.
    eval_coverage={}
    for group in ("validation_views","test_views"):
        counts=[]
        for i in split[group]:
            p,_=backproject(rows[i]);counts.append(dict(view=i,pixels=int(contains(p,scope).sum())))
        eval_coverage[group]=counts
    ready=len(good)>=3 and angle>=25 and any(r["pixels"]>=500 for r in eval_coverage["validation_views"]) and sum(r["pixels"]>=100 for r in eval_coverage["test_views"])>=2
    scope["coverage"]=dict(supported_views=good,max_angle_deg=angle,points=len(xyz),evaluation=eval_coverage,
        ready=ready,unseen_surfaces_verified=False,
        message="可以生成局部候选；没有观测到的背面仍需补拍" if ready else "视角跨度或留出验证覆盖不足，请补拍其他方向后再更新")
    origin=rows[selected[0]]["pose"][:2,3]-center[:2]
    base_angle=np.arctan2(origin[1],origin[0])
    names=["正面","左前方","左侧","左后方","背面","右后方","右侧","右前方"]
    seen=set()
    for i in good:
        offset=rows[i]["pose"][:2,3]-center[:2]
        a=(np.arctan2(offset[1],offset[0])-base_angle+np.pi/8)%(2*np.pi)
        seen.add(int(a/(np.pi/4)))
    missing=[names[i] for i in range(8) if i not in seen]
    scope["coverage"]["direction_hint"]="相对于首个方向，"+("、".join(missing)+"缺少观测；如需从这些方向使用，请补拍。" if missing else "各水平方向均有观测，但仍需检查遮挡和细节。")
    scope["region_id"]=str(payload.get("id",""))[:64]
    return write_preview(run,scope,d,rows,frames)

def write_preview(run,scope,d,rows,frames):
    scope["preview_id"]=hashlib.sha256(json.dumps(scope,sort_keys=True).encode()).hexdigest()
    preview_dir=run/"region_previews"/scope["preview_id"]
    if (preview_dir/"scope.json").exists():return json.loads((preview_dir/"scope.json").read_text())
    preview_dir.mkdir(parents=True,exist_ok=False)
    for f in frames:
        i=f["view"];image=np.uint8(np.clip(d["rgb"][i]*255,0,255));mask=scope_mask(rows[i],scope)
        overlay=image.copy();overlay[mask]=[255,175,30];image=cv2.addWeighted(image,.75,overlay,.25,0)
        y,x=np.where(mask)
        for region in scope.get("regions",[scope]):
            corners=box_corners(region);cam=(corners-rows[i]["pose"][:3,3])@rows[i]["pose"][:3,:3]
            if (cam[:,2]>.05).all():
                pix=cam@rows[i]["K"].T;uv=np.rint(np.clip(pix[:,:2]/pix[:,2,None],-1e7,1e7)).astype(int)
                for a in range(8):
                    for bit in (1,2,4):
                        b=a^bit
                        if a<b:cv2.line(image,tuple(uv[a]),tuple(uv[b]),(255,160,0),2)
        for s in scope["selections"]:
            if s["view"]==i and "pixels" in s:
                a,b,c,e=np.rint(s["pixels"]).astype(int);cv2.rectangle(image,(a,b),(c,e),(40,220,80),2)
        x0,y0,x1,y1=f["crop"]
        cv2.imwrite(str(preview_dir/f"view_{i:03d}.jpg"),cv2.cvtColor(image[y0:y1,x0:x1],cv2.COLOR_RGB2BGR))
    write_json(preview_dir/"scope.json",scope)
    return scope

def make_boxes(scope):
    if "regions" in scope:
        groups=[make_boxes(s) for s in scope["regions"]]
        return dict(old=[b for g in groups for b in g["old"]],current=[b for g in groups for b in g["current"]],manual_scope=True,margin_m=0)
    center,size,rotation=box_geometry(scope);half=size/2
    pose=np.eye(4);pose[:3,:3]=rotation;pose[:3,3]=center
    box=dict(view=scope["selections"][0]["view"],label="user_selected_region",pose=pose.tolist(),
             bounds=np.column_stack([-half,half]).ravel().tolist(),center=center.tolist(),size=(half*2).tolist(),
             support_views=scope["coverage"]["supported_views"],confidence3d=1.,confidence2d=1.)
    return dict(old=[box] if scope["mode"]=="replace" else [],current=[box],manual_scope=True,margin_m=0)

def update(source,preview_id,run,gpu,render_proof=None,reconstruction_key=None):
    source=output_path(source);run=output_path(run)
    if len(preview_id)!=64 or any(c not in "0123456789abcdef" for c in preview_id):raise ValueError("invalid_preview_id")
    scope=json.loads((source/"region_previews"/preview_id/"scope.json").read_text())
    if scope["preview_id"]!=preview_id or not scope["coverage"]["ready"]:raise ValueError("选区覆盖未通过，不能开始")
    if sha256(source/"job/views.npz")!=scope["views_sha256"]:raise ValueError("选区对应的几何已变化")
    if scope["mode"]=="delete":
        from .deletion import verify_render_proof
        verify_render_proof(source,preview_id,render_proof)
    if reconstruction_key:
        from .spirula import asset
        from .deletion import verify_candidate_proof
        if asset(source)['key']!=reconstruction_key:raise ValueError("重建版本已变化，请重新预览")
        verify_candidate_proof(source,preview_id,render_proof,reconstruction_key)
    run.mkdir(parents=True,exist_ok=False);cfg=json.loads((source/"config.json").read_text());lock=(ROOT/"runs/gpu.lock").open("w")
    try:
        if os.environ.get("ASTRA_WAIT_GPU")=="1":
            write_json(run/"status.json",dict(state="queued",stage="queued",updated=time.time()))
            fcntl.flock(lock,fcntl.LOCK_EX)
        else:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        from .pipeline import subprocess
        free,total=map(int,subprocess.check_output(["nvidia-smi","--id",str(gpu),"--query-gpu=memory.free,memory.total","--format=csv,noheader,nounits"],text=True).strip().split(","))
        if free<20000:raise ValueError("GPU_needs_20000_MiB_free")
        env=environment(gpu);env["VPS_GPU_MEMORY_FRACTION"]=str(min(.85,(free-2048)/total))
        if scope["mode"]=="add":env["PHONE_FREEZE_BASE"]="1"
        shutil.copytree(source/"job",run/"job")
        shutil.copy2(source/"config.json",run/"config.json");shutil.copy2(source/"source.json",run/"source.json")
        if (source/"base_reference_fingerprints.json").exists():shutil.copy2(source/"base_reference_fingerprints.json",run/"base_reference_fingerprints.json")
        write_json(run/"manual_scope.json",scope)
        target=run/"update";target.mkdir();(target/"boxer_native_game_v1").mkdir()
        if scope["mode"]=="delete":
            from .deletion import create_delete_candidate
            create_delete_candidate(source,scope,run)
            vp=cfg["vps_python"]
        elif reconstruction_key:
            from .spirula import fuse,asset
            vp=cfg["vps_python"];polish=run/"polish";polish.mkdir()
            write_json(run/"status.json",dict(state="running",stage="merge"))
            merged=fuse(source,scope,polish/"game_candidate.ply",polish/"lineage.npz",reconstruction_key)
            write_json(run/"merge_audit.json",merged);write_json(run/"reconstruction_used.json",asset(source))
        else:
            write_json(target/"boxer_native_game_v1/boxer_boxes.json",make_boxes(scope))
            gp,vp=cfg["game_python"],cfg["vps_python"]
            for stage in ["prepare-base","render","guarded"]:
                run_stage(run,stage,[gp,PHONE/"run_update.py",run/"job",target,"--root",ENGINE,"--stage",stage,
                     "--steps",cfg["steps"]]+(["--freeze-base"] if scope["mode"]=="add" else []),env)
            run_stage(run,"refine",[gp,PHONE/"refine_phone_scene.py",run/"job",target,run/"polish","--root",ENGINE,
                    "--source-candidate",target/"boxer_native_game_v1/guarded/game_candidate.ply",
                    "--iterations",max(cfg["polish_iterations"],2400),"--depth-weight",".03","--training-width","1024","--fused-init","--adaptive-density"],env)
        run_stage(run,"scope_audit",[vp,"-m","astra_scene.roi","audit","--output",run],env)
        for stage,cmd in [
            ("artifacts",[vp,"-m","astra_scene.postprocess","artifacts",run]),
            ("references",[vp,"-m","astra_scene.references",run]),
            ("finalize",[vp,"-m","astra_scene.postprocess","finalize",run])]:run_stage(run,stage,cmd,env)
    except BaseException as exc:
        write_json(run/"status.json",dict(state="failed",error=str(exc),updated=time.time()))
        raise
    finally:lock.close()

def audit(run):
    from plyfile import PlyData
    cfg=json.loads((run/"config.json").read_text());scope=json.loads((run/"manual_scope.json").read_text())
    base=PlyData.read(cfg["map_ply"],mmap=True)["vertex"].data
    candidate=PlyData.read(str(run/"polish/game_candidate.ply"),mmap=True)["vertex"].data
    lineage_path=run/"polish/lineage.npz"
    if not lineage_path.exists():lineage_path=run/"update/boxer_native_game_v1/guarded/lineage.npz"
    lineage=np.load(lineage_path)["original_ids"]
    if len(lineage)!=len(candidate):raise ValueError("candidate_lineage_length_mismatch")
    xyz=np.column_stack([base[n] for n in ["x","y","z"]]);outside=~contains(xyz,scope)
    if ((lineage>=len(base))|(lineage < -1)).any():raise ValueError("invalid_original_lineage")
    old_ids=lineage[lineage>=0]
    if len(np.unique(old_ids))!=len(old_ids):raise ValueError("duplicated_original_lineage")
    def preserved(pos):
        # SH0 can be losslessly extended to SH2. Original values remain byte exact.
        if not set(base.dtype.names).issubset(candidate.dtype.names):return False
        for name in base.dtype.names:
            if candidate[name][pos].tobytes()!=base[name][lineage[pos]].tobytes():return False
        for name in set(candidate.dtype.names)-set(base.dtype.names):
            if not (name.startswith("f_rest_") or name in ("nx","ny","nz")):return False
            if np.any(candidate[name][pos]!=0):return False
        return True
    if not np.isin(np.flatnonzero(outside),old_ids).all():raise ValueError("outside_scope_original_deleted")
    frozen=np.flatnonzero((lineage>=0)&outside[np.maximum(lineage,0)])
    for start in range(0,len(frozen),100000):
        ids=frozen[start:start+100000]
        if not preserved(ids):
            raise ValueError("outside_scope_original_modified")
    fresh=np.column_stack([candidate[n][lineage<0] for n in ["x","y","z"]])
    if not contains(fresh,scope).all():raise ValueError("new_gaussian_outside_scope")
    if scope["mode"] in ("add","delete"):
        if scope["mode"]=="add" and len(old_ids)!=len(base):raise ValueError("addition_mode_deleted_original")
        for start in range(0,len(lineage),100000):
            pos=np.arange(start,min(len(lineage),start+100000));pos=pos[lineage[pos]>=0]
            if not preserved(pos):
                raise ValueError("addition_mode_modified_original")
    write_json(run/"scope_audit.json",dict(passed=True,outside_original_rows_unchanged=len(frozen),new_gaussians=len(fresh),
       mode=scope["mode"],all_original_unchanged=scope["mode"]=="add",all_surviving_original_unchanged=scope["mode"] in ("add","delete"),bounds=scope["bounds"],oriented_box=scope.get("oriented_box"),padding_in_guard_m=0,
       note="Gaussian_center_scope; projected_splat_footprints_can_extend_beyond_center_bounds"))

def main():
    p=argparse.ArgumentParser();p.add_argument("action",choices=["update","audit"]);p.add_argument("--source",type=Path)
    p.add_argument("--reconstruction-key");p.add_argument("--preview");p.add_argument("--render-proof");p.add_argument("--output",required=True,type=Path);p.add_argument("--gpu",type=int,default=1);a=p.parse_args()
    if a.action=="audit":audit(a.output)
    else:update(a.source,a.preview,a.output,a.gpu,a.render_proof,a.reconstruction_key)
if __name__=="__main__":main()
