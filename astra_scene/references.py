"""Refresh real reference images and compose a MegaLoc index in one version."""
from __future__ import annotations
import argparse,json,shutil
from pathlib import Path
import cv2,h5py,numpy as np,yaml
from scipy.spatial import cKDTree
from .io import ROOT,write_json,sha256,rigid
from .postprocess import load,rows_from_data,render_map
from .geometry import project

def descriptor(f,name):
    if name not in f or "global_descriptor" not in f[name]:raise ValueError("missing_descriptor:"+name)
    v=np.asarray(f[name]["global_descriptor"],np.float32)
    if v.ndim!=1 or not np.isfinite(v).all() or np.linalg.norm(v)<1e-8:raise ValueError("invalid_descriptor:"+name)
    return v

def build(run):
    cfg,d,split,report=load(run);out=run/"artifacts";ref=out/"reference"
    for name in ["rgb","poses","calibration","depth"]:(ref/name).mkdir(parents=True,exist_ok=False)
    from vps.models.vpr_model import VPRModel
    from vps.refinement.gaussian_renderer import GsplatGaussianRenderer
    from vps.dynamic.image_mask import content_mask
    from PIL import Image
    base=Path(cfg["ref_root"])
    pinned=json.loads((run/"base_reference_fingerprints.json").read_text()) if (run/"base_reference_fingerprints.json").exists() else None
    changes=np.load(out/"confirmed_additions.npy")
    if (out/"confirmed_removals.npy").exists():
        changes=np.concatenate([changes,np.load(out/"confirmed_removals.npy")])
    if (out/"manual_deleted_centers.npy").exists():
        changes=np.concatenate([changes,np.load(out/"manual_deleted_centers.npy")])
    changes=changes[::max(1,len(changes)//5000)]
    tree=cKDTree(changes) if len(changes) else None
    camera_path=Path(cfg.get("reference_cameras","/home/xujie/FastGS/data_scene/xichuang0915/sparse/0/cameras.txt"))
    camera_rows=[line.split() for line in camera_path.read_text().splitlines() if line.strip() and not line.startswith("#")]
    if len(camera_rows)!=1 or camera_rows[0][1]!="PINHOLE":raise ValueError("requires_single_PINHOLE_reference_camera")
    cam=camera_rows[0];width,height=map(int,cam[2:4]);fx,fy,cx,cy=map(float,cam[4:8])
    k=np.array([[fx,0,cx],[0,fy,cy],[0,0,1]])
    renderer=GsplatGaussianRenderer("cuda");kept=[];retired=[];checked=0
    for image in sorted((base/"rgb").iterdir()):
        if image.suffix.lower() not in (".jpg",".jpeg",".png"):continue
        pose_path=base/"poses"/(image.stem+".txt");pose=rigid(np.loadtxt(pose_path))
        stale=False;fraction=0
        # Frustum test before rendering; do not retire every camera near an object.
        if tree is not None and tree.query(pose[:3,3])[0]<cfg["reference_refresh_radius_m"]:
            scale=min(320/width,320/height);h,w=round(height*scale),round(width*scale)
            rr=dict(depth=np.zeros((h,w)),K=np.diag([scale,scale,1])@k,pose=pose)
            ix,u,v,z=project(changes,rr)
            if len(ix)>=20:
                old=render_map(renderer,cfg["map_ply"],rr);new=render_map(renderer,out/"scene.ply",rr);checked+=1
                roi=np.zeros((h,w),np.uint8)
                visible=((old["alpha"][v,u]>.1)&(old["depth"][v,u]>=z-.15))|((new["alpha"][v,u]>.1)&(new["depth"][v,u]>=z-.15))
                roi[v[visible],u[visible]]=1
                roi=cv2.dilate(roi,np.ones((5,5),np.uint8)).astype(bool)
                delta=np.abs(new["rgb"].astype(float)-old["rgb"].astype(float)).mean(axis=2)/255
                fraction=float(((delta>.06)&roi).mean())
                stale=fraction>=cfg["reference_stale_fraction"]
        if stale:
            retired.append(dict(name=image.name,changed_fraction=fraction))
            continue
        # Copies form an immutable candidate; changing the old source later does
        # not silently alter an existing reference image or its pose.
        shutil.copy2(image,ref/"rgb"/image.name);shutil.copy2(pose_path,ref/"poses"/pose_path.name)
        if pinned:
            expected=pinned[image.name]
            if sha256(ref/"rgb"/image.name)!=expected["rgb_sha256"] or sha256(ref/"poses"/pose_path.name)!=expected["pose_sha256"]:
                raise ValueError("source_reference_changed_during_build:"+image.name)
        np.savetxt(ref/"calibration"/pose_path.name,k)
        kept.append(image.name)
    capture=json.loads(Path(report["capture"]).read_text());new_names=[]
    scope=json.loads((run/"manual_scope.json").read_text()) if (run/"manual_scope.json").exists() else None
    tmp=run/"new_reference_rgb";tmp.mkdir()
    for i in split["train_views"]:
        if scope and i not in scope["coverage"]["supported_views"]:continue
        with Image.open(capture["frames"][i]["image"]) as im:
            mask=content_mask(*im.size,*d["depths"][i].shape)
        y,x=np.where(mask);y0,y1=y.min(),y.max()+1;x0,x1=x.min(),x.max()+1
        name=f"phone_{run.name}_{i:04d}.png"
        if (ref/"rgb"/name).exists():raise ValueError("reference_name_collision")
        rgb=np.uint8(np.clip(d["rgb"][i,y0:y1,x0:x1]*255,0,255))
        cv2.imwrite(str(ref/"rgb"/name),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
        shutil.copy2(ref/"rgb"/name,tmp/name)
        K=d["intrinsics"][i].copy();K[0,2]-=x0;K[1,2]-=y0
        np.savetxt(ref/"calibration"/(Path(name).stem+".txt"),K)
        np.savetxt(ref/"poses"/(Path(name).stem+".txt"),d["poses"][i])
        np.save(ref/"depth"/(Path(name).stem+".npy"),np.where(d["confidence"][i,y0:y1,x0:x1],d["depths"][i,y0:y1,x0:x1],0))
        new_names.append(name)
    if len(kept)<6 or not new_names:raise ValueError("insufficient_reference_coverage")
    model_cfg=yaml.safe_load(Path(cfg["vps_config"]).read_text());model_cfg["system"]["device"]="cuda"
    model=VPRModel(model_cfg);model.extract_global_descriptors(tmp,run/"new_ref.h5")
    dimensions=set()
    with h5py.File(cfg["descriptors"],"r") as old,h5py.File(run/"new_ref.h5","r") as new,h5py.File(ref/"ref.h5","w") as target:
        for names,source in [(kept,old),(new_names,new)]:
            for name in names:
                dimensions.add(len(descriptor(source,name)));source.copy(source[name],target,name=name)
        if len(dimensions)!=1:raise ValueError("old_new_descriptor_dimension_mismatch")
        target.attrs["method"]=model.method
        target.attrs["preprocessing_json"]=json.dumps(model.retrieval_conf["preprocessing"],sort_keys=True)
        target.attrs["source_index_sha256"]=sha256(cfg["descriptors"])
    with h5py.File(ref/"ref.h5","r") as f:
        actual={p.name for p in (ref/"rgb").iterdir()}
        if set(f)!=actual:raise ValueError("index_rgb_set_mismatch")
        for name in actual:
            descriptor(f,name);rigid(np.loadtxt(ref/"poses"/(Path(name).stem+".txt")))
    # Exercise the actual VPS loader, not just HDF5 structure.
    from vps.manager.map_manager import MapManager
    from vps.maps.vpr_map import VPRMap,VPRMapConfig
    from vps.maps.pose_map import PoseMap,PoseMapConfig
    from vps.maps.map_instance import MapInstance
    key=cfg["map_id"]+"_candidate"
    vm=VPRMap(VPRMapConfig(key,ref,ref/"rgb",ref/"poses",ref/"ref.h5"))
    pm=PoseMap(PoseMapConfig(key,ref,ref/"poses"))
    manager=MapManager();manager.register(MapInstance(key,vm,pm));manager.prepare_map(key,model,check_ref=False)
    # Independent holdout descriptors test retrieval against the new mixed index.
    tests=run/"retrieval_queries";tests.mkdir()
    for i in split["test_views"]:
        rgb=np.uint8(np.clip(d["rgb"][i]*255,0,255))
        cv2.imwrite(str(tests/f"test_{i:03d}.png"),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
    model.extract_global_descriptors(tests,run/"test_ref.h5")
    retrieval=[]
    with h5py.File(ref/"ref.h5","r") as f,h5py.File(run/"test_ref.h5","r") as q:
        names=list(f);matrix=np.stack([descriptor(f,n) for n in names]);matrix/=np.maximum(np.linalg.norm(matrix,axis=1,keepdims=True),1e-8)
        for i in split["test_views"]:
            value=descriptor(q,f"test_{i:03d}.png");value/=np.linalg.norm(value)
            order=np.argsort(-(matrix@value))[:10]
            retrieval.append(dict(view=i,top10=[names[j] for j in order],
                 top1_camera_distance_m=float(np.linalg.norm(np.loadtxt(ref/"poses"/(Path(names[order[0]]).stem+".txt"))[:3,3]-d["poses"][i,:3,3]))))
    service=yaml.safe_load(Path(cfg["vps_config"]).read_text())
    original=next(m for m in service["service"]["maps"] if m["id"]==cfg["map_id"])
    map_cfg=dict(original.get("light_variants",{}).get("bright",original))
    map_cfg.update(id=key,ref_data_path=str(ref),ref_descriptors_path=str(ref/"ref.h5"),vggt_omega_ref_cache_path=None,
         gaussian_ply_path=str(out/"scene.ply"),nav_map_path=str(out/"map.png"),nav_yaml_path=str(out/"map.yaml"),
         calibration_dir=str(ref/"calibration"))
    service["service"]["maps"]=[map_cfg]
    service["service"]["temp_dir"]=str(run/"service_tmp")
    service["dynamic_update"]={"enabled":False}
    service["gsplat_refinement"]["save_render"]=False
    (out/"service_candidate.yaml").write_text(yaml.safe_dump(service,allow_unicode=True,sort_keys=False))
    write_json(out/"references.json",dict(index_complete=True,total_references=len(kept)+len(new_names),
       retained_old=len(kept),added_real_phone=len(new_names),retired=retired,staleness_render_checks=checked,
       descriptor_dimension=next(iter(dimensions)),method=model.method,preprocessing=model.retrieval_conf["preprocessing"],
       VPS_loader_passed=True,retrieval=retrieval,localization_accuracy_verified=False,
       cache_policy="new_map_omits_old_VGGT_cache; rebuilt_on_demand",
       limitation="retrieval_consistency_is_not_absolute_pose_accuracy; inherited_base_index_uses_deployed_configuration"))
    print("REFERENCE_INDEX_READY",len(kept)+len(new_names),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("run",type=Path);a=p.parse_args();build(a.run)
