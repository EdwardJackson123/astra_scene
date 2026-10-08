from __future__ import annotations
import argparse,html,json,math,shutil,time
from pathlib import Path
import cv2,numpy as np,yaml
from plyfile import PlyData
from .io import ROOT,output_path,write_json,sha256,rigid,intrinsic,independent
from .geometry import evidence,discover,update_grid,backproject,project

def load(run):
    run=output_path(run);cfg=json.loads((run/"config.json").read_text())
    job=run/"job";d=dict(np.load(job/"views.npz",allow_pickle=False))
    split=json.loads((job/"refit_report.json").read_text())
    report=json.loads((job/"report.json").read_text())
    return cfg,d,split,report

def check_geometry(run):
    cfg,d,split,report=load(run)
    gate=json.loads((run/"job/dense_reprojection.json").read_text())
    good=[r for r in gate["checks"] if r["matches"]>=cfg["min_dense_matches"]]
    passed=bool(gate.get("passed")) and len(good)>=cfg["min_geometry_pairs"]
    passed &= all(np.isfinite([r["median"],r["p90"]]).all() and r["median"]<=cfg["median_reprojection_px"] and r["p90"]<=cfg["p90_reprojection_px"] for r in good)
    groups=[split[n] for n in ["train_views","validation_views","test_views"]]
    ids=sum(groups,[])
    if len(ids)!=len(set(ids)) or min(map(len,groups))<1 or len(groups[0])<8 or len(groups[2])<3:
        raise ValueError("need_disjoint_train_validation_and_at_least_3_test_views")
    if any(i<0 or i>=len(d["poses"]) for i in ids):raise ValueError("split_out_of_bounds")
    for i in ids:
        rigid(d["poses"][i]);intrinsic(d["intrinsics"][i])
        if not np.isfinite(d["rgb"][i]).all():raise ValueError("nonfinite_rgb")
        if not np.isfinite(d["depths"][i][d["confidence"][i].astype(bool)]).all():raise ValueError("invalid_confident_depth")
    if Path(report["map_config"]["gaussian_ply_path"]).resolve()!=Path(cfg["map_ply"]).resolve():
        raise ValueError("geometry_map_mismatch")
    audit=dict(passed=passed,checked_pairs=len(good),registered_views=len(ids),total_views=len(d["poses"]),
               unregistered_views=sorted(set(range(len(d["poses"])))-set(ids)),
               split=split,depth_source="inferred_not_sensor_depth",independent_revisit=False)
    write_json(run/"geometry_check.json",audit)
    if not passed:raise ValueError("dense_geometry_failed_capture_with_more_overlap")

def rows_from_data(d):
    rows=[]
    for i in range(len(d["poses"])):
        depth=d["depths"][i].astype(np.float32)
        valid=d["confidence"][i].astype(bool)&np.isfinite(depth)&(depth>.6)&(depth<6)
        span=cv2.dilate(depth,np.ones((3,3),np.uint8))-cv2.erode(depth,np.ones((3,3),np.uint8))
        clear=cv2.erode(valid.astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool)&(span<.04+.01*depth)
        rows.append(dict(depth=depth,valid=valid,clear=clear,K=d["intrinsics"][i],pose=d["poses"][i],rgb=d["rgb"][i]))
    return rows

def read_gaussians(path):
    v=PlyData.read(str(path),mmap=True)["vertex"].data
    names=v.dtype.names
    required=["x","y","z","opacity"]+[f"f_dc_{i}" for i in range(3)]+[f"scale_{i}" for i in range(3)]+[f"rot_{i}" for i in range(4)]
    if any(n not in names for n in required):raise ValueError("invalid_gaussian_ply")
    rest=[n for n in names if n.startswith("f_rest_")]
    if len(rest) not in (0,9,24,45):raise ValueError("invalid_SH_capacity")
    for start in range(0,len(v),100000):
        if not all(np.isfinite(v[n][start:start+100000]).all() for n in names):raise ValueError("nonfinite_gaussian")
    p=np.column_stack([v[n] for n in ["x","y","z"]])
    return v,p,{0:0,9:1,24:2,45:3}[len(rest)]

def estimate_floor(xyz,cameras):
    # Estimate from spatially distributed horizontal samples below the cameras.
    p=xyz[(xyz[:,2]<np.percentile(cameras[:,2,3],10)-.3)]
    if len(p)<100:raise ValueError("floor_z_required_no_support")
    p=p[::max(1,len(p)//400000)]
    vox=np.unique(np.floor(p/.05).astype(np.int64),axis=0)
    z,count=np.unique(vox[:,2],return_counts=True)
    floor=float((z[np.argmax(count)]+.5)*.05)
    return floor,dict(method="dominant_horizontal_voxel_layer_below_camera",floor_z=floor,verified=False,
                      limitation="set_floor_z_after_site_check; histogram_can_select_a_table_or_lower_floor")

def render_map(renderer,path,row):
    from vps.refinement.gaussian_renderer import GaussianCamera
    h,w=row["depth"].shape;k=row["K"]
    r=renderer.render(Path(path),row["pose"],GaussianCamera(w,h,k[0,0],k[1,1],k[0,2],k[1,2]))
    return dict(rgb=r.rgb,depth=r.depth,alpha=r.alpha)

def artifacts(run):
    cfg,d,split,report=load(run);out=run/"artifacts";out.mkdir(exist_ok=False)
    scope=json.loads((run/"manual_scope.json").read_text()) if (run/"manual_scope.json").exists() else None
    if scope:
        from .roi import contains,scope_mask
        shutil.copy2(run/"manual_scope.json",out/"manual_scope.json")
        shutil.copy2(run/"scope_audit.json",out/"scope_audit.json")
        if (run/"manual_deleted_centers.npy").exists():
            shutil.copy2(run/"manual_deleted_centers.npy",out/"manual_deleted_centers.npy")
            shutil.copy2(run/"deletion_audit.json",out/"deletion_audit.json")
    from vps.refinement.gaussian_renderer import GsplatGaussianRenderer
    from vps.dynamic.image_mask import content_mask
    from PIL import Image
    rows=rows_from_data(d);ids=sorted(set(sum([split[k] for k in ["train_views","validation_views","test_views"]],[])))
    candidate=run/"polish/game_candidate.ply"
    old_v,old_xyz,old_sh=read_gaussians(cfg["map_ply"])
    new_v,new_xyz,new_sh=read_gaussians(candidate)
    reconstruction=json.loads((run/"reconstruction_used.json").read_text()) if (run/"reconstruction_used.json").exists() else {}
    if reconstruction.get("backend")=="spirula":
        if new_sh!=reconstruction.get("output_sh_degree",old_sh):raise ValueError("candidate_SH_does_not_match_request")
        if reconstruction.get("sh_selection")!="explicit" and new_sh!=old_sh:raise ValueError("default_SH_must_match_original")
        if new_sh<old_sh:raise ValueError("candidate_cannot_drop_original_SH_coefficients")
    elif old_sh!=new_sh:raise ValueError("unexpected_SH_degree_change")
    for name in ["reconstruction_used.json","merge_audit.json","object_policy.json","fusion_guard.json","context_fusion.json","context_inputs.json","automatic_review_required.json"]:
        if (run/name).exists():shutil.copy2(run/name,out/name)
    if (run/"reconstruction_used.json").exists():
        used=json.loads((run/"reconstruction_used.json").read_text())
        for field,filename in [("alignment","reconstruction_alignment.json"),("checkpoint_selection","checkpoint_selection.json")]:
            path=Path(used[field])
            if not path.resolve().is_relative_to(ROOT):raise ValueError("reconstruction_metadata_outside_workspace")
            shutil.copy2(path,out/filename)
    renderer=GsplatGaussianRenderer("cuda")
    base={};current={};metrics=[]
    capture=json.loads(Path(report["capture"]).read_text())
    for i in ids:
        base[i]=render_map(renderer,cfg["map_ply"],rows[i])
        current[i]=render_map(renderer,candidate,rows[i])
        h,w=rows[i]["depth"].shape
        with Image.open(capture["frames"][i]["image"]) as im:valid=content_mask(*im.size,h,w)
        if not valid.any():raise ValueError("empty_content_mask")
        target=d["rgb"][i];a=base[i]["rgb"].astype(np.float32)/255;b=current[i]["rgb"].astype(np.float32)/255
        roi=rows[i]["valid"]&((base[i]["alpha"]<.5)|(np.abs(base[i]["depth"]-rows[i]["depth"])>cfg["change_tolerance_m"]))
        old_l1=float(np.abs(a-target)[valid].mean());new_l1=float(np.abs(b-target)[valid].mean())
        if scope:roi &= scope_mask(rows[i],scope)
        known_stable=rows[i]["valid"]&~roi if not scope else valid&~scope_mask(rows[i],scope)
        row=dict(view=i,split="test" if i in split["test_views"] else "validation" if i in split["validation_views"] else "train",
          old_rgb_l1=old_l1,new_rgb_l1=new_l1,change_pixels=int(roi.sum()),
          changed_old_l1=float(np.abs(a-target)[roi].mean()) if roi.any() else None,
          changed_new_l1=float(np.abs(b-target)[roi].mean()) if roi.any() else None,
          stable_render_delta=float(np.abs(b-a)[known_stable].mean()) if known_stable.any() else None)
        if scope and scope["mode"]!="delete":
            from .quality import structural_metrics
            row["structure"]=structural_metrics(b,current[i]["depth"],current[i]["alpha"],rows[i],scope,dict(min_pixels=100))
        metrics.append(row)
        yy,xx=np.where(valid);y0,y1=yy.min(),yy.max()+1;x0,x1=xx.min(),xx.max()+1
        images=[base[i]["rgb"],current[i]["rgb"],np.uint8(np.clip(target*255,0,255))]
        strip=np.concatenate([im[y0:y1,x0:x1] for im in images],axis=1)
        cv2.imwrite(str(out/f"view_{i:03d}.jpg"),cv2.cvtColor(strip,cv2.COLOR_RGB2BGR),[cv2.IMWRITE_JPEG_QUALITY,94])
        print("VIEW",i,old_l1,new_l1,flush=True)
    np.savez_compressed(out/"base_renders.npz",ids=np.array(ids),depth=np.array([base[i]["depth"] for i in ids]),
                        alpha=np.array([base[i]["alpha"] for i in ids]))
    add,rgb,discovery=discover(rows,split["train_views"],base,cfg)
    # Only keep observed additions represented by the delivered Gaussian scene.
    positions=independent(rows,split["train_views"],cfg["view_baseline_m"])
    agree=np.zeros(len(add),np.int16)
    for i in positions:
        ix,u,v,z=project(add,rows[i])
        agree[ix]+=(current[i]["alpha"][v,u]>.5)&(np.abs(current[i]["depth"][v,u]-z)<=cfg["change_tolerance_m"])
    add=add[agree>=cfg["min_support_views"]]
    if scope:
        add=add[contains(add,scope)]
        if scope["mode"]=="delete":add=add[:0]
    np.save(out/"confirmed_additions.npy",add)
    # Use sufficiently opaque Gaussians for occupancy evidence, not all splats.
    physical=old_xyz[old_v["opacity"]>0]
    support,free,_=evidence(physical,rows,positions,cfg["surface_tolerance_m"])
    removed_mask=(free>=cfg["min_clear_views"])&(support==0)
    if scope:
        removed_mask &= contains(physical,scope)
        if scope["mode"]=="add":removed_mask[:]=False
    removed=physical[removed_mask]
    np.save(out/"confirmed_removals.npy",removed)
    retained=physical[~removed_mask]
    if cfg["floor_z"] is None:
        cfg["floor_z"],floor_report=estimate_floor(physical,d["poses"][ids])
    else:floor_report=dict(method="configured",floor_z=cfg["floor_z"],verified=bool(cfg["floor_verified"]))
    meta=yaml.safe_load(Path(cfg["nav_yaml"]).read_text());original=cv2.imread(cfg["nav_map"],0)
    if original is None:raise ValueError("base_grid_missing")
    # Guard retained candidate obstacle geometry as well as surviving base points.
    retained=new_xyz[new_v["opacity"]>0] # Candidate already includes every surviving original point; do not resurrect deleted old geometry.
    grid,grid_report=update_grid(original,meta,add,removed,retained,rows,split["train_views"],cfg)
    cv2.imwrite(str(out/"map.png"),grid)
    (out/"map.yaml").write_text(yaml.safe_dump({**meta,"image":"map.png"}))
    delta=cv2.cvtColor(original,cv2.COLOR_GRAY2BGR)
    old_probability=original.astype(float)/255 if meta.get("negate",0) else 1-original.astype(float)/255
    new_probability=grid.astype(float)/255 if meta.get("negate",0) else 1-grid.astype(float)/255
    old_occ=old_probability>meta.get("occupied_thresh",.65)
    new_occ=new_probability>meta.get("occupied_thresh",.65)
    new_free=new_probability<meta.get("free_thresh",.196)
    delta[~old_occ&new_occ]=(0,0,255)
    delta[old_occ&new_free]=(0,180,0)
    delta[old_occ&~new_occ&~new_free]=(0,190,255)
    delta[(grid!=original)&(old_occ==new_occ)]=(40,130,255)
    cv2.imwrite(str(out/"grid_comparison.png"),np.concatenate([cv2.cvtColor(original,cv2.COLOR_GRAY2BGR),cv2.cvtColor(grid,cv2.COLOR_GRAY2BGR),delta],axis=1))
    shutil.copy2(candidate,out/"scene.ply")
    np.save(out/"trajectory_c2w.npy",d["poses"])
    test=[r for r in metrics if r["split"]=="test"]
    improved=sum(r["new_rgb_l1"]<r["old_rgb_l1"] for r in test)
    visual_pass=all(r["new_rgb_l1"]-r["old_rgb_l1"]<=cfg["max_test_rgb_regression"] for r in test) and improved/len(test)>=cfg["min_test_improved_fraction"]
    # Changed-region and unchanged-region checks cannot be replaced by whole-image mean.
    roi_test=[r for r in test if r["change_pixels"]>=100]
    visual_pass &= bool(roi_test) and all(r["changed_new_l1"]<=r["changed_old_l1"]+.01 for r in roi_test)
    stable=[r["stable_render_delta"] for r in test if r["stable_render_delta"] is not None]
    visual_pass &= bool(stable) and max(stable)<=(.015 if scope else .05)
    structural_pass=None
    if scope and scope["mode"]!="delete":
        from .quality import evaluate
        structural_review=evaluate(run,{"original":Path(cfg["map_ply"]),"candidate":candidate},out/"local_quality")
        structural_pass=all(r["passed"] for r in structural_review["views"] if r["candidate"]=="candidate")
        from .quality_review import review
        review(run,Path(cfg["map_ply"]),candidate,out/"local_quality")
        visual_pass &= structural_pass
    elif not scope:
        # Automatic proposals have not supplied a reviewed target for structural acceptance.
        structural_pass=False
        visual_pass=False
    navigation=None
    if scope and scope["mode"] in ("delete","replace"):
        from .geometry import grid_cells
        deleted=np.load(run/"manual_deleted_centers.npy") if scope["mode"]=="delete" else physical[contains(physical,scope)]
        low_height,high_height=cfg["obstacle_height_m"]
        deleted=deleted[(deleted[:,2]>=cfg["floor_z"]+low_height)&(deleted[:,2]<=cfg["floor_z"]+high_height)]
        cells,valid_cells=grid_cells(deleted,meta,original.shape);cells=np.unique(cells[valid_cells],axis=0)
        remaining=0;cleared=0
        if len(cells):
            x,y=cells[:,0],cells[:,1]
            old_occ=old_probability[y,x]>meta.get("occupied_thresh",.65)
            safe_free=new_probability[y,x]<meta.get("free_thresh",.196)
            remaining=int((old_occ&~safe_free).sum());cleared=int((old_occ&safe_free).sum())
        navigation=dict(cleared_cells=cleared,retained_obstacle_cells=remaining,needs_more_capture=remaining>0,unknown_cells=grid_report.get("occupied_to_unknown",0),mode=scope["mode"],added_cells=grid_report["actual_added_obstacles"],
            message=("已清除 "+str(cleared)+" 格障碍；还有 "+str(remaining)+" 格仍有障碍或缺少腾空证据，保持原障碍状态；模型更新不等于开放通行。" if remaining else "选区内原有障碍栅格已完成检查。"))
        write_json(out/"navigation_update.json",navigation)
    audit=dict(structural_gate_passed=structural_pass,quality_policy_version=3,navigation_update=navigation,views=metrics,visual_gate_passed=bool(visual_pass),test_views=len(test),
       improved_test_views=improved,SH_degree=new_sh,base_SH_degree=old_sh,base_gaussians=len(old_v),candidate_gaussians=len(new_v),
       discovery=discovery,grid_supported_additions=len(add),grid=grid_report,floor=floor_report,
       manual_region=scope,field_acceptance=False,depth_source="estimated_from_same_video_not_metric_ground_truth",
       limitations=["unseen_surfaces_not_validated","reflective_thin_objects_require_review","no_independent_revisit",
                    "no_navigation_or_absolute_localization_validation"])
    write_json(out/"quality.json",audit)

def finalize(run):
    cfg=json.loads((run/"config.json").read_text())
    out=run/"artifacts";quality=json.loads((out/"quality.json").read_text())
    refs=json.loads((out/"references.json").read_text())
    source=json.loads((run/"source.json").read_text())
    for item in source["sources"].values():
        if sha256(item["path"])!=item["sha256"]:raise ValueError("base_assets_changed_during_build")
    files={}
    for p in sorted(out.rglob("*")):
        if p.is_file() and p.name not in ("bundle.json","index.html","bundle.zip"):
            files[str(p.relative_to(out))]=dict(bytes=p.stat().st_size,sha256=sha256(p))
    candidate_ok=quality["visual_gate_passed"] and refs["index_complete"] and not (run/"automatic_review_required.json").exists()
    bundle=dict(schema="astra.scene.bundle.v1",map_id=cfg["map_id"],state="candidate_ready" if candidate_ok else "needs_review",
         deployable=False,field_acceptance=False,source=source,files=files,trajectory_sha256=sha256(out/"trajectory_c2w.npy"),
         service_config="service_candidate.yaml",three_assets_same_capture=True,
         base_reference_start_fingerprint_available=(run/"base_reference_fingerprints.json").exists(),
         limitations=quality["limitations"]+["floor_requires_site_verification"])
    write_json(out/"bundle.json",bundle)
    sections=[]
    for r in quality["views"]:
        sections.append(f'<section><h3>视角 {r["view"]} · {html.escape(r["split"])}</h3><p>RGB 误差：{r["old_rgb_l1"]:.4f} → {r["new_rgb_l1"]:.4f}</p><a href="view_{r["view"]:03d}.jpg"><img loading="lazy" src="view_{r["view"]:03d}.jpg"></a></section>')
    state="候选已生成，等待现场验收" if candidate_ok else "候选需复查，质量检查未全部通过"
    page='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>场景更新结果</title><style>body{font:16px/1.6 system-ui;max-width:1400px;margin:32px auto;padding:0 24px;background:#f4f6f8;color:#182638}section,.card{background:white;padding:20px;border-radius:14px;margin:20px 0}img{max-width:100%}a{color:#1760b0}.grid{image-rendering:pixelated}</style>'
    page+=f'<h1>{html.escape(cfg["map_id"])} · 手机场景更新</h1><div class="card"><h2>{state}</h2><p>已检查 {len(quality["views"])} 个注册视角，其中 {quality["test_views"]} 个未用于训练；未注册视角见几何检查记录。没有现场复访精度证明。</p><p>从左到右：旧场景 / 更新候选 / 手机实拍。每个视角都可展开查看，不能证明未拍摄的背面。</p><p>栅格实际改变 {quality["grid"]["changed_cells"]} 格；定位图库 {refs["total_references"]} 张。</p>'
    if (run/"automatic_review_required.json").exists():
        policy=json.loads((run/"object_policy.json").read_text())
        page+='<section><h2>自动整物体更新</h2><p>本轮自动选择 '+str(len(policy.get("regions",[])))+' 个范围，包含旧位置替换。覆盖待复核 '+str(len(policy.get("blocked",{})))+' 项；生成结果不等于验收通过。</p><a href="object_policy.json">查看逐区域选择与暂缓原因</a></section>'
    if quality.get("navigation_update"):
        page+='<p><strong>导航更新：</strong>'+html.escape(quality["navigation_update"]["message"])+'</p>'
    page+='<a href="bundle.zip">下载完整候选包 ZIP</a> · <a href="scene.ply">3DGS PLY</a> · <a href="map.png">栅格 PNG</a> · <a href="map.yaml">栅格 YAML</a> · <a href="reference/ref.h5">检索索引</a> · <a href="service_candidate.yaml">候选服务配置</a> · <a href="bundle.json">版本校验清单</a></div>'
    if (out/"local_quality/index.html").exists():
        page+='<section><h2>局部结构与连续视角检查</h2><p>必须同时检查选区误差、观测表面缺失和前景几何偏差。推断深度不等于现场真值。</p><a href="local_quality/index.html">展开高分辨率对照</a><video controls loop preload="metadata" style="width:100%" src="local_quality/continuous_review.mp4"></video><p>连续视频为相邻虚拟视角，仅供人工检查，不证明未拍摄背面的质量。</p></section>'
    page+='<section><h2>栅格：原图 / 更新 / 变化标记</h2><img class="grid" src="grid_comparison.png"><p>红色表示新增障碍，绿色表示确认可通行，黄色表示原障碍已有移除证据但通行空间待确认，橙色表示其他变化。保留原坐标、比例和未观测区域。障碍边界保留细结构；自由空间来自严格观测检查，仍需现场核对地面高度与通行条件。</p></section>'
    stable_limit=.015 if quality.get("manual_region") else .05
    failures=[f'测试视角 {r["view"]} 的保护区域外观差异 {r["stable_render_delta"]:.4f} 超过 {stable_limit:.4f}' for r in quality["views"] if r["split"]=="test" and r["stable_render_delta"] is not None and r["stable_render_delta"]>stable_limit]
    for r in quality["views"]:
        if r["split"]!="test":continue
        m=r.get("structure")
        if m and not m["passed"]:
            failures.append(f'测试视角 {r["view"]}：选区结构检查未通过，详情见质量报告（孔洞/深度不一致不能用整图平均误差掩盖）')
    if failures:page+='<section><h2>需要复查的原因</h2><ul>'+''.join('<li>'+html.escape(x)+'</li>' for x in failures)+'</ul><p>外观差异增大不等于定位误差；需结合原图判断是否有不应出现的背景修改。</p></section>'
    page+='<p><a href="quality.json">查看逐视角质量检查</a></p>'
    page+=''.join(sections)+'</html>'
    (out/"index.html").write_text(page,encoding="utf-8")
    import zipfile
    with zipfile.ZipFile(out/"bundle.zip","w",compression=zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as archive:
        for path in sorted(out.rglob("*")):
            if path.is_file() and path.name!="bundle.zip":
                archive.write(path,str(path.relative_to(out)))
    write_json(run/"status.json",dict(state=bundle["state"],stage="complete",report=str(out/"index.html"),updated=time.time()))

def verify(path):
    path=Path(path);b=json.loads(path.read_text());errors=[]
    for name,entry in b["files"].items():
        p=(path.parent/name).resolve()
        if not p.is_relative_to(path.parent.resolve()) or not p.is_file() or sha256(p)!=entry["sha256"]:
            errors.append(name)
    if errors:raise ValueError("bundle_integrity_failed:"+",".join(errors))
    return dict(verified_files=len(b["files"]),deployable=b["deployable"])

def main():
    p=argparse.ArgumentParser();p.add_argument("action",choices=["geometry","artifacts","finalize","verify"]);p.add_argument("path",type=Path);a=p.parse_args()
    if a.action=="verify":print(verify(a.path))
    else:{"geometry":check_geometry,"artifacts":artifacts,"finalize":finalize}[a.action](output_path(a.path))
if __name__=="__main__":main()
