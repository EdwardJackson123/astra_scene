"""Generate an isolated automatic review bundle, including conservative occupancy updates."""
from pathlib import Path
import fcntl,json,shutil,time
import numpy as np
from .io import ROOT,output_path,write_json,sha256
from .pipeline import environment,run_stage

def resolve_backend(cfg,reconstruction,backend=None):
    requested=backend or cfg.get("fusion_backend","context")
    if requested not in ("scoped","context"):
        raise ValueError("unsupported_fusion_backend")
    if requested=="context" and any(reconstruction.get(k,0)!=0 for k in ("base_sh_degree","output_sh_degree")):
        if backend is not None:
            raise ValueError("context_backend_requires_SH0")
        return "scoped","当前连续区域优化支持SH0；本次高阶SH使用兼容融合后端，未降阶"
    return requested,None

def build_candidate(source,run,regions,gpu,policy,backend=None,prepared_context=None):
    from .postprocess import load
    from .roi import validate_bounds
    from .spirula import fuse,asset
    source=output_path(source);run=output_path(run)
    if not regions:raise ValueError("empty_automatic_regions")
    for region in regions:validate_bounds(region)
    cfg,d,split,_=load(source)
    reconstruction=asset(source,True)
    backend,backend_note=resolve_backend(cfg,reconstruction,backend)
    policy={**policy,"fusion_backend":backend,"backend_note":backend_note or policy.get("backend_note")}
    if prepared_context is not None and backend!="context":
        raise ValueError("prepared_context_requires_context_backend")
    bounds=np.asarray([r["bounds"] for r in regions])
    scope=dict(schema="astra.automatic_objects.v1",regions=regions,mode="replace",
        bounds=[bounds[:,0].min(0).tolist(),bounds[:,1].max(0).tolist()],
        label="自动整物体候选",manual_scope=False,automatic_scope=True,
        source_run=source.name,views_sha256=sha256(source/"job/views.npz"),
        selections=[dict(view=split["train_views"][0])],
        coverage=dict(ready=not policy.get("blocked"),supported_views=policy.get("train_views",[]),
            unseen_surfaces_verified=False,message="覆盖不足项目见自动复核报告"),
        margin_m=0,padding_m=0)
    run.mkdir(parents=True,exist_ok=False)
    write_json(run/"status.json",dict(state="queued",stage="queued",updated=time.time()))
    # This separate candidate path does not relax the manually approved ROI API.
    write_json(run/"automatic_review_required.json",dict(reason="automatic_object_identity_and_background_not_field_verified",policy=policy,deployable=False))
    with (ROOT/"runs/gpu.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        try:
            shutil.copytree(source/"job",run/"job")
            for name in ["config.json","source.json","base_reference_fingerprints.json"]:
                if (source/name).exists():shutil.copy2(source/name,run/name)
            write_json(run/"manual_scope.json",scope)
            (run/"polish").mkdir()
            write_json(run/"status.json",dict(state="running",stage="merge",updated=time.time()))
            reconstruction=asset(source,True)
            merged=fuse(source,scope,run/"polish/game_candidate.ply",run/"polish/lineage.npz",reconstruction["key"])
            if backend=="context":
                from .context_fusion import run as refine_context, verified_prepared
                if prepared_context is None:
                    prepared_context=run/"context_fusion"
                    write_json(run/"status.json",dict(state="running",stage="context_fusion",updated=time.time()))
                    refine_context(source,run,prepared_context)
                    import torch
                    torch.cuda.empty_cache()
                context,context_report=verified_prepared(prepared_context,reconstruction,run/"polish/game_candidate.ply",source)
                scope=json.loads((context/"manual_scope.json").read_text())
                write_json(run/"manual_scope.json",scope)
                shutil.copy2(context/"scene.ply",run/"polish/game_candidate.ply")
                shutil.copy2(context/"lineage.npz",run/"polish/lineage.npz")
                write_json(run/"context_fusion.json",context_report)
                write_json(run/"context_inputs.json",json.loads((context/"inputs.json").read_text()))
                with np.load(context/"lineage.npz") as ids:
                    original_ids=ids["original_ids"]
                    merged.update(new_gaussians=int((original_ids<0).sum()),
                        fusion_backend="context",local_refinement=True,retrained=True,reconstruction_retrained=False,
                        base_fields_byte_preserved=False,outside_context_fields_byte_preserved=True,
                        limitation="Context boundary permits old appearance updates; original geometry is frozen, new geometry is bounded. No navigation certificate.")
                # Count removals from lineage rather than from the original box scope.
                from plyfile import PlyData
                total=len(PlyData.read(cfg["map_ply"],mmap=True)["vertex"].data)
                merged["removed_original"]=total-int((original_ids>=0).sum())
            write_json(run/"merge_audit.json",merged);write_json(run/"reconstruction_used.json",reconstruction)
            write_json(run/"object_policy.json",policy)
            if policy.get("fusion_guard"):
                write_json(run/"fusion_guard.json",policy["fusion_guard"])
            env=environment(gpu);vp=cfg["vps_python"]
            for stage,cmd in [
                ("scope_audit",[vp,"-m","astra_scene.roi","audit","--output",run]),
                ("artifacts",[vp,"-m","astra_scene.postprocess","artifacts",run]),
                ("references",[vp,"-m","astra_scene.references",run]),
                ("finalize",[vp,"-m","astra_scene.postprocess","finalize",run])]:
                run_stage(run,stage,cmd,env)
        except BaseException as exc:
            write_json(run/"status.json",dict(state="failed",stage="failed",error=str(exc),updated=time.time()))
            raise
