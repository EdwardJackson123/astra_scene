from __future__ import annotations
import argparse, contextlib, fcntl, json, os, signal, subprocess, sys, time
from pathlib import Path
from .io import ROOT,output_path,write_json,sha256

ENGINE=ROOT/"vendor/engine"
PHONE=ROOT/"vendor/phone"

def environment(gpu):
    env=os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(gpu),VPS_ROOT=str(ENGINE),
      OMP_NUM_THREADS="4",OPENBLAS_NUM_THREADS="4",CUDA_HOME="/usr/local/cuda-12.8",
      TORCH_EXTENSIONS_DIR=str(ROOT/"cache/torch_extensions"),TORCH_CUDA_ARCH_LIST="12.0",MAX_JOBS="4",
      PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
      TORCH_HOME=str(ROOT/"cache/torch"),XDG_CACHE_HOME=str(ROOT/"cache"),
      PYTHONPYCACHEPREFIX=str(ROOT/"cache/pycache"),
      MPLCONFIGDIR=str(ROOT/"cache/matplotlib"),TMPDIR=str(ROOT/"cache/tmp"),
      PYTHONPATH=str(ROOT)+os.pathsep+str(ENGINE)+os.pathsep+env.get("PYTHONPATH",""))
    env["PATH"]="/home/xujie/miniconda3/envs/vps/bin:/usr/local/cuda-12.8/bin:"+env.get("PATH","")
    (ROOT/"cache/tmp").mkdir(exist_ok=True)
    return env

def run_stage(run,name,cmd,env,timeout=3600):
    write_json(run/"status.json",dict(state="running",stage=name,updated=time.time()))
    print("START",name,flush=True)
    with (run/(name+".log")).open("w") as log:
        proc=subprocess.Popen(list(map(str,cmd)),env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            rc=proc.wait(timeout=timeout)
        except BaseException:
            os.killpg(proc.pid,signal.SIGTERM)
            try:proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGKILL);proc.wait()
            raise
    if rc: raise RuntimeError(name+"_failed: see "+str(run/(name+".log")))
    write_json(run/(name+".done.json"),dict(command=list(map(str,cmd)),completed=time.time()))
    print("DONE",name,flush=True)

def build(video,run,config,gpu,prepared=None,geometry_only=False,spirula=False,sh_degree='auto'):
    run=output_path(run);run.mkdir(parents=True,exist_ok=False)
    cfg=json.loads(Path(config).read_text())
    video=Path(video).resolve()
    if not video.is_file(): raise ValueError("video_missing")
    write_json(run/"config.json",cfg)
    lock=(ROOT/"runs/gpu.lock").open("w")
    try:
        if os.environ.get("ASTRA_WAIT_GPU")=="1":
            write_json(run/"status.json",dict(state="queued",stage="queued",updated=time.time()))
            fcntl.flock(lock,fcntl.LOCK_EX)
        else:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        memory=subprocess.check_output(["nvidia-smi","--id",str(gpu),"--query-gpu=memory.free,memory.total","--format=csv,noheader,nounits"],text=True)
        free,total=[int(x.strip()) for x in memory.strip().split(",")]
        if free<20000:raise RuntimeError("GPU_needs_20000_MiB_free")
        env=environment(gpu);env["VPS_GPU_MEMORY_FRACTION"]=str(min(.85,(free-2048)/total))
        manifest=dict(video=str(video),video_sha256=sha256(video),config_sha256=sha256(config),
                      created=time.time(),gpu=gpu,depth_source="inferred_not_measured",published=False,
                      sources={k:dict(path=cfg[k],sha256=sha256(cfg[k])) for k in ["map_ply","nav_map","nav_yaml","descriptors"]})
        reference_sources={}
        refroot=Path(cfg["ref_root"])
        for im in sorted((refroot/"rgb").iterdir()):
            if im.suffix.lower() in (".jpg",".jpeg",".png"):
                pose=refroot/"poses"/(im.stem+".txt")
                reference_sources[im.name]=dict(rgb_sha256=sha256(im),pose_sha256=sha256(pose))
        write_json(run/"base_reference_fingerprints.json",reference_sources)
        manifest["reference_fingerprints_sha256"]=sha256(run/"base_reference_fingerprints.json")
        manifest["code_sha256"]={str(p.relative_to(ROOT)):sha256(p) for p in sorted((ROOT/"astra_scene").glob("*.py"))}
        write_json(run/"source.json",manifest)
        bp,gp,vp=cfg["backend_python"],cfg["game_python"],cfg["vps_python"]
        if prepared:
            import shutil
            job=Path(prepared).resolve()
            report=json.loads((job/"report.json").read_text())
            if Path(report["map_config"]["gaussian_ply_path"]).resolve()!=Path(cfg["map_ply"]).resolve():
                raise ValueError("prepared_map_mismatch")
            capture=json.loads(Path(report["capture"]).read_text())
            if capture["source_sha256"]!=manifest["video_sha256"]:
                raise ValueError("prepared_video_mismatch")
            shutil.copytree(job,run/"job")
            write_json(run/"reused_geometry.json",dict(source=str(job),views_sha256=sha256(job/"views.npz"),
                       reason="explicit_cached_geometry_regression_not_fresh_reconstruction"))
        else:
            if video.suffix.lower()==".json":
                from .capture import import_photos
                write_json(run/"status.json",dict(state="running",stage="extract"))
                import_photos(video,run/"capture",cfg["max_frames"])
            else:
                run_stage(run,"extract",[bp,PHONE/"phone_video.py","extract",video,run/"capture","--max-frames",cfg.get("video_max_frames",0)],env)
            cmd=[vp,PHONE/"localize_video_vps.py",run/"capture/capture.json",run/"vps","--root",ENGINE,
                 "--config",cfg["vps_config"],"--ref-root",cfg["ref_root"],"--descriptors",cfg["descriptors"],"--map-id",cfg["map_id"]]
            if cfg.get("ref_cache"):cmd+=["--ref-cache",cfg["ref_cache"]]
            run_stage(run,"localize",cmd,env)
            run_stage(run,"reconstruct",[bp,PHONE/"phone_video.py","reconstruct",run/"capture/capture.json",run/"raw_job",
                 "--root",ENGINE,"--ref-root",cfg["ref_root"],"--map-ply",cfg["map_ply"],
                 "--vps-localization",run/"vps/localization.json"],env)
            run_stage(run,"anchor_render",[gp,PHONE/"run_update.py",run/"raw_job",run/"geometry_renders",
                 "--root",ENGINE,"--stop-after","render"],env)
            run_stage(run,"bundle",[gp,PHONE/"bundle_phone.py","--source-job",run/"raw_job",
                 "--old-renders",run/"geometry_renders/old_renders.npz","--output",run/"job"],env)
        run_stage(run,"geometry_check",[vp,"-m","astra_scene.postprocess","geometry",run],env)
        if spirula:
            run_stage(run,"spirula",[vp,"-m","astra_scene.spirula","--source",run,"--gpu",gpu,"--sh-degree",sh_degree],env)
        if geometry_only:
            write_json(run/"status.json",dict(state="geometry_ready",stage="geometry_check",updated=time.time()));return
        run_stage(run,"update",[gp,PHONE/"run_update.py",run/"job",run/"update","--root",ENGINE,
                 "--boxer-assets",ROOT/"vendor/boxer_assets","--steps",cfg["steps"],
                 "--target-labels",cfg["target_labels"],"--stop-after","guarded"],env)
        run_stage(run,"refine",[gp,PHONE/"refine_phone_scene.py",run/"job",run/"update",run/"polish",
                 "--root",ENGINE,"--source-candidate",run/"update/boxer_native_game_v1/guarded/game_candidate.ply",
                 "--iterations",max(cfg["polish_iterations"],2400),"--depth-weight",".03","--training-width","1024","--fused-init","--adaptive-density"],env)
        run_stage(run,"artifacts",[vp,"-m","astra_scene.postprocess","artifacts",run],env)
        run_stage(run,"references",[vp,"-m","astra_scene.references",run],env)
        run_stage(run,"finalize",[vp,"-m","astra_scene.postprocess","finalize",run],env)
    except BaseException as exc:
        write_json(run/"status.json",dict(state="failed",error=type(exc).__name__+":"+str(exc),updated=time.time()))
        raise
    finally:
        lock.close()

def main():
    p=argparse.ArgumentParser(description="手机视频增量更新")
    p.add_argument("--video",required=True);p.add_argument("--output",required=True)
    p.add_argument("--config",default=str(ROOT/"configs/xichuang0915.json"));p.add_argument("--gpu",type=int,default=1)
    p.add_argument("--prepared",help="explicitly reuse validated geometry for regression")
    p.add_argument("--geometry-only",action="store_true")
    p.add_argument("--spirula",action="store_true")
    p.add_argument("--sh-degree",default="auto",choices=["auto","0","1","2"])
    a=p.parse_args();build(a.video,a.output,a.config,a.gpu,a.prepared,a.geometry_only,a.spirula,a.sh_degree)

if __name__=="__main__":main()
