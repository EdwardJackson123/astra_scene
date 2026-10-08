"""Use multiview surfaces rendered from the final 3DGS, not raw Gaussian centers."""
from gpu_config import gpu_index
import argparse,json,time,fcntl
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from grid_common import classes,constrained_result
from rgbd_grid_submap import fuse_view_voxels
from gaussian_grid import supported_voxels


def run(context,baseline,dest,model_ply=None):
    from connected_map_update import setup,write
    from PIL import Image
    import yaml,cv2
    started=time.monotonic();cfg=json.loads(Path(context).read_text());fusion=Path(cfg['output']);dest=Path(dest);dest.mkdir(parents=True,exist_ok=False)
    ctx=setup(SimpleNamespace(source=Path(cfg['source']),alignment_dir=Path(cfg['alignment']),output=dest/'inputs',base_map=Path(cfg['base_map']),gpu=gpu_index()))
    root,source,alignment,_,base,data,registration,rows,ids,held=ctx
    from astra_scene.geometry import grid_cells,backproject,evidence
    from astra_scene.postprocess import render_map
    from artifact_cache import digest
    baseline_report=json.loads((baseline/'navigation_update.json').read_text())
    views=baseline_report.get('geometric_evidence_views',baseline_report.get('train_views'))
    if not views:raise ValueError('baseline missing independent training views')
    if set(views)&set(held):raise ValueError('holdout leakage')
    meta=data['map_metadata'];original=np.array(Image.open(source/'map.png').convert('L'));shape=original.shape
    allowed=np.array(Image.open(baseline/'allowed_scope.png'))>0;floor=json.loads((alignment/'artifacts/floor.json').read_text())['floor_z']
    ply=Path(model_ply) if model_ply else fusion/'artifacts/scene.ply';model_rows={};obstacles=[];grounds=[];samples=dest/'render_samples';samples.mkdir()
    def cells(p):
        uv,ok=grid_cells(p,meta,shape);ix=np.flatnonzero(ok);ok[ix]&=allowed[uv[ix,1],uv[ix,0]];return uv,ok
    with (root/'runs/gpu.lock').open('a') as lock:
        # Do not displace another reconstruction job.
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        from vps.refinement.gaussian_renderer import GsplatGaussianRenderer
        renderer=GsplatGaussianRenderer('cuda')
        for j,i in enumerate(views):
            r=render_map(renderer,ply,rows[i]);depth=r['depth'];valid=np.isfinite(depth)&(depth>.4)&(depth<7)&(r['alpha']>.9)&rows[i]['keep']
            span=cv2.dilate(depth,np.ones((3,3),np.uint8))-cv2.erode(depth,np.ones((3,3),np.uint8))
            valid=(cv2.erode(valid.astype(np.uint8),np.ones((3,3),np.uint8))>0)&(span<.07+.015*depth)
            model_rows[i]=dict(rows[i],depth=depth,valid=valid,clear=valid)
            p,_=backproject(model_rows[i],stride=2);uv,ok=cells(p);p=p[ok]
            obstacles.append(p[(p[:,2]>=floor+.15)&(p[:,2]<=floor+1.8)])
            grounds.append(p[np.abs(p[:,2]-floor)<.10])
            if j in (0,len(views)//2,len(views)-1):Image.fromarray(r['rgb']).save(samples/f'{i:03d}.jpg')
            if j%10==0:print('RENDER',j+1,'/',len(views),flush=True)
        del renderer
    points,votes=fuse_view_voxels(obstacles);hit,clear,_=evidence(points,model_rows,views,.10)
    keep=(votes>=3)&(hit>=3)&(clear<=1);p=points[keep]
    connected,connectivity=supported_voxels(p,resolution=meta['resolution'],min_component=4);p=p[connected]
    selected_indices=np.flatnonzero(keep)[connected]
    occupied=np.zeros(shape,bool);uv,ok=cells(p);occupied[uv[ok,1],uv[ok,0]]=True
    # A surface too weak to insert may still contradict a deletion. Avoid
    # interpreting removal of a small component as evidence of physical space.
    veto=np.zeros(shape,bool);uv,ok=cells(points[(hit>=2)&(clear<=1)]);veto[uv[ok,1],uv[ok,0]]=True
    floor_views=np.zeros(shape,int)
    for p in grounds:
        uv,ok=cells(p);uv=np.unique(uv[ok],axis=0);floor_views[uv[:,1],uv[:,0]]+=1
    covered=(floor_views>=3)&allowed;proposed=original.copy();negate=meta.get('negate',0)
    _,already_free,_=classes(original,meta)
    proposed[covered&~veto&~already_free]=1 if negate else 254;proposed[occupied]=255 if negate else 0
    result,counts=constrained_result(original,proposed,allowed,meta);oc,_,_=classes(original,meta);_,nf,_=classes(result,meta);cleared=oc&nf
    held_obstacle=np.zeros(shape,bool);held_floor=np.zeros(shape,bool)
    for i in held:
        p,_=backproject(rows[i],stride=2);uv,ok=cells(p);m=ok&(p[:,2]>=floor+.15)&(p[:,2]<=floor+1.8);held_obstacle[uv[m,1],uv[m,0]]=True
        m=ok&(np.abs(p[:,2]-floor)<.10);held_floor[uv[m,1],uv[m,0]]=True
    conflicts=cleared&held_obstacle
    report=dict(**counts,source='multiview rendered surfaces of final fused scene.ply',source_ply=str(ply),source_ply_sha256=digest(ply),
        render_views=views,raw_surface_voxels=len(points),accepted_surface_voxels=int(connected.sum()),observed_floor_cells=int(covered.sum()),connectivity=connectivity,
        cleared_with_holdout_obstacle=int(conflicts.sum()),cleared_with_holdout_ground=int((cleared&held_floor).sum()),
        candidate_audit_passed=not bool(conflicts.any()),held_out_used_for_selection=False,published=False,navigation_deployment_accepted=False,
        state='GEOMETRY_PREVIEW_NOT_NAVIGATION_APPROVED',metric_scale_conflict=registration.get('metric_scale_conflict',True),
        policy='3DGS rendered depth/alpha; >=3 independent model views and geometric consistency plus connected 3D support for additions; weak repeated model surface vetoes erasure; floor observed in >=3 views permits replacement; preserve original outside model coverage; no inherited additions or old retirement rules',
        limitations=['model-consistent surfaces are not necessarily physically correct','rendered floor does not certify full robot clearance','same-capture development audit, not independent revisit'],elapsed_s=time.monotonic()-started)
    for name,a in [('map.png',result),('original_map.png',original),('allowed_scope.png',np.uint8(allowed)*255)]:Image.fromarray(a).save(dest/name)
    config={k:meta[k] for k in ('resolution','origin','negate','occupied_thresh','free_thresh','mode') if k in meta};config['image']='map.png'
    (dest/'map.yaml').write_text(yaml.safe_dump(config,sort_keys=False));write(dest/'navigation_update.json',report)
    np.savez_compressed(dest/'grid_evidence.npz',occupied=occupied,erasure_veto=veto,cleared=cleared,floor_views=floor_views,held_obstacle=held_obstacle,held_floor=held_floor,conflict=conflicts)
    np.savez_compressed(dest/'model_surfaces.npz',points=points,view_votes=votes,model_support=hit,model_free=clear,selected_indices=selected_indices)
    print('RESULT',json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--context',type=Path,required=True);p.add_argument('--baseline',type=Path,required=True);p.add_argument('--destination',type=Path,required=True);p.add_argument('--model-ply',type=Path);a=p.parse_args();run(a.context,a.baseline,a.destination,a.model_ply)
