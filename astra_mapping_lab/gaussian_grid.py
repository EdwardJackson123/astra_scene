"""Rasterize the final fused Gaussian model into a local navigation-grid preview."""
import argparse,json,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from navigation_candidate import classes,constrained_result


def supported_voxels(points,resolution=.05,min_component=4):
    """Drop isolated 3-D components, retaining connected thin structures."""
    from scipy.ndimage import label
    if not len(points):return np.zeros(0,bool),dict(voxels=0,components=0,rejected_voxels=0)
    vox,inv=np.unique(np.floor(points/resolution).astype(np.int64),axis=0,return_inverse=True)
    lo=vox.min(0);shape=vox.max(0)-lo+1
    if np.prod(shape)>30_000_000:raise ValueError('local cloud extent unexpectedly large')
    volume=np.zeros(shape,bool);v=vox-lo;volume[tuple(v.T)]=True
    labels,n=label(volume,np.ones((3,3,3),int));ids=labels[tuple(v.T)];counts=np.bincount(ids)
    keep=counts[ids]>=min_component
    return keep[inv],dict(voxels=len(vox),components=int(n),rejected_voxels=int((~keep).sum()))


def run(context,baseline,dest):
    from connected_map_update import setup,write
    from geometry_boundary import gaussian_extents
    from PIL import Image
    from plyfile import PlyData
    import yaml
    started=time.monotonic();cfg=json.loads(Path(context).read_text());fusion=Path(cfg['output']);dest=Path(dest);dest.mkdir(parents=True,exist_ok=False)
    ctx=setup(SimpleNamespace(source=Path(cfg['source']),alignment_dir=Path(cfg['alignment']),output=dest/'inputs',base_map=Path(cfg['base_map']),gpu=1))
    root,source,alignment,_,base,data,registration,rows,ids,held=ctx
    from astra_scene.geometry import grid_cells,grid_centers,backproject,project
    from artifact_cache import digest
    meta=data['map_metadata'];original=np.array(Image.open(source/'map.png').convert('L'));shape=original.shape
    allowed=np.array(Image.open(baseline/'allowed_scope.png'))>0
    floor=json.loads((alignment/'artifacts/floor.json').read_text())['floor_z']
    ply=fusion/'artifacts/scene.ply';v=PlyData.read(ply,mmap=True)['vertex'].data
    def cells(p):
        uv,ok=grid_cells(p,meta,shape);ix=np.flatnonzero(ok);ok[ix]&=allowed[uv[ix,1],uv[ix,0]];return uv,ok
    points=np.column_stack([v[n] for n in ('x','y','z')]);uv,ok=cells(points)
    height=(points[:,2]>=floor+.15)&(points[:,2]<=floor+1.8)
    indices=np.flatnonzero(ok&height&np.isfinite(points).all(1));raw=np.zeros(shape,bool);raw[uv[indices,1],uv[indices,0]]=True
    local=v[indices];alpha=1/(1+np.exp(-np.clip(local['opacity'],-60,60)))
    q=np.column_stack([local['rot_'+str(i)] for i in range(4)])
    finite=np.isfinite(q).all(1)&(np.linalg.norm(q,axis=1)>1e-8)
    for n in ('scale_0','scale_1','scale_2'):finite &= np.isfinite(local[n])
    chosen=np.flatnonzero(finite&(alpha>=.5));extent=gaussian_extents(local[chosen],sigma=2.)
    valid=(np.sort(extent,axis=1)[:,1]<=.20)&(extent.max(1)<=.60)
    chosen=chosen[valid];p=points[indices[chosen]]
    consistent,connectivity=supported_voxels(p,resolution=meta['resolution'],min_component=4)
    selected=p[consistent];occupied=np.zeros(shape,bool);uv,ok=cells(selected);occupied[uv[ok,1],uv[ok,0]]=True
    # The model defines occupancy. Empty cells are a model-derived preview,
    # not evidence that an unobserved physical volume is free.
    proposed=original.copy();negate=meta.get('negate',0)
    proposed[allowed]=1 if negate else 254;proposed[occupied]=255 if negate else 0
    result,counts=constrained_result(original,proposed,allowed,meta)
    oc,_,_=classes(original,meta);_,nf,_=classes(result,meta);cleared=oc&nf
    # Independent sensor observations audit the produced map; never select pixels.
    held_obstacle=np.zeros(shape,bool);held_floor=np.zeros(shape,bool)
    for i in held:
        p,_=backproject(rows[i],stride=2);uv,ok=cells(p)
        m=ok&(p[:,2]>=floor+.15)&(p[:,2]<=floor+1.8);held_obstacle[uv[m,1],uv[m,0]]=True
        m=ok&(np.abs(p[:,2]-floor)<.10);held_floor[uv[m,1],uv[m,0]]=True
    conflict=cleared&held_obstacle
    report=dict(**counts,source='final fused scene.ply',source_ply=str(ply),source_ply_sha256=digest(ply),
        total_gaussians=len(v),height_and_scope_gaussians=len(indices),opacity_shape_gaussians=len(chosen),retained_gaussians=len(selected),
        raw_occupied_cells=int(raw.sum()),filtered_occupied_cells=int(occupied.sum()),connectivity=connectivity,
        parameters=dict(height_above_floor=[.15,1.8],min_opacity=.5,max_second_extent=.20,max_extent=.60,min_component_voxels=4),
        cleared_with_holdout_obstacle=int(conflict.sum()),cleared_with_holdout_ground=int((cleared&held_floor).sum()),
        held_out_used_for_selection=False,candidate_audit_passed=not bool(conflict.any()),
        state='GEOMETRY_PREVIEW_NOT_NAVIGATION_APPROVED',published=False,navigation_deployment_accepted=False,
        metric_scale_conflict=registration.get('metric_scale_conflict',True),floor_z=floor,
        limitations=['empty model projection does not prove free physical space','Gaussian opacity and connectedness are rendering/structure heuristics, not calibrated occupancy probability','height range and scale not robot calibrated','held-out data already used in development'],
        elapsed_s=time.monotonic()-started)
    for name,a in [('map.png',result),('original_map.png',original),('allowed_scope.png',np.uint8(allowed)*255),('raw_obstacles.png',np.uint8(raw)*255)]:Image.fromarray(a).save(dest/name)
    config={k:meta[k] for k in ('resolution','origin','negate','occupied_thresh','free_thresh','mode') if k in meta};config['image']='map.png'
    (dest/'map.yaml').write_text(yaml.safe_dump(config,sort_keys=False));write(dest/'navigation_update.json',report)
    np.savez_compressed(dest/'grid_evidence.npz',raw=raw,occupied=occupied,cleared=cleared,held_obstacle=held_obstacle,held_floor=held_floor,conflict=conflict)
    print('RESULT',json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--context',type=Path,required=True);p.add_argument('--baseline',type=Path,required=True);p.add_argument('--destination',type=Path,required=True);a=p.parse_args();run(a.context,a.baseline,a.destination)
