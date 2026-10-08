"""Rebuild a measured local grid; no inherited additions or morphology."""
import argparse,json,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from navigation_candidate import classes,constrained_result


def fuse_view_voxels(parts,size=.05):
    """One vote per camera per voxel, independent of pixel sampling density."""
    keys=[];centers=[]
    for p in parts:
        if not len(p):continue
        k,inv=np.unique(np.floor(p/size).astype(np.int64),axis=0,return_inverse=True)
        n=np.bincount(inv);q=np.column_stack([np.bincount(inv,weights=p[:,j])/n for j in range(3)])
        keys.append(k);centers.append(q)
    if not keys:return np.empty((0,3)),np.empty(0,int)
    k,inv=np.unique(np.concatenate(keys),axis=0,return_inverse=True);n=np.bincount(inv);p=np.concatenate(centers)
    return np.column_stack([np.bincount(inv,weights=p[:,j])/n for j in range(3)]),n


def run(context,baseline,dest):
    from connected_map_update import setup,write
    from PIL import Image
    from scipy.spatial import cKDTree
    import yaml
    started=time.monotonic();cfg=json.loads(Path(context).read_text());fusion=Path(cfg['output']);dest=Path(dest);dest.mkdir(parents=True,exist_ok=False)
    ctx=setup(SimpleNamespace(source=Path(cfg['source']),alignment_dir=Path(cfg['alignment']),output=dest/'inputs',base_map=Path(cfg['base_map']),gpu=1))
    root,source,alignment,_,base,data,registration,rows,ids,held=ctx
    from astra_scene.geometry import grid_cells,grid_centers,backproject,evidence,project
    oldreport=json.loads((baseline/'navigation_update.json').read_text());views=oldreport['geometric_evidence_views'];oldviews=oldreport['surface_source_views']
    if set(views)&set(held):raise ValueError('holdout leakage')
    original=np.array(Image.open(source/'map.png').convert('L'));meta=data['map_metadata'];shape=original.shape
    allowed=np.array(Image.open(baseline/'allowed_scope.png'))>0
    prior=np.load(baseline/'cell_evidence.npz');semantic=prior['semantic']
    floor=json.loads((alignment/'artifacts/floor.json').read_text())['floor_z']
    def permitted(p):
        uv,ok=grid_cells(p,meta,shape);ix=np.flatnonzero(ok);ok[ix]&=allowed[uv[ix,1],uv[ix,0]]
        return uv,ok
    def clip(p):
        uv,ok=permitted(p);return p[ok&(p[:,2]>=floor+.15)&(p[:,2]<=floor+1.8)]
    current_parts=[];old_parts=[]
    for i in views:current_parts.append(clip(backproject(rows[i],stride=2)[0]))
    for i in oldviews:
        with np.load(fusion/'old_renders'/f'{i:03d}.npz') as r:
            row=dict(rows[i],depth=r['depth'],valid=(r['alpha']>.9)&(r['depth']>.4)&(r['depth']<7)&rows[i]['keep'])
            old_parts.append(clip(backproject(row,stride=2)[0]))
    current,current_votes=fuse_view_voxels(current_parts);old,old_votes=fuse_view_voxels(old_parts)
    print('VOXELS',len(current),len(old),flush=True)
    hit,clear,_=evidence(current,rows,views,.07)
    accepted=(current_votes>=3)&(hit>=3)&(clear<=1)
    physical=current[accepted]
    # Current evidence is rebuilt; old additions cannot bypass the new checks.
    occupied=np.zeros(shape,bool);uv,ok=permitted(physical);occupied[uv[ok,1],uv[ok,0]]=True
    # Weak but repeated positive evidence may veto erasure without being strong
    # enough to insert an obstacle. Denoising additions cannot authorize removal.
    veto_points=current[(hit>=2)&(clear<=1)]
    veto=np.zeros(shape,bool);uv,ok=permitted(veto_points);veto[uv[ok,1],uv[ok,0]]=True
    supported_old=old[old_votes>=3]
    hs,fc,_=evidence(supported_old,rows,views,.10);uv,ok=permitted(supported_old);flat=uv[ok,1]*shape[1]+uv[ok,0]
    total=np.bincount(flat,minlength=original.size).reshape(shape)
    cleared=np.bincount(flat,weights=(fc[ok]>=4)&(hs[ok]==0),minlength=original.size).reshape(shape)
    stable=np.bincount(flat,weights=hs[ok]>=2,minlength=original.size).reshape(shape)
    oc,_,_=classes(original,meta);eligible=oc&semantic&(total>=2)&(cleared==total)&~veto
    y,x=np.where(eligible);xy=grid_centers(np.column_stack([x,y]),meta,shape);ground=np.column_stack([xy,np.full(len(x),floor)])
    ground_votes=np.zeros(shape,int)
    for i in views:
        ix,u,v,z=project(ground,rows[i]);r=rows[i];ground_votes[y[ix],x[ix]]+=r['valid'][v,u]&r['keep'][v,u]&(np.abs(r['depth'][v,u]-z)<.10)
    free=eligible&(ground_votes>=3)
    scope=json.loads((fusion/'artifacts/context_scope.json').read_text());assembly=np.load(fusion/'assembly_support.npz')
    tree=cKDTree(np.concatenate([assembly[r['id']] for r in scope['regions']]))
    addpoints=physical[tree.query(physical,workers=4)[0]<.18];add=np.zeros(shape,bool);uv,ok=permitted(addpoints);add[uv[ok,1],uv[ok,0]]=True
    proposed=original.copy();proposed[free]=1 if meta.get('negate',0) else 254;proposed[add]=255 if meta.get('negate',0) else 0
    result,counts=constrained_result(original,proposed,allowed,meta)
    held_obstacle=np.zeros(shape,bool);held_ground=np.zeros(shape,bool)
    for i in held:
        p,_=backproject(rows[i],stride=2);uv,ok=permitted(p)
        m=ok&(p[:,2]>=floor+.15)&(p[:,2]<=floor+1.8);held_obstacle[uv[m,1],uv[m,0]]=True
        m=ok&(np.abs(p[:,2]-floor)<.10);held_ground[uv[m,1],uv[m,0]]=True
    conflicts=int((free&held_obstacle).sum())
    report=dict(**counts,policy='new local RGB-D occupancy; >=3 independent endpoint voxel votes + depth support; historical surface voxels require >=3 render views; all reliable old voxels cleared in >=4 views + measured ground; keep unobserved original',
        current_raw_voxels=len(current),current_accepted_voxels=len(physical),current_veto_voxels=len(veto_points),old_raw_voxels=len(old),old_reliable_voxels=len(supported_old),
        cleared_with_holdout_obstacle=conflicts,cleared_with_holdout_ground=int((free&held_ground).sum()),
        published=False,navigation_deployment_accepted=False,candidate_audit_passed=conflicts==0,held_out_used_for_selection=False,
        inherited_additions=False,uncertain_edits_applied=False,metric_scale_conflict=registration.get('metric_scale_conflict',True),
        geometric_evidence_views=views,elapsed_s=time.monotonic()-started)
    for name,a in [('map.png',result),('original_map.png',original),('allowed_scope.png',np.uint8(allowed)*255)]:Image.fromarray(a).save(dest/name)
    meta={k:meta[k] for k in ('resolution','origin','negate','occupied_thresh','free_thresh','mode') if k in meta};meta['image']='map.png'
    (dest/'map.yaml').write_text(yaml.safe_dump(meta,sort_keys=False));write(dest/'navigation_update.json',report)
    np.savez_compressed(dest/'cell_evidence.npz',total=total,cleared=cleared,stable=stable,current_obstacle=occupied,erasure_veto=veto,ground_views=ground_votes,free=free,semantic=semantic,held_obstacle=held_obstacle,held_ground=held_ground)
    np.savez_compressed(dest/'measured_voxels.npz',points=current,votes=current_votes,depth_support=hit,free_contradictions=clear,accepted=accepted)
    print('RESULT',json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--context',type=Path,required=True);p.add_argument('--baseline',type=Path,required=True);p.add_argument('--destination',type=Path,required=True);a=p.parse_args();run(a.context,a.baseline,a.destination)
