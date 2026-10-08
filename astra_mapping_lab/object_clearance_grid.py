"""Same-object 3D support plus conservative multi-radius grid clearance."""
import json,numpy as np
from pathlib import Path
from PIL import Image,ImageDraw
from scipy.ndimage import binary_closing,label,maximum_filter,distance_transform_edt
from scipy.spatial import ConvexHull
from grid_common import classes
from recording import atomic
VERSION='object_clearance_binary_v1'

def preserve_cores(free,proposed,radii):
    distance=distance_transform_edt(free);cores={r:distance>=r for r in radii}
    labs,n=label(proposed,np.ones((3,3)));added=np.zeros_like(free);rejected=[]
    for k in range(1,n+1):
        patch=labs==k;trial=added|patch;dist=distance_transform_edt(free&~trial)
        lost={str(r):int((cores[r]&~(dist>=r)).sum()) for r in radii}
        if any(lost.values()):rejected.append(dict(cells=int(patch.sum()),lost_centers=lost))
        else:added|=patch
    return added,rejected

def supported_proposals(assembly, regions, meta, shape, floor, occupied, allowed, model_veto, sensor_hits):
    """Propose short gaps within the same object and height layer only."""
    h,w=shape;occ=occupied
    yaw=meta['origin'][2];c,t=np.cos(yaw),np.sin(yaw)
    proposed=np.zeros_like(occ);owners={}
    for reg in regions:
     name=reg['id']
     if not name.startswith('current_') or name not in assembly:continue
     p=assembly[name];p=p[(p[:,2]>=floor+.15)&(p[:,2]<=floor+1.8)]
     xy=(p[:,:2]-meta['origin'][:2])@np.array([[c,-t],[t,c]]);uv=np.floor(xy/meta['resolution']).astype(int);uv[:,1]=h-1-uv[:,1];ok=(uv[:,0]>=0)&(uv[:,0]<w)&(uv[:,1]>=0)&(uv[:,1]<h);uv=uv[ok];p=p[ok]
     if len(uv)<3 or np.linalg.matrix_rank(uv-uv.mean(0))<2:continue
     hull=ConvexHull(uv);im=Image.new('L',(w,h));ImageDraw.Draw(im).polygon([tuple(v) for v in uv[hull.vertices]],fill=1);scope=np.array(im).astype(bool)&allowed
     candidate=binary_closing(occ&scope,np.ones((3,3)))&scope&~occ
     # Require an actual point cluster in the gap and nearby occupied points from
     # the SAME reviewed object in the SAME height layer, not just its 2D hull.
     supported=np.zeros_like(occ)
     bins=np.floor((p[:,2]-floor)/.1).astype(int)
     for z in np.unique(bins):
      v=uv[bins==z];counts=np.zeros_like(occ,dtype=int);np.add.at(counts,(v[:,1],v[:,0]),1)
      layer=counts>=3
      adjacent=maximum_filter((layer&occ).astype(np.uint8),size=3)>0
      supported|=layer&adjacent
     patch=candidate&supported&(model_veto|(sensor_hits>=3))
     proposed|=patch;owners[name]=int(patch.sum())
    return proposed,owners


def apply(context,directory,report):
    import yaml
    d=Path(directory);cfg=json.loads(Path(context).read_text());fusion=Path(cfg['output'])
    a=np.array(Image.open(d/'map.png').convert('L'));meta=yaml.safe_load((d/'map.yaml').read_text());allowed=np.array(Image.open(d/'allowed_scope.png'))>0
    occ,free,_=classes(a,meta)
    with np.load(d/'grid_evidence.npz') as archive:
        sensor_hits=archive['sensor_hit_views']
    with np.load(Path(report['model_seed'])/'grid_evidence.npz') as archive:
        model_veto=archive['erasure_veto']
    regions=json.loads((fusion/'artifacts/context_scope.json').read_text())['regions']
    with np.load(fusion/'assembly_support.npz') as archive:
        assembly={r['id']:archive[r['id']] for r in regions if r['id'].startswith('current_') and r['id'] in archive}
    floor=json.loads((Path(cfg['alignment'])/'artifacts/floor.json').read_text())['floor_z']
    proposed,owners=supported_proposals(assembly,regions,meta,a.shape,floor,occ,allowed,model_veto,sensor_hits)
    radii=[r/meta['resolution'] for r in (.1,.15,.2,.25,.3)]
    added,rejected=preserve_cores(free,proposed,radii)
    result=a.copy();result[added]=255 if meta.get('negate',0) else 0
    assert np.array_equal(result[~allowed],a[~allowed])
    old=np.array(Image.open(d/'original_map.png').convert('L'));old_occ,_,_=classes(old,meta);oc,fr,un=classes(result,meta)
    report.update(changed_cells=int((result!=old).sum()),added_obstacles=int((oc&~old_occ).sum()),cleared_obstacles=int((old_occ&fr).sum()))
    from grid_common import fragmentation
    report['fragmentation_final']=fragmentation(oc,allowed)
    report['binary_export'].update(free_cells=int(fr.sum()),blocked_cells=int(oc.sum()),unknown_cells=int(un.sum()))
    before_distance=distance_transform_edt(free)
    after_distance=distance_transform_edt(fr)
    audit=dict(version=VERSION,proposed_cells=int(proposed.sum()),added_cells=int(added.sum()),removed_cells=0,rejected=rejected,objects=owners,
        reference_radii_map_units=[.1,.15,.2,.25,.3],reference_core_losses={str(r):int(((before_distance>=r)&~(after_distance>=r)).sum()) for r in radii},
        actual_robot_clearance_verified=False,metric_scale_verified=False,published=False)
    report['object_clearance']=audit
    Image.fromarray(result).save(d/'map.png');atomic(d/'object_clearance.json',audit);np.savez_compressed(d/'object_clearance_evidence.npz',added=added,proposed=proposed)
    atomic(d/'binary_export.json',report['binary_export'])
    return report
