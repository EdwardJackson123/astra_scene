"""Prune only Gaussian supports contradicted by independent training RGB-D views."""
import argparse,json,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np


def joint_clearance(votes,support,min_views=4):
    """Every support probe needs clearance; any positive support vetoes removal."""
    votes=np.asarray(votes);support=np.asarray(support)
    return (votes>=min_views).all(axis=0)&(support==0).all(axis=0)


def run(context,baseline,dest):
    from connected_map_update import setup,write
    from geometry_boundary import gaussian_extents
    from scipy.spatial.transform import Rotation
    from plyfile import PlyData,PlyElement
    from PIL import Image
    started=time.monotonic();cfg=json.loads(Path(context).read_text());fusion=Path(cfg['output']);dest=Path(dest);dest.mkdir(parents=True,exist_ok=False)
    ctx=setup(SimpleNamespace(source=Path(cfg['source']),alignment_dir=Path(cfg['alignment']),output=dest/'inputs',base_map=Path(cfg['base_map']),gpu=1))
    root,source,alignment,_,base,data,registration,rows,ids,held=ctx
    from astra_scene.geometry import grid_cells,evidence
    from artifact_cache import digest
    src=fusion/'artifacts/scene.ply';source_hash=digest(src);ply=PlyData.read(src,mmap=True);v=ply['vertex'].data
    xyz=np.column_stack([v[n] for n in ('x','y','z')]);meta=data['map_metadata'];allowed=np.array(Image.open(baseline/'allowed_scope.png'))>0
    floor=json.loads((alignment/'artifacts/floor.json').read_text())['floor_z'];views=json.loads((baseline/'navigation_update.json').read_text())['geometric_evidence_views']
    if set(views)&set(held):raise ValueError('holdout leakage')
    def permitted(p):
        uv,ok=grid_cells(p,meta,allowed.shape);ix=np.flatnonzero(ok);ok[ix]&=allowed[uv[ix,1],uv[ix,0]]
        return ok&(p[:,2]>=floor+.15)&(p[:,2]<=floor+1.8)
    scope_ids=np.flatnonzero(permitted(xyz));p=xyz[scope_ids]
    support,clear,_=evidence(p,rows,views,.12)
    candidates=scope_ids[(clear>=4)&(support==0)]
    print('CANDIDATES',len(candidates),'/',len(scope_ids),flush=True)
    q=np.column_stack([v['rot_'+str(i)][candidates] for i in (1,2,3,0)]).astype(float)
    scales=np.exp(np.column_stack([v['scale_'+str(i)][candidates] for i in range(3)]))
    valid=np.isfinite(q).all(1)&np.isfinite(scales).all(1)&(np.linalg.norm(q,axis=1)>1e-8)
    candidates=candidates[valid];q=q[valid];scales=scales[valid]
    q/=np.linalg.norm(q,axis=1,keepdims=True);rotation=Rotation.from_quat(q).as_matrix()
    # Wide splats need full covariance integration; this experiment preserves them.
    extent=2*np.sqrt(np.sum(rotation*rotation*scales[:,None,:]**2,axis=2));bounded=(extent.max(1)<=.35)
    candidates=candidates[bounded];rotation=rotation[bounded];scales=scales[bounded];p=xyz[candidates]
    probe_free=[];probe_support=[];all_inside=np.ones(len(p),bool)
    probes=[p]
    for axis in range(3):
        offset=2*scales[:,axis,None]*rotation[:,:,axis]
        probes.extend([p-offset,p+offset])
    for j,probe in enumerate(probes):
        all_inside &= permitted(probe)
        hs,fc,_=evidence(probe,rows,views,.12);probe_free.append(fc);probe_support.append(hs)
        print('SUPPORT_PROBE',j+1,'/',len(probes),flush=True)
    probe_free=np.array(probe_free);probe_support=np.array(probe_support)
    removal=joint_clearance(probe_free,probe_support)&all_inside;removed_ids=candidates[removal]
    keep=np.ones(len(v),bool);keep[removed_ids]=False
    output=dest/'scene_cleaned.ply';PlyData([PlyElement.describe(v[keep],'vertex')],text=False,byte_order=ply.byte_order,comments=list(ply.comments),obj_info=list(ply.obj_info)).write(str(output))
    assert digest(src)==source_hash,'source changed during experiment'
    report=dict(source_ply=str(src),source_sha256=source_hash,candidate_ply=str(output),candidate_sha256=digest(output),
        original_gaussians=len(v),retained_gaussians=int(keep.sum()),removed_gaussians=len(removed_ids),scope_gaussians=len(scope_ids),
        center_clearance_candidates=len(candidates),training_views=views,held_out_used_for_selection=False,
        policy='center and six rotated 2-sigma principal-axis probes; each >=4 free observations and zero surface supports; all probes inside allowed ROI/device scope and robot-height band; preserve broad Gaussians',
        modified_parameters=False,removed_only=True,published=False,navigation_deployment_accepted=False,
        limitations=['seven support probes do not cover the entire Gaussian volume','depth alignment has unresolved absolute-scale conflict','same-capture development experiment, independent revisit required'],elapsed_s=time.monotonic()-started)
    write(dest/'cleanup.json',report)
    np.savez_compressed(dest/'removed_evidence.npz',indices=removed_ids,centers=xyz[removed_ids],free_votes=probe_free[:,removal],surface_votes=probe_support[:,removal])
    print('RESULT',json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--context',type=Path,required=True);p.add_argument('--baseline',type=Path,required=True);p.add_argument('--destination',type=Path,required=True);a=p.parse_args();run(a.context,a.baseline,a.destination)
