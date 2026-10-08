"""Multi-view fused initialization. Evaluation frames never supply seeds."""
import json
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from plyfile import PlyData,PlyElement
from .geometry import backproject,voxel_fuse,evidence,project
from .postprocess import rows_from_data
from .io import independent

def owners(xyz,boxes):
    owner=np.full(len(xyz),-1,np.int64)
    for j,b in enumerate(boxes['current'],len(boxes['old'])):
        pose=np.asarray(b['pose']);bounds=np.asarray(b['bounds']).reshape(3,2)
        q=(xyz-pose[:3,3])@np.linalg.inv(pose[:3,:3]).T
        inside=((q>=bounds[:,0])&(q<=bounds[:,1])).all(1)
        owner[(owner<0)&inside]=j
    return owner

def fused_initialization(job,candidate,lineage,boxes,out,base_renders=None,voxel=.008):
    d=dict(np.load(job/'views.npz'))
    split=json.loads((job/'refit_report.json').read_text())
    rows=rows_from_data(d);ids=independent(rows,split['train_views'],.15)
    points=[];colors=[]
    for i in ids:
        p,c=backproject(rows[i]);inside=owners(p,boxes)>=0
        points.append(p[inside]);colors.append(c[inside])
    if not sum(map(len,points)):raise ValueError('no_local_training_geometry')
    xyz,rgb=voxel_fuse(np.concatenate(points),np.concatenate(colors),voxel)
    support,free,_=evidence(xyz,rows,ids,.04)
    keep=(support>=3)&(free<=1)&(owners(xyz,boxes)>=0)
    novelty=np.zeros(len(xyz),np.int16)
    if base_renders is None:raise ValueError('base_renders_required_to_avoid_duplicate_surfaces')
    previous=dict(np.load(base_renders))
    if previous['depth'].shape!=d['depths'].shape:raise ValueError('base_render_shape_mismatch')
    for i in ids:
        ix,u,v,z=project(xyz,rows[i])
        observed=rows[i]['valid'][v,u]&(np.abs(rows[i]['depth'][v,u]-z)<=.04)
        changed=(previous['alpha'][i,v,u]<.5)|(np.abs(previous['depth'][i,v,u]/np.maximum(previous['alpha'][i,v,u],1e-4)-z)>.06)
        novelty[ix]+=observed&changed
    keep &= novelty>=2
    proposed=len(xyz);xyz=xyz[keep];rgb=rgb[keep]
    if len(xyz)<500:raise ValueError('insufficient_multiview_fused_geometry')
    if len(xyz)>150000:raise ValueError('local_geometry_too_large')
    tree=cKDTree(xyz);dist,ix=tree.query(xyz,k=min(12,len(xyz)))
    delta=xyz[ix]-xyz[ix].mean(1)[:,None,:]
    cov=np.einsum('nki,nkj->nij',delta,delta)/delta.shape[1]
    values,axes=np.linalg.eigh(cov)
    # Tangent axes span the local surface; thin normal axis avoids spherical dots.
    mat=axes[:,:,[2,1,0]].copy()
    neg=np.linalg.det(mat)<0;mat[neg,:,2]*=-1
    quat=Rotation.from_matrix(mat).as_quat()[:,[3,0,1,2]]
    spacing=np.clip(np.median(dist[:,1:5],axis=1),.003,.016)
    planar=(values[:,0]<.2*np.maximum(values[:,1],1e-9))&(dist[:,-1]<.06)
    scale=np.column_stack([spacing,spacing,np.where(planar,spacing*.3,spacing)])
    v=PlyData.read(str(candidate),mmap=True)['vertex'].data
    old=np.asarray(lineage['original_ids'])>=0
    seed=np.zeros(len(xyz),dtype=v.dtype)
    for k,n in enumerate(('x','y','z')):seed[n]=xyz[:,k]
    for k in range(3):
        seed['f_dc_'+str(k)]=(np.clip(rgb[:,k],0,1)-.5)/.28209479177387814
        seed['scale_'+str(k)]=np.log(scale[:,k])
    for k in range(4):seed['rot_'+str(k)]=quat[:,k]
    seed['opacity']=0 # sigmoid=.5; optimize jointly rather than opaque one-view seeds
    dest=out/'fused_initialization.ply'
    merged=np.concatenate([v[old],seed])
    PlyData([PlyElement.describe(merged,'vertex')],text=False).write(str(dest))
    new_lineage=dict(original_ids=np.concatenate([lineage['original_ids'][old],np.full(len(xyz),-1,np.int64)]),
        owner=np.concatenate([lineage['owner'][old],owners(xyz,boxes)]))
    np.savez_compressed(out/'fused_lineage.npz',**new_lineage)
    report=dict(method='train_only_multiview_voxel_fusion_pca_surface_initialization',voxel_m=voxel,
        support_tolerance_m=.04,minimum_support_views=3,maximum_free_contradictions=1,
        training_views=ids,proposed_voxels=proposed,accepted_gaussians=len(xyz),
        planar_fraction=float(planar.mean()),replaced_new_gaussians=int((~old).sum()),
        minimum_novel_views=2,base_renders=str(base_renders),evaluation_used=False,original_rows_byte_identical=True)
    (out/'initialization_report.json').write_text(json.dumps(report,indent=2))
    return dest,new_lineage
