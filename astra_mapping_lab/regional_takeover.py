"""Single-owner regional replacement of an already trained reconstruction.

Semantic/connected footprints define the candidate, while a grid ROI only
limits permission. The new model is copied verbatim in owned columns; old
geometry is retired as a region, not retained behind each new surface.
This is a versioned visual-map preview, not proof that retired space is free.
"""
import argparse,json,shutil
from pathlib import Path
import numpy as np


def sample_field(points, field, origin, resolution):
    ij=np.floor((np.asarray(points)[:,:2]-origin)/resolution).astype(np.int64)
    valid=(ij[:,0]>=0)&(ij[:,1]>=0)&(ij[:,0]<field.shape[1])&(ij[:,1]<field.shape[0])
    result=np.zeros(len(ij),dtype=field.dtype)
    result[valid]=field[ij[valid,1],ij[valid,0]]
    return result


def owner(points, field, origin, resolution, z_range):
    p=np.asarray(points)
    return sample_field(p,field,origin,resolution).astype(bool)&(p[:,2]>=z_range[0])&(p[:,2]<=z_range[1])


def build(ctx, prior, coverage_mode='sensor'):
    import cv2
    from scipy.ndimage import distance_transform_edt
    from scipy.spatial import cKDTree
    from plyfile import PlyData,PlyElement
    from connected_map_update import write
    from astra_scene.geometry import project,voxel_fuse
    from astra_scene.io import sha256
    from map_update import roi_weights
    root,source,alignment,out,base,data,registration,rows,ids,held=ctx
    dest=out/'artifacts'
    if (dest/'scene.ply').exists():raise ValueError('new version required; candidate already exists')
    if json.loads((prior/'inputs.json').read_text())!=json.loads((out/'inputs.json').read_text()):
        raise ValueError('prior proposals belong to different map, alignment, ROI or observations')
    old=PlyData.read(base,mmap=True)['vertex'].data
    new=PlyData.read(alignment/'aligned_masked_sh0.ply',mmap=True)['vertex'].data
    xyz=lambda v:np.column_stack([v[k] for k in ('x','y','z')])
    op,np_=xyz(old),xyz(new)
    scope=json.loads((prior/'artifacts/context_scope.json').read_text())
    proposals=json.loads((prior/'analysis/evidence.json').read_text())['proposals']
    decisions={a['id']:a for a in json.loads((prior/'analysis/ai.json').read_text())['items']}
    support=np.load(prior/'component_support.npz')
    ownership=np.load(prior/'boxer_weights.npz')
    pools={'current':np_[ownership['new']>0], 'old':op[ownership['old']>0]}
    # Keep the previously nominated instances and their completed extents.
    # A single room-wide hull would silently fill unreviewed gaps, so every
    # instance is rasterized separately before taking its union.
    resolution=.05
    bounds=np.array([r['bounds'] for r in scope['regions']])
    margin=.6 if coverage_mode=='sensor' else 1.5
    origin=np.floor((bounds[:,0,:2].min(0)-margin)/resolution)*resolution
    end=bounds[:,1,:2].max(0)+margin
    shape=tuple(np.ceil((end-origin)/resolution).astype(int)[::-1])
    proposed=np.zeros(shape,np.uint8);records=[]
    for region in scope['regions']:
        key=region['id'];a=decisions[key]
        if a.get('ai_validation_failed') or a['decision']=='ignore':raise ValueError('invalid nominated scope')
        p=support[key];b=np.array(region['bounds'])
        # Sensor-supported footprint plus completed structure extrema, per
        # instance. Polygon completion is deliberately a region hypothesis.
        xy=p[:,:2];xy=xy[((xy>=b[0,:2]-.1)&(xy<=b[1,:2]+.1)).all(1)]
        pool=pools[region.get('evidence_side','current')]
        completed=pool[((pool>=b[0])&(pool<=b[1])).all(1)]
        xy=np.concatenate([xy,completed[::max(1,len(completed)//12000),:2]])
        if len(xy)<3:continue
        uv=np.rint((xy-origin)/resolution).astype(np.int32)
        mask=np.zeros(shape,np.uint8);cv2.fillConvexPoly(mask,cv2.convexHull(uv),1)
        # 20 cm context margin belongs to this assembly; it is not the ROI.
        mask=cv2.dilate(mask,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9)))
        proposed|=mask
        records.append(dict(id=key,ai_decision=a['decision'],bounds=region['bounds'],proposed_cells=int(mask.sum()),
            complete_region_hypothesis=True,publication_approved=False))
    yy,xx=np.indices(shape)
    centers=np.column_stack([origin[0]+(xx.ravel()+.5)*resolution,origin[1]+(yy.ravel()+.5)*resolution,np.zeros(xx.size)])
    permission=(roi_weights(centers,data['map_metadata'],data['roi'],.001)>0).reshape(shape)
    proposed&=permission.astype(np.uint8)
    semantic_core=proposed.copy()
    if coverage_mode=='visual':
        # Reach nearby observed background so that an object's clipping seam
        # does not run through its silhouette. Never expand to the whole ROI.
        proposed=cv2.dilate(proposed,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(41,41)))
        proposed&=permission.astype(np.uint8)
    floor=json.loads((alignment/'artifacts/floor.json').read_text())['floor_z']
    z_range=[floor-.25,floor+3.1]
    nominated=owner(np_,proposed,origin,resolution,z_range)
    points=np_[nominated & (new['opacity']>-2.2)]
    points,_=voxel_fuse(points,np.zeros_like(points),.07)
    votes=np.zeros(shape,np.int16)
    visibility_ids=ids
    render_cache=out/'fresh_renders'
    if coverage_mode=='visual':
        import fcntl,gc,torch
        from astra_scene.postprocess import render_map
        protocol=json.loads((prior/'boxer_protocol.json').read_text())
        visibility_ids=[x['aligned_view'] for x in protocol['view_mapping']]
        render_cache.mkdir(exist_ok=True)
        with (root/'runs/gpu.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            from vps.refinement.gaussian_renderer import GsplatGaussianRenderer
            renderer=GsplatGaussianRenderer('cuda')
            for i in visibility_ids:
                path=render_cache/f'{i:03d}.npz'
                if not path.exists():np.savez_compressed(path,**render_map(renderer,alignment/'aligned_masked_sh0.ply',rows[i]))
            del renderer;gc.collect();torch.cuda.empty_cache()
    for i in visibility_ids:
        ix,u,v,z=project(points,rows[i]);row=rows[i]
        if coverage_mode=='sensor':
            visible=row['keep'][v,u]&row['valid'][v,u]&(np.abs(row['depth'][v,u]-z)<.20)
        else:
            rendered=np.load(render_cache/f'{i:03d}.npz')
            photo=np.abs(rendered['rgb'][v,u]/255.-row['rgb'][v,u]).mean(1)<.18
            visible=row['keep'][v,u]&photo&(rendered['alpha'][v,u]>.8)&(np.abs(rendered['depth'][v,u]-z)<.35+.02*z)
        p=points[ix[visible]];ij=np.floor((p[:,:2]-origin)/resolution).astype(int)
        valid=(ij[:,0]>=0)&(ij[:,1]>=0)&(ij[:,0]<shape[1])&(ij[:,1]<shape[0]);ij=ij[valid]
        # One observation per view per column; Gaussian density cannot inflate it.
        present=np.zeros(shape,np.uint8);present[ij[:,1],ij[:,0]]=1
        votes+=cv2.dilate(present,np.ones((3,3),np.uint8)).astype(np.int16)
    covered=votes>=3
    field=proposed.astype(bool)&covered
    # Close only sub-cell slits, never bridge components across an unseen aisle.
    field=cv2.morphologyEx(field.astype(np.uint8),cv2.MORPH_CLOSE,np.ones((3,3),np.uint8)).astype(bool)&proposed.astype(bool)&covered
    nlabels,labels,stats,_=cv2.connectedComponentsWithStats(field.astype(np.uint8),8)
    for k in range(1,nlabels):
        if stats[k,cv2.CC_STAT_AREA]<16:field[labels==k]=False
    if not field.any():raise ValueError('no adequately observed regional coverage')
    fresh_owned=owner(np_,field,origin,resolution,z_range)&(roi_weights(np_,data['map_metadata'],data['roi'],.001)>0)
    old_owned=owner(op,field,origin,resolution,z_range)&(roi_weights(op,data['map_metadata'],data['roi'],.001)>0)
    ni=np.flatnonzero(fresh_owned)
    removed=np.flatnonzero(old_owned)
    old_keep=~old_owned;oi=np.flatnonzero(old_keep)
    candidate=np.empty(len(oi)+len(ni),dtype=old.dtype);candidate[:len(oi)]=old[oi]
    for name in old.dtype.names:candidate[name][len(oi):]=new[name][ni]
    assert not (owner(xyz(candidate[:len(oi)]),field,origin,resolution,z_range)&(roi_weights(op[oi],data['map_metadata'],data['roi'],.001)>0)).any()
    # The entire fresh contribution is copied exactly; no size clamps, sparse
    # surface filters, opacity fade or joint optimization damage its quality.
    for name in old.dtype.names:assert np.array_equal(candidate[name][len(oi):],new[name][ni])
    work=out/'context_refinement';work.mkdir(exist_ok=True)
    PlyData([PlyElement.describe(candidate,'vertex')],text=False).write(work/'initial.ply')
    # Geometry has a single owner. Only OLD static architectural color is
    # corrected in a narrow outer seam; new reconstruction remains immutable.
    outside_distance=distance_transform_edt(~field)*resolution
    distances=sample_field(op[oi],outside_distance,origin,resolution)
    seam=(distances>0)&(distances<=.25)&(roi_weights(op[oi],data['map_metadata'],data['roi'],.001)>0)
    seam&=(np.abs(op[oi,2]-floor)<.12)|((op[oi,2]>floor+2.65)&(op[oi,2]<floor+2.95))
    si=np.flatnonzero(seam)
    distance,nearest=cKDTree(np_[ni]).query(op[oi[si]],workers=4)
    si=si[distance<.14];nearest=nearest[distance<.14]
    strength=np.clip(1-distances[si]/.25,0,1)
    for name in ('f_dc_0','f_dc_1','f_dc_2'):
        candidate[name][si]=(1-strength)*candidate[name][si]+strength*new[name][ni[nearest]]
    frozen=np.ones(len(oi),bool);frozen[si]=False
    assert candidate[:len(oi)][frozen].tobytes()==old[oi[frozen]].tobytes()
    for name in old.dtype.names:assert np.array_equal(candidate[name][len(oi):],new[name][ni])
    PlyData([PlyElement.describe(candidate,'vertex')],text=False).write(dest/'scene.ply')
    np.savez_compressed(dest/'lineage.npz',old_ids=oi,new_ids=ni,old_weights=np.ones(len(oi)),new_weights=np.ones(len(ni)),old_color_seam_ids=si)
    np.savez_compressed(dest/'takeover_field.npz',field=field,proposed=proposed,semantic_core=semantic_core,coverage_votes=votes,origin=origin,resolution=resolution,z_range=z_range)
    np.save(dest/'replacement_support.npy',np_[ni][::max(1,len(ni)//50000)])
    np.save(dest/'removed_old_centers.npy',op[removed])
    cv2.imwrite(str(dest/'takeover_footprint.png'),np.flipud(field.astype(np.uint8)*255))
    write(dest/'context_scope.json',scope)
    write(dest/'takeover.json',dict(method='single owner per observed semantic region; immutable new Spirula; old color seam only',regions=records,
        proposed_cells=int(proposed.sum()),covered_cells=int(field.sum()),area_m2=float(field.sum()*resolution**2),
        field='takeover_field.npz',resolution_m=resolution,z_range=z_range,minimum_separated_training_views=3,
        region_retirement_is_not_free_space=True,whole_roi_replacement=False,disconnected_regions_not_enclosed_by_global_box=True,
        new_geometry_colors_opacity_unchanged=True,new_gaussian_count=len(ni),old_color_seam_count=len(si),old_color_seam_width_m=.25,
        optimization_iterations=0,local_registration_changed=False,metric_scale_conflict=True,preview_only=True))
    audit=json.loads((dest/'takeover.json').read_text());audit.update(coverage_mode=coverage_mode,coverage_views=visibility_ids,
        photometric_coverage_is_not_metric_depth_validation=coverage_mode=='visual',context_extension_m=1.0 if coverage_mode=='visual' else 0)
    write(dest/'takeover.json',audit)
    write(dest/'fusion.json',dict(base_points=len(old),candidate_points=len(candidate),inserted_points=len(ni),removed_old_points=len(removed),
        frozen_old_points=int(frozen.sum()),old_color_seam_points=len(si),source_sha256=sha256(base),candidate_sha256=sha256(dest/'scene.ply'),
        new_copied_byte_equivalent_for_all_common_fields=True,old_geometry_unchanged=True,whole_roi_replacement=False,
        deletion_policy='semantic regional model retirement, NOT observed free space',outside_support_byte_identical=True))
    write(dest/'bundle.json',dict(state='NEEDS_REVIEW',published=False,quality_accepted=False,scene='scene.ply',person_mask=True,
        navigation_updated=False,reference_database_updated=False,robot_services_modified=False,metric_scale_conflict=True,
        evaluation_scope='actual robot-height capture viewpoints only'))
    assert sha256(base)==registration['base_sha256']
    print('REGIONAL_TAKEOVER',len(ni),len(removed),'seam',len(si),'area',field.sum()*resolution**2,flush=True)


if __name__=='__main__':
    raise SystemExit('Superseded: column takeover changes architectural background. Use boxer_rgbd.py --phase fuse --fusion-mode protected.')
