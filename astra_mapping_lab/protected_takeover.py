"""Complete device volumes with explicit old/new architectural protection.

Unlike column takeover, this never extrudes a footprint through the room or
expands into background for blending. Old and new use the same bounded union
of reviewed device envelopes, with architecture independently protected.
"""
from gpu_config import gpu_index
import argparse,json,os,shutil,time
from pathlib import Path
import numpy as np


def in_volumes(points,regions,margin=.08):
    selected=np.zeros(len(points),bool)
    for r in regions:
        b=np.asarray(r['bounds'])
        selected|=((points>=b[0]-margin)&(points<=b[1]+margin)).all(1)
    return selected


def ensure_fresh_cache(ctx,prior):
    import fcntl,gc,torch
    from astra_scene.postprocess import render_map
    root,source,alignment,out,base,data,registration,rows,ids,held=ctx
    cache=out/'fresh_renders';cache.mkdir(exist_ok=True)
    views=[x['aligned_view'] for x in json.loads((prior/'boxer_protocol.json').read_text())['view_mapping']]
    with (root/'runs/gpu.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        from vps.refinement.gaussian_renderer import GsplatGaussianRenderer
        renderer=GsplatGaussianRenderer('cuda')
        for i in views:
            path=cache/f'{i:03d}.npz'
            if not path.exists():np.savez_compressed(path,**render_map(renderer,alignment/'aligned_masked_sh0.ply',rows[i]))
        del renderer;gc.collect();torch.cuda.empty_cache()
    return cache


def segment_architecture(ctx, prior, cache):
    import fcntl,subprocess,concurrent.futures
    from PIL import Image
    from connected_map_update import write
    root,source,alignment,out,base,data,registration,rows,ids,held=ctx
    views=[x['aligned_view'] for x in json.loads((prior/'boxer_protocol.json').read_text())['view_mapping']]
    selected=[views[j] for j in np.unique(np.linspace(0,len(views)-1,24).round().astype(int))]
    env=os.environ.copy();env.update(SS_LANG='en',SS_NN_LOG='0',VK_ICD_FILENAMES='/usr/share/vulkan/icd.d/nvidia_icd.json',OMP_NUM_THREADS='4')
    gpu=subprocess.check_output(['nvidia-smi','--id',str(gpu_index()),'--query-gpu=uuid','--format=csv,noheader'],text=True).strip().replace('GPU-','').replace('-','')
    def work(task):
        from artifact_cache import sam_mask
        i,side=task;directory=out/'architecture'/f'{side}_{i:03d}'
        rgb=np.uint8(rows[i]['rgb']*255) if side=='current' else np.load(cache/f'{i:03d}.npz')['rgb']
        mask,info=sam_mask(rgb,root,directory,'uuid:'+gpu,'ceiling;window;wall;floor;outdoor background',
            'robot;equipment;metal frame;table;chair;cart;person',True,0,env)
        return f'{side}_{i}',mask,info
    masks={};cache_records=[];tasks=[(i,side) for side in ('current','old') for i in selected]
    with (root/'runs/gpu.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            for n,(key,mask,info) in enumerate(pool.map(work,tasks)):
                cache_records.append(dict(view=key,**info))
                masks[key]=mask;print('ARCHITECTURE',n+1,len(tasks),key,float(mask.mean()),flush=True)
    np.savez_compressed(out/'architecture_masks.npz',**masks)
    write(out/'architecture.json',dict(views=selected,train_only=True,method='independent old/current SAM3 architectural masks',
        protected_classes=['ceiling','window','wall','floor','outdoor background'],segmentation_ground_truth=False,
        cache_hits=sum(x['cache_hit'] for x in cache_records),cache_records=cache_records))


def build(ctx, prior, old_cache, fresh_cache, geometry=False):
    from plyfile import PlyData,PlyElement
    from connected_map_update import write
    from astra_scene.geometry import project
    from astra_scene.io import sha256
    from map_update import roi_weights
    root,source,alignment,out,base,data,registration,rows,ids,held=ctx
    from instance_policy import enabled,policy
    instance_policy=enabled()
    bounded_context=policy() in ('bounded_context_v1','bounded_context_v2')
    bounded_covariance=policy()=='bounded_context_v2'
    if json.loads((prior/'inputs.json').read_text())!=json.loads((out/'inputs.json').read_text()):raise ValueError('prior input mismatch')
    for cache in (old_cache,fresh_cache):
        if json.loads((cache.parent/'inputs.json').read_text())!=json.loads((out/'inputs.json').read_text()):raise ValueError('render cache input mismatch')
    dest=out/'artifacts'
    if (dest/'scene.ply').exists():raise ValueError('candidate already exists; use a new version')
    old=PlyData.read(base,mmap=True)['vertex'].data;new=PlyData.read(alignment/'aligned_masked_sh0.ply',mmap=True)['vertex'].data
    xyz=lambda v:np.column_stack([v[k] for k in ('x','y','z')])
    op,np_=xyz(old),xyz(new)
    scope=json.loads((prior/'artifacts/context_scope.json').read_text());regions=scope['regions']
    proposals={p['id']:p for p in json.loads((prior/'analysis/evidence.json').read_text())['proposals']}
    review={a['id']:a for a in json.loads((prior/'analysis/ai.json').read_text())['items']}
    if json.loads((prior/'analysis/ai.json').read_text()).get('policy')=='automatic_geometry_semantics_v3':
        from semantic_router import guard_regions
        regions=guard_regions(ctx,regions,old_cache,fresh_cache)
        if not regions:
            write(out/'automatic_outcome.json',dict(state='NEEDS_CAPTURE',reason='local reconstruction regressed',published=False))
            raise ValueError('local quality guard retained original map')
        scope=dict(scope,regions=regions)
    masks=np.load(out/'architecture_masks.npz');views=json.loads((out/'architecture.json').read_text())['views']
    floor=json.loads((alignment/'artifacts/floor.json').read_text())['floor_z']
    roi_old=roi_weights(op,data['map_metadata'],data['roi'],.001)>0
    upper=op[roi_old&(op[:,2]>floor+1.8)&(op[:,2]<floor+3.8),2]
    if len(upper)<100:raise ValueError('insufficient architecture to establish ceiling protection')
    hist,edges=np.histogram(upper,bins=np.arange(floor+1.8,floor+3.825,.025))
    peak=int(hist.argmax());ceiling=float((edges[peak]+edges[peak+1])/2)
    completions=[]
    if geometry:
        from geometry_boundary import GeometryBoundary,gaussian_extents,floor_seam_colors
        support=np.load(prior/'assembly_support.npz')
        physical=np.load(prior/'artifacts/supported_depth_points.npy')
        if instance_policy:
            from instance_boundary_review import review as review_boundaries
            regions=review_boundaries(ctx,regions,support,proposals,old_cache)
            scope=dict(scope,regions=regions)
        field=GeometryBoundary([support[r['id']] for r in regions if r['id'] in support],physical,floor)
        if instance_policy:
            from instance_volume import InstanceVolumes
            instances=InstanceVolumes({r['id']:support[r['id']] for r in regions if r['id'] in support})
        write(dest/'completion_review.json',dict(regions=[dict(id=r['id'],semantic_decision=review[r['id']]['decision'],
            connected_voxels=len(support[r['id']]) if r['id'] in support else 0,
            semantic_completeness_verified=False) for r in regions],
            policy='deferred regions remain review candidates; observed connected geometry bounds completion',published=False))
    for region in regions:
        p=proposals[region['id']];a=review[region['id']]
        equipment=p['source_label'] in ('robot','metal frame','equipment cart','cart','trolley','easel')
        if (not geometry or bounded_context) and region.get('evidence_side')=='old' and equipment and a.get('change_type') in ('layout','move','remove'):
            before=np.array(region['bounds']);b=before.copy();original=np.array(p['bounds'])
            # Previous scopes bounded only the points already cleared by depth.
            # They stopped at 1.49 m and could never retire an old upper frame.
            # Complete the nominated equipment's height, while the independent
            # architectural veto below still wins over this volume.
            b[0,:2]=np.minimum(b[0,:2],original[0,:2]);b[1,:2]=np.maximum(b[1,:2],original[1,:2])
            b[1,2]=max(b[1,2],ceiling-.08)
            region['bounds']=b.tolist();completions.append(dict(id=region['id'],before=before.tolist(),after=b.tolist(),
                basis='reviewed old equipment assembly, upper parts previously truncated by clearance-only bounds'))
    selected={};protected={};counts={}
    retirement=np.zeros(len(old),bool);retirement_report={}
    if geometry and instance_policy:
        from fusion_visibility import complete_old_devices,flat_floor
        retirement,retirement_report=complete_old_devices(ctx,op,old,field,regions,prior,fresh_cache,floor,ceiling)
    for side,points,vertices in [('old',op,old),('current',np_,new)]:
        eligible=in_volumes(points,regions,.08)&(roi_weights(points,data['map_metadata'],data['roi'],.001)>0)
        if geometry:
            from scipy.spatial import cKDTree
            permission=roi_weights(points,data['map_metadata'],data['roi'],.001)>0
            floor_mask,_=field.floor_membership(points)
            if instance_policy and side=='current':
                floor_ix=np.flatnonzero(floor_mask&permission)
                floor_mask[floor_ix]=flat_floor(floor_mask[floor_ix],gaussian_extents(vertices[floor_ix])[:,2])
            # Retire/admit the full instance volume, not fragments within a
            # fixed radius of its sampled surface. Different instances remain
            # separate, with the architectural veto applied below.
            if bounded_context:
                # Replay the established reviewed device envelope, but let
                # measured floor support alone authorize ground replacement.
                # Otherwise the box admits unobserved floor and creates seams.
                device=in_volumes(points,regions,.08)&(np.abs(points[:,2]-floor)>=.18)
                if bounded_covariance and side=='current':
                    # Keep the established broad-splat guard independently
                    # of whole-volume ownership. A center inside a box does
                    # not authorize a metre-wide footprint over its background.
                    device_ids=np.flatnonzero(device&permission)
                    extent=gaussian_extents(vertices[device_ids])
                    device[device_ids]&=np.sort(extent,axis=1)[:,1]<.35
            elif instance_policy:device=instances.contains(points)&~floor_mask
            else:
                near=field.tree.query(points,workers=4)[0]<.8
                indexes=np.flatnonzero(near&permission&~floor_mask)
                device=np.zeros(len(points),bool)
                device[indexes]=field.device_membership(points[indexes],gaussian_extents(vertices[indexes]),side)
            eligible=(device|floor_mask)&permission
            if side=='old':eligible|=retirement
        ixall=np.flatnonzero(eligible);q=points[ixall];vote=np.zeros(len(q),np.int16)
        for i in views:
            cache=old_cache if side=='old' else fresh_cache
            render=np.load(cache/f'{i:03d}.npz');ix,u,v,z=project(q,rows[i])
            visible=(render['alpha'][v,u]>.7)&(np.abs(render['depth'][v,u]-z)<.25+.01*z)
            if geometry and instance_policy and side=='current':
                # Expected render depth can lie between translucent surfaces;
                # it must not disable a window/wall semantic veto. Measured
                # foreground occlusion still prevents voting on hidden points.
                visible=(render['alpha'][v,u]>.7)&rows[i]['keep'][v,u]
                visible &= ~rows[i]['valid'][v,u] | (z<=rows[i]['depth'][v,u]+.20)
            vote[ix]+=masks[f'{side}_{i}'][v,u]&visible
        # Local floor under a nominated assembly is allowed by the user. It is
        # still bounded by the device volumes, never by the whole grid ROI.
        floor_context=np.abs(q[:,2]-floor)<.18
        if geometry and instance_policy and side=='current':floor_context=flat_floor(floor_context,gaussian_extents(vertices[ixall])[:,2])
        architectural=(vote>=2)&~floor_context
        if geometry and side=='old':architectural &= ~retirement[ixall]
        architectural|=q[:,2]<floor-.18
        # Protect the dominant old upper horizontal level independently of
        # occasional segmentation misses; thin device rods remain distinguishable.
        axes=np.sort(np.exp(np.column_stack([vertices['scale_'+str(j)][ixall] for j in range(3)])),axis=1)
        architectural|=(q[:,2]>=ceiling-.05)&(axes[:,1]>.025)
        protected[side]=ixall[architectural];selected[side]=ixall[~architectural]
        counts[side]=dict(eligible=len(ixall),protected=int(architectural.sum()),selected=int((~architectural).sum()),minimum_semantic_views=2)
        print('PROTECTED_OWNERSHIP',side,counts[side],flush=True)
    removed=selected['old'];ni=selected['current'];keep=np.ones(len(old),bool);keep[removed]=False;oi=np.flatnonzero(keep)
    candidate=np.empty(len(oi)+len(ni),dtype=old.dtype);candidate[:len(oi)]=old[oi]
    seam=dict(changed=0)
    inserted=new[ni]
    if geometry:inserted,seam=floor_seam_colors(old,op,new,np_,ni,field)
    for k in old.dtype.names:candidate[k][len(oi):]=inserted[k]
    assert candidate[:len(oi)].tobytes()==old[oi].tobytes()
    for k in old.dtype.names:
        if not geometry or not k.startswith('f_dc_'):assert np.array_equal(candidate[k][len(oi):],new[k][ni])
    assert keep[protected['old']].all()
    assert not np.intersect1d(ni,protected['current']).size
    PlyData([PlyElement.describe(candidate,'vertex')],text=False).write(dest/'scene.ply')
    work=out/'context_refinement';work.mkdir(exist_ok=True);os.link(dest/'scene.ply',work/'initial.ply')
    np.savez_compressed(dest/'lineage.npz',old_ids=oi,new_ids=ni,old_weights=np.ones(len(oi)),new_weights=np.ones(len(ni)),
        protected_old_ids=protected['old'],excluded_architecture_new_ids=protected['current'])
    np.save(dest/'replacement_support.npy',np_[ni][::max(1,len(ni)//50000)])
    np.save(dest/'removed_old_centers.npy',op[removed]);write(dest/'context_scope.json',scope)
    write(dest/'takeover.json',dict(method=('connected assembly geometry and independently measured floor' if geometry else 'bounded 3-D device volume takeover with architectural veto'),regions=regions,counts=counts,
        fusion_policy=policy() or 'default',bounded_device_context=bounded_context,
        device_covariance_guard=bounded_covariance,
        geometry_boundary=geometry,floor_seam=seam,
        old_assembly_completions=completions,
        footprint_extrusion=False,background_expansion_m=0,volume_margin_m=.08,architecture_protection=True,
        local_floor_context_allowed=True,local_floor_half_band_m=.18,floor_permission='only inside nominated device volumes',
        old_architecture_byte_identical=True,all_retained_old_byte_identical=True,inserted_new_unchanged=not bool(seam['changed']),
        color_modification=bool(seam['changed']),optimization_iterations=0,whole_roi_replacement=False,region_retirement_is_not_free_space=True,
        protected_ceiling_level_estimate=ceiling,ceiling_estimator='dominant original-map upper-height histogram, not independent calibration',
        semantic_classification_is_not_ground_truth=True,metric_scale_conflict=True,preview_only=True))
    write(dest/'fusion.json',dict(base_points=len(old),candidate_points=len(candidate),inserted_points=len(ni),removed_old_points=len(removed),
        frozen_old_points=len(oi),old_color_seam_points=0,source_sha256=sha256(base),candidate_sha256=sha256(dest/'scene.ply'),
        new_copied_byte_equivalent_for_all_common_fields=not bool(seam['changed']),new_geometry_and_opacity_unchanged=True,new_floor_seam_color_points=seam['changed'],all_retained_old_byte_identical=True,old_architecture_byte_identical=True,
        outside_support_byte_identical=True,whole_roi_replacement=False))
    if geometry and instance_policy:write(dest/'instance_retirement.json',dict(**retirement_report,
        ownership='per_instance_3d_volume',volumes=len(instances.volumes),
        global_hull=False,point_surface_distance_clipping=False))
    write(dest/'bundle.json',dict(state='NEEDS_REVIEW',published=False,quality_accepted=False,scene='scene.ply',person_mask=True,
        navigation_updated=False,reference_database_updated=False,robot_services_modified=False,metric_scale_conflict=True))
    assert sha256(base)==registration['base_sha256']
    print('PROTECTED_TAKEOVER',len(ni),len(removed),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--alignment-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--base-map',type=Path,required=True);p.add_argument('--prior',type=Path,required=True)
    p.add_argument('--old-cache',type=Path,required=True);p.add_argument('--fresh-cache',type=Path,required=True);p.add_argument('--gpu',type=int,default=1)
    a=p.parse_args()
    from connected_map_update import setup,evaluate_stage,geometry_review_stage
    ctx=setup(a)
    if not (a.output/'architecture_masks.npz').exists():segment_architecture(ctx,a.prior,a.old_cache)
    build(ctx,a.prior,a.old_cache,a.fresh_cache)
    evaluate_stage(ctx,include_extra_views=False);geometry_review_stage(ctx)
