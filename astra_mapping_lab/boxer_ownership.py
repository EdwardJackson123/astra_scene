"""Boxer/SAM instance ownership; old removal and new insertion are independent.

Research previews can retire a semantically reviewed old instance even where
its back is now occluded. This is explicitly inferred instance retirement,
never a depth-clearance claim or authorization to clear a navigation cell.
"""
import json
import numpy as np


def segment(ctx, cache):
    import torch,cv2,fcntl
    from segment_anything import sam_model_registry,SamPredictor
    from boxer_regions import corners
    from context_rgbd import nominate
    from connected_map_update import write
    root,source,alignment,out,base,data,registration,rows,ids,held=ctx
    proposals=json.loads((out/'analysis/evidence.json').read_text())['proposals']
    ai=json.loads((out/'analysis/ai.json').read_text())
    chosen=nominate(ai['items'],proposals)
    ps=[p for p in proposals if p['id'] in chosen]
    from instance_policy import enabled
    instance_policy=enabled();prompted={}
    if instance_policy:
        from instance_box_refinement import refine
        prompted=refine(ctx,ps,cache/'old_renders')
    protocol=json.loads((out/'boxer_protocol.json').read_text())
    views=[x['aligned_view'] for x in protocol['view_mapping']]
    masks={};records=[]
    with (root/'runs/gpu.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        model=sam_model_registry['vit_b'](checkpoint='/home/xujie/VPS_dynamic/runtime/game_weights/sam_vit_b_01ec64.pth').cuda().eval()
        predictor=SamPredictor(model)
        for side in ('current','old'):
            for i in views:
                row=rows[i];h,w=row['depth'].shape
                image=np.uint8(np.clip(row['rgb']*255,0,255)) if side=='current' else np.load(cache/'old_renders'/f'{i:03d}.npz')['rgb']
                predictor.set_image(image)
                for p in ps:
                    if p['evidence_side']!=side:continue
                    box=prompted.get((p['id'],i))
                    if instance_policy and box is None:continue
                    q=corners(p['oriented_box']);cam=(q-row['pose'][:3,3])@row['pose'][:3,:3]
                    if (cam[:,2]<.1).any():continue
                    uv=cam@row['K'].T;uv=uv[:,:2]/uv[:,2,None]
                    rect=np.r_[uv.min(0),uv.max(0)];delta=(rect[2:]-rect[:2])*.15
                    rect=np.clip(rect+np.r_[-delta,delta],[0,0,0,0],[w-1,h-1,w-1,h-1])
                    if instance_policy:rect=np.asarray(box)*np.array([w,h,w,h])/1000.
                    if np.prod(np.maximum(0,rect[2:]-rect[:2]))<400:continue
                    with torch.no_grad():mm,score,_=predictor.predict(box=rect,multimask_output=True)
                    m=mm[int(np.argmax(score))]
                    if max(score)<.8 or m.mean()>.65 or m.sum()<80:continue
                    if side=='current':m=m&row['keep']
                    masks[p['id']+'__'+str(i)]=m
                    records.append(dict(id=p['id'],view=i,score=float(max(score)),pixels=int(m.sum())))
                print('BOXER_SAM',side,i,len(masks),flush=True)
        del predictor,model;torch.cuda.empty_cache()
    np.savez_compressed(out/'boxer_sam_masks.npz',**masks)
    write(out/'boxer_sam.json',dict(records=records,selected=chosen,method='same SAM ViT-B box-prompt stage as established Boxer pipeline',person_mask=True,train_only=True))


def ownership(ctx, old, new):
    from instance_policy import enabled
    complete_instances=enabled()
    from connected_map_update import write
    from context_rgbd import nominate
    from connected_regions import components,validate_review
    from astra_scene.geometry import project,evidence,voxel_fuse
    from map_update import roi_weights
    from scipy.spatial import cKDTree
    root,source,alignment,out,base,data,registration,rows,ids,held=ctx
    proposals=json.loads((out/'analysis/evidence.json').read_text())['proposals']
    ai=json.loads((out/'analysis/ai.json').read_text());validate_review(ai['items'],proposals)
    chosen=nominate(ai['items'],proposals);masks=np.load(out/'boxer_sam_masks.npz')
    xyz=lambda v:np.column_stack([v[n] for n in ('x','y','z')])
    op,np_=xyz(old),xyz(new)
    floor=json.loads((alignment/'artifacts/floor.json').read_text())['floor_z']
    weights={'current':np.zeros(len(new)), 'old':np.zeros(len(old))}
    newaxes=np.sort(np.exp(np.column_stack([new['scale_'+str(j)] for j in range(3)])),axis=1)
    oldaxes=np.sort(np.exp(np.column_stack([old['scale_'+str(j)] for j in range(3)])),axis=1)
    reports=[];regions=[];insert_support=[];inferred_removed=[];assemblies={}
    for p in proposals:
        if p['id'] not in chosen:continue
        if max(p['oriented_box']['size'])<.4:
            reports.append(dict(id=p['id'],included=False,reason='small item belongs to larger layout, not a standalone update'))
            continue
        side=p['evidence_side'];pts=np_ if side=='current' else op;axes=newaxes if side=='current' else oldaxes
        bounds=np.array(p['bounds']);search=bounds+np.array([[-.35,-.35,-.05],[.35,.35,.35]])
        if complete_instances:
            # Search is only a computational envelope. Membership below must
            # be supported by independently refined image-instance masks.
            search=bounds+np.array([[-.8,-.8,-.15],[.8,.8,.8]])
        # Large equipment can be partially detected at its base: upper parts
        # must still be linked and segmented in multiple views, not just boxed.
        if p['source_label'] in ('metal frame','equipment cart','trolley','cart','robot','easel'):
            search[1,2]=max(search[1,2],floor+2.8)
        m=(pts>=search[0]).all(1)&(pts<=search[1]).all(1)&(pts[:,2]>floor+.06)
        m&=(axes[:,2]<.25)|((axes[:,1]<.06)&(axes[:,2]<.55))
        indexes=np.flatnonzero(m);q=pts[indexes];votes=np.zeros(len(q),np.int16)
        maskviews=[]
        for key in masks.files:
            if not key.startswith(p['id']+'__'):continue
            i=int(key.split('__')[1]);maskviews.append(i);ix,u,v,z=project(q,rows[i])
            if side=='current':
                # Do not require three depth hits on a thin rod. Image-instance
                # agreement nominates it; contradictory rays below veto it.
                visible=rows[i]['keep'][v,u] & (~rows[i]['valid'][v,u] | (rows[i]['depth'][v,u]>=z-.18))
            else:
                r=np.load(out/'old_renders'/f'{i:03d}.npz')
                visible=(r['alpha'][v,u]>.7)&(np.abs(r['depth'][v,u]-z)<.20)
            votes[ix]+=masks[key][v,u]&visible
        seeded=votes>=2
        if seeded.sum()<80:
            reports.append(dict(id=p['id'],included=False,reason='insufficient multiview instance support',points=int(seeded.sum())));continue
        # Only spatially connected geometry near repeatedly segmented surfaces
        # completes the instance. No erasure of the full enclosing box.
        # Previous code connected only already-segmented points, so a partial
        # 2-D detection could never recover the upper beam. Grow over the full
        # bounded geometry graph, with repeatedly segmented points as seeds.
        # Large near-ceiling sheets are excluded before graph construction.
        architecture=(q[:,2]>2.55)&(axes[indexes,1]>.025)
        graphpoints=q[seeded] if complete_instances else q[~architecture | seeded]
        surf,_=voxel_fuse(graphpoints,np.zeros_like(graphpoints),.035)
        groups=[np.arange(len(surf))] if complete_instances else components(surf,.12,12)
        if not groups:continue
        seed_distance=cKDTree(q[seeded]).query(surf,workers=4)[0]
        linked=[g for g in groups if (seed_distance[g]<.045).sum()>=max(12,int(len(g)*.01))]
        if not linked:continue
        supported=surf[np.concatenate(linked)]
        near=cKDTree(supported).query(q,workers=4)[0]<.055
        selected=indexes[near];ss,ff,_=evidence(pts[selected],rows,ids,.18)
        if side=='current':
            keep=~((ff>=4)&(ff>2*np.maximum(ss,1)))
            selected=selected[keep]
            weights[side][selected]=1
            insert_support.append(pts[selected])
            strict=0;inferred=0
        else:
            strict_mask=(ff>=4)&(ss==0)
            clear_fraction=float(strict_mask.mean());stable_fraction=float((ss>=3).mean())
            # Semantic old-instance retirement is separate from verified free
            # space. This is allowed only in the versioned research candidate.
            retire_instance=clear_fraction>=.15 and stable_fraction<.45
            keep=strict_mask | (retire_instance & (ss<3))
            inferred=int((keep&~strict_mask).sum());strict=int(strict_mask.sum())
            if inferred:inferred_removed.append(pts[selected[keep&~strict_mask]])
            selected=selected[keep];weights[side][selected]=1
        if not len(selected):continue
        # Keep the full connected assembly before clearance filtering. Its upper
        # parts must not be lost just because the current sensor cannot see them.
        assemblies[p['id']]=supported.astype('float32')
        selectedpts=pts[selected]
        b=np.array([selectedpts.min(0)-.03,selectedpts.max(0)+.03])
        regions.append(dict(id=p['id'],bounds=b.tolist(),mode='replace',evidence_side=side))
        reports.append(dict(id=p['id'],included=True,evidence_side=side,selected_points=len(selected),mask_views=len(maskviews),
            contradicted_fraction=float((~keep).mean()) if side=='current' else 0.,
            strict_free_points=strict,inferred_instance_retirement_points=inferred,scope_bounds=b.tolist(),
            search_boundary_reached=bool((supported.max(0)>=search[1]-.04).any() or (supported.min(0)[:2]<=search[0,:2]+.04).any()),
            complete_geometry_graph=True,seeded_gaussians=int(seeded.sum()),connected_voxels=len(supported),
            clear_fraction=clear_fraction if side=='old' else None,stable_fraction=stable_fraction if side=='old' else None))
        print('BOXER_OWNERSHIP',p['id'],len(selected),'inferred',inferred,flush=True)
    if ai.get('policy')=='automatic_geometry_semantics_v3':
        from semantic_router import completed_regions
        accepted=completed_regions(ctx,proposals,ai['items'],assemblies,reports)
        assemblies={k:v for k,v in assemblies.items() if k in accepted}
        regions=[r for r in regions if r['id'] in accepted]
        for r in reports:
            if r.get('included') and r['id'] not in accepted:r.update(included=False,reason='automatic completion check retained old region')
        if not assemblies:
            write(out/'automatic_outcome.json',dict(state='NEEDS_CAPTURE',reason='no complete supported assembly',published=False))
            raise ValueError('自动完整性检查未通过：已保留原地图，请补采遮挡或缺失区域')
        tree=cKDTree(np.concatenate(list(assemblies.values())))
        for side,points in [('old',op),('current',np_)]:
            active=np.flatnonzero(weights[side]>0);weights[side][active[tree.query(points[active],workers=4)[0]>.10]]=0
        insert_support=[np_[weights['current']>0]]
    # Background exposed by removing an old assembly is admitted only near
    # repeatedly measured current surfaces, not by filling its old volume.
    current=np.concatenate(insert_support) if insert_support else np.empty((0,3))
    if not len(current):raise ValueError('no supported Boxer new instance')
    oldids=np.flatnonzero(weights['old']>0)
    from astra_scene.context_fusion import field
    scope=dict(regions=regions,mode='replace',permission_roi=data['roi'],roi_role='permission_only')
    searchids=np.flatnonzero(field(np_,scope,.15)>0)
    physical=np.load(out/'artifacts/supported_depth_points.npy')
    dist=cKDTree(physical).query(np_[searchids],workers=4)[0]
    surfaceids=searchids[dist<.10]
    ss,ff,_=evidence(np_[surfaceids],rows,ids,.12)
    surfaceids=surfaceids[(ss>=3)&(ff<=1)]
    # Only newly revealed background behind a retired old instance qualifies.
    bgvotes=np.zeros(len(surfaceids),np.int16)
    for i in ids:
        ix,u,v,z=project(np_[surfaceids],rows[i]);r=np.load(out/'old_renders'/f'{i:03d}.npz')
        bgvotes[ix]+=rows[i]['valid'][v,u]&(np.abs(rows[i]['depth'][v,u]-z)<.12)&(r['alpha'][v,u]>.8)&(r['depth'][v,u]<z-.22)
    background=surfaceids[bgvotes>=3];weights['current'][background]=1
    # Avoid co-located old/new surfaces. A supported new surface replaces old
    # surfels locally, independently of old object labels.
    replacement=np.flatnonzero(weights['current']>0)
    d=cKDTree(np_[replacement]).query(op,workers=4)[0]
    co=np.flatnonzero(d<.10);ss,ff,_=evidence(op[co],rows,ids,.12)
    weights['old'][co[ss>=3]]=1
    permission_old=roi_weights(op,data['map_metadata'],data['roi'],.15)
    permission_new=roi_weights(np_,data['map_metadata'],data['roi'],.15)
    weights['old']*=permission_old;weights['current']*=permission_new
    np.savez_compressed(out/'boxer_weights.npz',old=weights['old'],new=weights['current'])
    np.savez_compressed(out/'assembly_support.npz',**assemblies)
    np.save(out/'artifacts/inferred_retired_old_centers.npy',np.concatenate(inferred_removed) if inferred_removed else np.empty((0,3)))
    write(out/'artifacts/boxer_ownership.json',dict(regions=reports,background_points=len(background),
        old_points=int((weights['old']>0).sum()),new_points=int((weights['current']>0).sum()),
        inferred_instance_retirement=True,preview_only=True,published=False,navigation_clearance_authorized=False,
        replacement_is_box_volume=False,method='Boxer -> independent SAM instances -> connected geometry -> separate old/new ownership'))
    write(out/'artifacts/context_scope.json',scope)
    return weights['old'],weights['current'],scope,reports
