"""RGB-D adapter for the established context-fusion algorithm.

Same smooth regional field, 320-step optimizer, bounded new geometry and weak
depth loss as astra_scene.context_fusion. Adds person masks, permission clipping
and visibility-gated retention of unknown old geometry. Preview != publication.
"""
import json, time, gc
from pathlib import Path
import numpy as np


def nominate(items, proposals):
    byid={p['id']:p for p in proposals};result=[]
    for item in items:
        p=byid[item['id']]
        visible_layout=item.get('change_type') in ('layout','add','move','remove')
        enough_geometry=p['novelty_points']>=20 or p['clearance_points']>=100
        if (item['decision']=='update' or (item['decision']=='defer' and visible_layout)) and not item.get('ai_validation_failed') and len(set(item.get('supporting_views',[])))>=2 and enough_geometry:
            result.append(item['id'])
    return result


def run(ctx, iterations=320, ownership_policy="sensor"):
    import fcntl
    from connected_map_update import write
    from astra_scene.context_fusion import field,context_scope
    from astra_scene.geometry import evidence,voxel_fuse,project
    from astra_scene.quality import target_mask
    from astra_scene.io import sha256
    from map_update import roi_weights,weighted_opacity
    from connected_regions import components,validate_review,confirmed_stale
    from plyfile import PlyData,PlyElement
    from scipy.spatial import cKDTree
    import torch
    from gsplat.rendering import rasterization
    root,source,alignment,out,base,data,registration,rows,ids,held=ctx
    dest=out/'artifacts';work=out/'context_refinement';work.mkdir(exist_ok=False)
    proposals=json.loads((out/'analysis/evidence.json').read_text())['proposals']
    ai=json.loads((out/'analysis/ai.json').read_text())
    if ai.get('complete') is not True:raise ValueError('incomplete semantic review')
    validate_review(ai['items'],proposals);chosen=nominate(ai['items'],proposals)
    if not chosen:raise ValueError('no semantically supported layout candidate')
    raw_support=np.load(out/'component_support.npz')
    old=PlyData.read(base,mmap=True)['vertex'].data;new=PlyData.read(alignment/'aligned_masked_sh0.ply',mmap=True)['vertex'].data
    xyz=lambda v:np.column_stack([v[n] for n in ('x','y','z')])
    op=xyz(old);np_=xyz(new)
    floor=json.loads((alignment/'artifacts/floor.json').read_text())['floor_z']
    axes=np.sort(np.exp(np.column_stack([new['scale_'+str(j)] for j in range(3)])),axis=1)
    # A long, thin rod is not a large floating blob. The previous max-axis-only
    # filter broke gantries into lower and upper fragments.
    thin=(axes[:,1]<.06)&(axes[:,2]<.50)
    good=(new['opacity']>-2.2)&((axes[:,2]<.15)|thin)
    if ownership_policy == 'boxer':
        from boxer_ownership import ownership
        ow,nw,scope,completion=ownership(ctx,old,new)
        regions=scope['regions'];chosen=[r['id'] for r in regions]
        band=.30;expanded=context_scope(scope,band)
        def weight(p):return field(p,scope,band)*roi_weights(p,data['map_metadata'],data['roi'],band)
        stale=np.zeros(0,dtype=bool)
        write(dest/'completion.json',dict(regions=completion,policy='Boxer -> SAM -> connected instances -> separate old/new ownership',deployable=False))
    elif ownership_policy == 'sensor':
        regions=[];completion=[]
        for p in proposals:
            if p['id'] not in chosen:continue
            bounds=np.array(p['bounds']);search=bounds+np.array([[-.45,-.45,0],[.45,.45,0]])
            search[1,2]=max(bounds[1,2],floor+2.8)
            q=np_[good & (np_>=search[0]).all(1)&(np_<=search[1]).all(1)&(np_[:,2]>floor+.10)]
            q,_=voxel_fuse(q,np.zeros_like(q),.03)
            groups=components(q,.14,10);seed_dist=cKDTree(raw_support[p['id']]).query(q)[0]
            linked=[g for g in groups if (seed_dist[g]<.12).sum()>=10]
            grown=bounds.copy();count=0
            if linked:
                connected=q[np.concatenate(linked)];count=len(connected)
                grown=np.array([np.minimum(bounds[0],connected.min(0)-.025),np.maximum(bounds[1],connected.max(0)+.025)])
            # Each assembly has its own bounded extent. Never take a room-wide hull.
            if (grown[1]-grown[0]).max()>4 or np.prod(grown[1]-grown[0])>24:
                grown=bounds.copy();reason='completion growth capped; original assembly retained for review'
            else:reason='bounded connected Gaussian components recover structures missing sensor depth'
            region=dict(id=p['id'],bounds=grown.tolist(),mode='replace',source='semantic layout nomination plus connected 3-D completion')
            regions.append(region);completion.append(dict(id=p['id'],original_bounds=bounds.tolist(),bounds=grown.tolist(),connected_points=count,reason=reason,
                search_bounds=search.tolist(),thin_rod_aware=True,connectivity_m=.14,search_boundary_reached=bool(linked and ((connected.max(0)>=search[1]-.05).any() or (connected.min(0)[:2]<=search[0,:2]+.05).any())),
                ai_decision=next(a['decision'] for a in ai['items'] if a['id']==p['id']),preview_only=True,coverage_verified=False))
        scope=dict(regions=regions,mode='replace',permission_roi=data['roi'],roi_role='permission_only')
        write(dest/'context_scope.json',scope);write(dest/'completion.json',dict(regions=completion,policy='nominate -> complete -> preview -> validate',deployable=False))
        band=.30;expanded=context_scope(scope,band)
        def weight(p):return field(p,scope,band)*roi_weights(p,data['map_metadata'],data['roi'],band)
        ow=weight(op);nw=weight(np_)
        # Only insert current structures or a background newly revealed behind an
        # old structure. A bounding volume cannot authorize a second ceiling/floor.
        masks=np.load(out/'semantic_masks.npz');candidate_ids=np.flatnonzero(nw>1e-4)
        current_votes=np.zeros(len(candidate_ids),np.int16);background_votes=current_votes.copy();positions=np_[candidate_ids]
        for key in masks.files:
            if not key.startswith('current_'):continue
            i=int(key.split('_')[1]);ix,u,v,z=project(positions,rows[i]);row=rows[i]
            visible=row['keep'][v,u]&row['valid'][v,u]&(row['depth'][v,u]>=z-.15)
            current_votes[ix]+=visible&masks[key][v,u]
            old_render=np.load(out/'old_renders'/f'{i:03d}.npz')
            background_votes[ix]+=visible&(np.abs(row['depth'][v,u]-z)<.15)&masks['old_'+str(i)][v,u]&(old_render['alpha'][v,u]>.8)&(old_render['depth'][v,u]<z-.22)
        physical=np.load(dest/'supported_depth_points.npy');distance=cKDTree(physical).query(positions,workers=4)[0]
        slender=(axes[candidate_ids,1]<.025)&(axes[candidate_ids,2]<.35)&(axes[candidate_ids,2]>3*np.maximum(axes[candidate_ids,1],1e-6))
        supported=(distance<.18)&((current_votes>=3)|(background_votes>=3))
        supported|=slender&(current_votes>=3)
        nw[candidate_ids[~supported]]=0
        write(work/'new_ownership.json',dict(proposed=len(candidate_ids),supported=int(supported.sum()),rejected=int((~supported).sum()),
            current_structure_points=int((current_votes>=3).sum()),revealed_background_points=int((background_votes>=3).sum()),
            supported_thin_structure_points=int((slender&(current_votes>=3)).sum()),minimum_semantic_views=3,maximum_sensor_surface_distance_m=.18,
            stable_architectural_surfaces_not_replaced_by_box_membership=True))
        visible_ids=np.flatnonzero(ow>0)
        old_structure_votes=np.zeros(len(visible_ids),np.int16)
        for key in masks.files:
            if not key.startswith('old_'):continue
            i=int(key.split('_')[1]);ix,u,v,z=project(op[visible_ids],rows[i]);old_render=np.load(out/'old_renders'/f'{i:03d}.npz')
            old_structure_votes[ix]+=masks[key][v,u]&(old_render['alpha'][v,u]>.8)&(np.abs(old_render['depth'][v,u]-z)<.18)
        ss,ff,_=evidence(op[visible_ids],rows,ids,.12)
        # Current depth occlusion or missing data cannot authorize old geometry edits.
        replacement_ids=np.flatnonzero(nw>1e-4)
        near_current=cKDTree(np_[replacement_ids]).query(op[visible_ids],workers=4)[0]<.18 if len(replacement_ids) else np.zeros(len(visible_ids),bool)
        observed=((ss>=3)&near_current)|((ff>=4)&(ss==0)&(old_structure_votes>=3));ow[visible_ids[~observed]]=0
        # RGB-D clearance uses a wider tolerance than blend ownership to avoid
        # treating alignment noise as absence. Confirmed obsolete points are removed
        # even in the seam; opacity blending them back would create a ghost object.
        clear_s,clear_f,_=evidence(op[visible_ids],rows,ids,.22)
        stale=confirmed_stale(clear_s,clear_f)&(old_structure_votes>=3);ow[visible_ids[stale]]=1.
    else:
        raise ValueError('unknown ownership policy')
    old_ids=np.flatnonzero(ow<1-1e-6);new_ids=np.flatnonzero(nw>1e-4)
    initial=np.empty(len(old_ids)+len(new_ids),dtype=old.dtype);count=len(old_ids);initial[:count]=old[old_ids]
    for name in old.dtype.names:initial[name][count:]=new[name][new_ids]
    selected_slender=(axes[new_ids,1]<.025)&(axes[new_ids,2]<.35)&(axes[new_ids,2]>3*np.maximum(axes[new_ids,1],1e-6))
    if ownership_policy == 'sensor':
        for j in range(3):initial['scale_'+str(j)][count:]=np.minimum(initial['scale_'+str(j)][count:],np.log(np.where(selected_slender,.35,.12)))
    weights=np.r_[1-ow[old_ids],nw[new_ids]];active=np.r_[ow[old_ids]>0,np.ones(len(new_ids),bool)]
    initial['opacity'][active]=weighted_opacity(initial['opacity'][active],weights[active])
    active_ids=np.flatnonzero(active);frozen_ids=np.flatnonzero(~active)
    PlyData([PlyElement.describe(initial,'vertex')],text=False).write(work/'initial.ply')
    write(work/'inputs.json',dict(base_sha256=sha256(base),fresh_sha256=sha256(alignment/'aligned_masked_sh0.ply'),
        rows_sha256=sha256(alignment/'aligned_frames.json'),scope=scope,iterations=iterations,band_m=band,width=640,depth_weight=.03,
        upstream_code_sha256=sha256(root/'astra_scene/context_fusion.py'),adapter_code_sha256=sha256(Path(__file__)),preview_only=True,
        confirmed_stale_not_blended=int(stale.sum()),stale_minimum_free_views=6,stale_surface_votes=0,stale_tolerance_m=.22))
    torch.manual_seed(12);np.random.seed(12);torch.set_num_threads(4)
    with (root/'runs/gpu.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        tensor=lambda a:torch.as_tensor(np.asarray(a).copy(),dtype=torch.float32,device='cuda')
        arrays=dict(means=xyz(initial),colors=np.column_stack([initial['f_dc_'+str(j)] for j in range(3)])*.28209479177387814+.5,
                    opacity=initial['opacity'][:,None],scale=np.column_stack([initial['scale_'+str(j)] for j in range(3)]))
        fixed={k:tensor(v[frozen_ids]) for k,v in arrays.items()};params={k:torch.nn.Parameter(tensor(v[active_ids])) for k,v in arrays.items()}
        anchors={k:v.detach().clone() for k,v in params.items()}
        quats=tensor(np.column_stack([initial['rot_'+str(j)] for j in range(4)])[np.r_[frozen_ids,active_ids]])
        quats/=torch.linalg.vector_norm(quats,dim=1,keepdim=True).clamp_min(1e-8)
        fresh_mask=tensor((active_ids>=count)[:,None]);optimizer=torch.optim.Adam([dict(params=[params[k]],lr=lr) for k,lr in [('means',.00015),('colors',.003),('opacity',.008),('scale',.001)]])
        validation=sorted(held,key=lambda i:rows[i]['frame'])[::2];test=sorted(held,key=lambda i:rows[i]['frame'])[1::2]
        train=[i for i,f in enumerate(data['frames']) if f['reconstruction_split']=='train'];metadata={}
        for i in train+validation:
            row=rows[i];mask=target_mask(row,expanded)&row['keep'];mask &= row['valid']
            metadata[i]=dict(K=tensor(row['K'])[None],view=tensor(np.linalg.inv(row['pose']))[None],truth=tensor(row['rgb']),depth=tensor(row['depth']),
                mask=torch.as_tensor(mask,device='cuda'),keep=torch.as_tensor(row['keep'],device='cuda'))
        def render(i):
            m=metadata[i];p={k:torch.cat([fixed[k],v],0) for k,v in params.items()}
            rgb,alpha,_=rasterization(p['means'],quats,p['scale'].exp(),p['opacity'].sigmoid().flatten(),p['colors'].clamp(0,1),m['view'],m['K'],640,480,packed=True,render_mode='RGB+ED')
            return rgb[0,:,:,:3],rgb[0,:,:,3],alpha[0,:,:,0]
        def score():
            values=[]
            with torch.no_grad():
                for i in validation:
                    pred,dep,alpha=render(i);m=metadata[i];mask=m['mask']
                    if mask.sum()<100:continue
                    rgb=(pred-m['truth']).abs()[mask].mean();de=torch.relu((dep-m['depth']).abs()-(.08+.01*m['depth']))[mask].clamp(max=1).mean()
                    values.append(float(rgb+.03*de))
            if not values:raise ValueError('no validation coverage for context candidate')
            return float(np.mean(values))
        snapshot=lambda:{k:v.detach().cpu().clone() for k,v in params.items()}
        bestscore=score();best=snapshot();selected=0;history=[dict(step=0,validation_score=bestscore)];started=time.time()
        print('CONTEXT_INIT',len(initial),'editable',len(active_ids),'regions',chosen,'validation',bestscore,flush=True)
        for step in range(1,iterations+1):
            if (step-1)%len(train)==0:order=np.random.permutation(train)
            i=int(order[(step-1)%len(train)]);m=metadata[i];pred,dep,alpha=render(i);mask=m['mask'];outside=m['keep']&~mask
            error=(pred-m['truth']).abs();loss=error[mask].mean() if mask.any() else error[m['keep']].mean()
            if outside.any():loss=loss+.5*error[outside].mean()
            if mask.any():loss=loss+.03*torch.relu((dep-m['depth']).abs()-(.08+.01*m['depth']))[mask].clamp(max=1).mean()+.005*torch.relu(.5-alpha[mask]).mean()
            loss=loss+.001*(params['opacity']-anchors['opacity']).square().mean()+.01*(params['scale']-anchors['scale']).square().mean()
            optimizer.zero_grad(set_to_none=True);loss.backward();params['means'].grad.mul_(fresh_mask);params['scale'].grad.mul_(fresh_mask)
            if step<40:params['means'].grad.zero_();params['scale'].grad.zero_()
            optimizer.step()
            with torch.no_grad():
                for key,bound in [('means',.03),('scale',.25)]:params[key].copy_(torch.maximum(torch.minimum(params[key],anchors[key]+bound*fresh_mask),anchors[key]-bound*fresh_mask))
                params['colors'].clamp_(0,1);params['opacity'].clamp_(-16,16)
            if step%40==0 or step==iterations:
                value=score();history.append(dict(step=step,validation_score=value))
                if value<bestscore:bestscore=value;best=snapshot();selected=step
                write(work/'progress.json',dict(step=step,total=iterations,best_step=selected,best_score=bestscore,elapsed_s=time.time()-started));print('CONTEXT_STEP',step,value,flush=True)
        final=initial.copy()
        for j,key in enumerate(('x','y','z')):final[key][active_ids]=best['means'].numpy()[:,j]
        for j in range(3):final['f_dc_'+str(j)][active_ids]=(best['colors'].numpy()[:,j]-.5)/.28209479177387814;final['scale_'+str(j)][active_ids]=best['scale'].numpy()[:,j]
        final['opacity'][active_ids]=best['opacity'].numpy().ravel()
        escaped=weight(xyz(final[active_ids]))<=0
        for key in ('x','y','z'):final[key][active_ids[escaped]]=initial[key][active_ids[escaped]]
        assert final[frozen_ids].tobytes()==old[old_ids[frozen_ids]].tobytes()
        assert all(np.array_equal(final[key][:count],old[key][old_ids]) for key in ('x','y','z','scale_0','scale_1','scale_2'))
        PlyData([PlyElement.describe(final,'vertex')],text=False).write(dest/'scene.ply')
        np.savez_compressed(dest/'lineage.npz',old_ids=old_ids,new_ids=new_ids,old_weights=1-ow[old_ids],new_weights=nw[new_ids])
        np.save(dest/'removed_old_centers.npy',op[ow>=1-1e-6])
        write(dest/'context_fusion.json',dict(method='established context-fusion optimizer adapted to masked sensor RGB-D',regions=chosen,iterations=iterations,
            selected_step=selected,validation_score=bestscore,history=history,band_m=band,depth_weight=.03,maximum_new_displacement_per_axis_m=.03,
            source_geometry_unchanged=True,unobserved_old_geometry_retained=ownership_policy=='sensor',ownership_policy=ownership_policy,person_masks_used_for_all_rgb_and_depth_losses=True,
            validation=[rows[i]['frame'] for i in validation],test=[rows[i]['frame'] for i in test],test_used_for_selection=False,
            deployable=False,scale_conflict=True))
        write(dest/'fusion.json',dict(candidate_points=len(final),base_points=len(old),inserted_points=len(new_ids),removed_old_points=int((ow>=1-1e-6).sum()),
            transition_old_points=int((active_ids<count).sum()),frozen_old_points=len(frozen_ids),outside_support_byte_identical=True,source_sha256=sha256(base),
            candidate_sha256=sha256(dest/'scene.ply'),candidate_regions=chosen,whole_roi_replacement=False,scope='context_scope.json',confirmed_stale_not_blended=int(stale.sum())))
    assert sha256(base)==registration['base_sha256']
    print('CONTEXT_DONE',dest/'scene.ply',flush=True)
