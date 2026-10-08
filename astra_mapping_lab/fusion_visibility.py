"""Separate thin-device retirement from planar background preservation."""
import numpy as np


def retirement_mask(distance, middle_axis, instance_votes, covered_views, permitted):
    # Completes an already nominated instance, not a free-space assertion.
    return (permitted & (distance < .45) & (middle_axis < .06)
            & (instance_votes >= 3) & (covered_views >= 3))


def flat_floor(center_member, vertical_extent):
    # A Gaussian centered at floor height may still cover an entire window.
    return center_member & (vertical_extent < .10)


def replacement_visible(valid, model_depth, point_depth):
    return valid & (model_depth >= point_depth - .20)


def complete_old_devices(ctx, points, vertices, field, regions, prior, fresh_cache, floor, ceiling):
    from astra_scene.geometry import project
    from map_update import roi_weights
    root,source,alignment,out,base,data,registration,rows,ids,held=ctx
    masks=np.load(prior/'boxer_sam_masks.npz')
    accepted={r['id'] for r in regions if r.get('evidence_side')=='old'}
    axes=np.sort(np.exp(np.column_stack([vertices['scale_'+str(k)] for k in range(3)])),axis=1)
    distance=field.tree.query(points,workers=4)[0]
    permission=(roi_weights(points,data['map_metadata'],data['roi'],.001)>0)
    permission &= (points[:,2]>floor+.12)&(points[:,2]<ceiling-.08)
    ix=np.flatnonzero(permission&(distance<.45)&(axes[:,1]<.06))
    q=points[ix];votes=np.zeros(len(q),np.int16);covered=votes.copy()
    byview={}
    for key in masks.files:
        name,view=key.rsplit('__',1)
        if name in accepted:
            i=int(view)
            if i in held:raise ValueError('held-out mask used for retirement')
            byview[i]=masks[key].copy() if i not in byview else byview[i]|masks[key]
    for i,mask in byview.items():
        r=np.load(fresh_cache/f'{i:03d}.npz');j,u,v,z=project(q,rows[i])
        valid=rows[i]['keep'][v,u]&(r['alpha'][v,u]>.95)
        # Count each camera once even when several proposals overlap.
        visible=replacement_visible(valid,r['depth'][v,u],z)
        votes[j]+=visible&mask[v,u]
        covered[j]+=visible
    accept=retirement_mask(distance[ix],axes[ix,1],votes,covered,np.ones(len(ix),bool))
    result=np.zeros(len(points),bool);result[ix[accept]]=True
    return result,dict(version='instance_surface_v1',candidates=len(ix),selected=int(accept.sum()),
        minimum_instance_views=3,maximum_completion_distance_m=.45,
        inferred_instance_retirement=True,authorizes_navigation_clearance=False,
        held_out_used=False,accepted_old_instances=sorted(accepted))
