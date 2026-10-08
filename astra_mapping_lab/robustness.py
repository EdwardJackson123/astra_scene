"""Experimental training-only gates. Never treats missing depth as free space."""
import numpy as np


POLICY=dict(min_support=2,min_clear=3,max_unknown_fraction=.35,max_contradicted_fraction=.15,
            min_observed_views=3,max_alignment_displacement=.10,min_spatial_second_axis=.25)


def classify_rays(points, rows, view_ids, project, tolerance=.12):
    if len(set(view_ids))!=len(view_ids):raise ValueError('duplicate view IDs')
    counts={k:np.zeros(len(points),np.int16) for k in ('support','clear','occluded')}
    views=0
    for i in view_ids:
        row=rows[i];ix,u,v,z=project(points,row)
        truth=row['depth'][v,u]
        valid=row['valid'][v,u]&row['keep'][v,u]&np.isfinite(truth)&(truth>0)&np.isfinite(z)&(z>0)
        ix,u,v,z,truth=ix[valid],u[valid],v[valid],z[valid],truth[valid]
        if len(ix):views+=1
        tol=tolerance+.01*truth;delta=truth-z
        counts['support'][ix]+=np.abs(delta)<=tol
        counts['clear'][ix]+=delta>tol
        counts['occluded'][ix]+=delta < -tol
    counts['observed_views']=views
    return counts


def decision(old, new, colocated, alignment_ok=True, policy=None):
    policy={**POLICY,**(policy or {})}
    # Removal and insertion are separate permissions; unknown rays cannot erase.
    remove=((old['clear']>=policy['min_clear'])&(old['support']==0))|colocated
    add=(new['support']>=policy['min_support']) & (new['clear']<np.maximum(4,2*new['support']))
    def fraction(mask):return float(np.mean(mask)) if len(mask) else 0.
    unknown_old=fraction(~remove)
    contradicted_new=fraction((new['clear']>=3)&(new['clear']>new['support']))
    unsupported_new=fraction(~add)
    reasons=[]
    if not alignment_ok:reasons.append('unstable_alignment')
    observed=[c['observed_views'] for c in (old,new) if len(c['support'])]
    if not observed or min(observed)<policy['min_observed_views']:reasons.append('insufficient_views')
    if unknown_old>policy['max_unknown_fraction']:reasons.append('old_removal_not_verified')
    if contradicted_new>policy['max_contradicted_fraction']:reasons.append('new_geometry_contradicted')
    if unsupported_new>.50:reasons.append('new_surface_not_supported')
    return dict(accepted=not reasons,reasons=reasons,unknown_removal_fraction=unknown_old,
                contradicted_new_fraction=contradicted_new,unsupported_new_fraction=unsupported_new,
                strict_free_fraction=fraction((old['clear']>=policy['min_clear'])&(old['support']==0)),
                colocated_replacement_fraction=fraction(colocated),
                old_views=old['observed_views'],new_views=new['observed_views'])


def alignment_review(local, mapped, views, registration, stable_upper=2.3, stable_lower=.1):
    from map_update import fit_similarity
    local=np.asarray(local);mapped=np.asarray(mapped);views=np.asarray(views)
    reasons=[];s=registration['scale'];R=np.array(registration['rotation']);t=np.array(registration['translation'])
    transformed=s*local@R.T+t
    error=np.linalg.norm(transformed-mapped,axis=1)
    fit=((mapped[:,2]>stable_upper)|(mapped[:,2]<stable_lower))&(views%4!=3)&(error<.15)
    groups=np.unique(views[fit]);a=local[fit];b=mapped[fit]
    if not registration.get('accepted'):reasons.append('existing_alignment_gate_failed')
    spread=np.std((b-b.mean(0))@np.linalg.svd(b-b.mean(0),full_matrices=False)[2].T,axis=0) if len(b)>=3 else np.zeros(3)
    if spread[1]<POLICY['min_spatial_second_axis']:reasons.append('spatially_concentrated_anchors')
    displacements=[];scales=[]
    if len(groups)<6:reasons.append('too_few_fitting_views')
    else:
        # Entire views are omitted, not individual pixels from the same image.
        probes=local[::max(1,len(local)//1000)];reference=s*probes@R.T+t
        for group in groups:
            ix=fit&(views!=group)
            try:
                si,ri,ti=fit_similarity(local[ix],mapped[ix])
                displacements.append(float(np.percentile(np.linalg.norm(si*probes@ri.T+ti-reference,axis=1),95)))
                scales.append(float(si))
            except (ValueError,np.linalg.LinAlgError):reasons.append('view_omission_fit_failed')
    worst=max(displacements,default=None)
    if worst is not None and worst>POLICY['max_alignment_displacement']:reasons.append('view_omission_unstable')
    return dict(candidate_alignment_accepted=not reasons,reasons=sorted(set(reasons)),fit_views=len(groups),
        spatial_standard_deviation=spread.tolist(),view_omission_p95_displacements=displacements,
        worst_displacement=worst,view_omission_scales=scales,metric_scale_conflict=bool(registration.get('metric_scale_conflict')),
        navigation_deployment_accepted=False,units='original map units; absolute metric scale not verified',
        policy=POLICY,thresholds_status='engineering defaults, not field calibrated')
