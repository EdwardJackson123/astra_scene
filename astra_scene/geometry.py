"""Visibility evidence: missing depth and foreground occlusion never prove removal."""
import cv2
import numpy as np
from scipy.spatial import cKDTree
from .io import independent

def project(xyz, row):
    camera=(xyz-row["pose"][:3,3])@row["pose"][:3,:3]
    z=camera[:,2]
    pix=camera@row["K"].T
    uv=np.rint(pix[:,:2]/np.maximum(z[:,None],1e-8)).astype(np.int64)
    h,w=row["depth"].shape
    ok=(z>0)&np.isfinite(camera).all(axis=1)&(uv[:,0]>=0)&(uv[:,0]<w)&(uv[:,1]>=0)&(uv[:,1]<h)
    ix=np.flatnonzero(ok)
    return ix,uv[ix,0],uv[ix,1],z[ix]

def backproject(row, mask=None, stride=1):
    mask=row["valid"].copy() if mask is None else (mask&row["valid"])
    if stride>1:
        sampling=np.zeros_like(mask);sampling[::stride,::stride]=True;mask &= sampling
    v,u=np.where(mask);z=row["depth"][v,u]
    p=np.column_stack([u,v,np.ones(len(u))])@np.linalg.inv(row["K"]).T*z[:,None]
    return p@row["pose"][:3,:3].T+row["pose"][:3,3], row["rgb"][v,u]

def evidence(xyz, rows, ids, tolerance=.06, chunk=100000):
    surface=np.zeros(len(xyz),np.int16); free=surface.copy(); occluded=surface.copy()
    for start in range(0,len(xyz),chunk):
        stop=min(start+chunk,len(xyz));p=xyz[start:stop]
        for i in ids:
            row=rows[i];ix,u,v,z=project(p,row)
            valid=row["valid"][v,u];clear=row["clear"][v,u]
            delta=row["depth"][v,u]-z
            surface[start+ix] += valid & (np.abs(delta)<=tolerance)
            free[start+ix] += clear & (delta>tolerance)
            occluded[start+ix] += valid & (delta < -tolerance)
    return surface,free,occluded

def voxel_fuse(xyz,rgb,size):
    keys=np.floor(xyz/size).astype(np.int64)
    _,inverse=np.unique(keys,axis=0,return_inverse=True)
    count=np.bincount(inverse)
    p=np.column_stack([np.bincount(inverse,weights=xyz[:,j])/count for j in range(3)])
    c=np.column_stack([np.bincount(inverse,weights=rgb[:,j])/count for j in range(3)])
    return p,c

def geometry_audit(rows, ids, cfg):
    pairs=[]
    for a,b in zip(ids[:-1],ids[1:]):
        xyz,_=backproject(rows[a],stride=6)
        ix,u,v,z=project(xyz,rows[b])
        ok=rows[b]["valid"][v,u]
        delta=np.abs(rows[b]["depth"][v,u]-z)
        residual=delta[ok]
        if len(residual)<100:
            pairs.append(dict(a=a,b=b,matches=len(residual),passed=False,reason="too_little_overlap"));continue
        median=float(np.median(residual));p75=float(np.percentile(residual,75))
        pairs.append(dict(a=a,b=b,matches=len(residual),median_depth_error_m=median,p75_depth_error_m=p75,
                          passed=median<=cfg["max_pair_median_m"] and p75<=cfg["max_pair_p75_m"]))
    good=sum(p["passed"] for p in pairs)
    positions=independent(rows,ids,cfg["view_baseline_m"])
    return dict(passed=len(positions)>=cfg["min_support_views"] and len(pairs)>=3 and good/len(pairs)>=cfg["min_pair_pass_fraction"],
                pairs=pairs,independent_views=len(positions),
                limitation="depth_consistency_only_not_absolute_pose_ground_truth")

def discover(rows, ids, base_depths, cfg):
    positions=independent(rows,ids,cfg["view_baseline_m"])
    pts=[];colors=[]
    for i in positions:
        row=rows[i];base=base_depths[i]
        # Low old alpha can indicate unmodelled geometry; a confirmed measured
        # surface still needs independent cross-view support.
        changed=row["valid"] & ((base["alpha"]<.5)|(np.abs(base["depth"]-row["depth"])>cfg["change_tolerance_m"]))
        p,c=backproject(row,changed,stride=cfg["pixel_stride"])
        pts.append(p);colors.append(c)
    if not pts or not sum(map(len,pts)): return np.empty((0,3)),np.empty((0,3)),dict(proposed=0,accepted=0)
    xyz,rgb=voxel_fuse(np.concatenate(pts),np.concatenate(colors),cfg["voxel_m"])
    support,free,_=evidence(xyz,rows,positions,cfg["surface_tolerance_m"])
    # Require novelty in several base renders as well as measured support.
    novelty=np.zeros(len(xyz),np.int16)
    for i in positions:
        ix,u,v,z=project(xyz,rows[i]);base=base_depths[i]
        valid=rows[i]["valid"][v,u] & (np.abs(rows[i]["depth"][v,u]-z)<=cfg["surface_tolerance_m"])
        novelty[ix] += valid & ((base["alpha"][v,u]<.5)|(np.abs(base["depth"][v,u]-z)>cfg["change_tolerance_m"]))
    keep=(support>=cfg["min_support_views"])&(free<=cfg["max_free_contradictions"])&(novelty>=cfg["min_support_views"])
    if keep.sum()>cfg["max_new_gaussians"]: raise ValueError("change_region_too_large_split_capture")
    return xyz[keep],rgb[keep],dict(proposed=len(xyz),accepted=int(keep.sum()),independent_views=len(positions))

def grid_cells(xyz, meta, shape):
    yaw=float(meta["origin"][2]);c,s=np.cos(yaw),np.sin(yaw)
    local=(xyz[:,:2]-np.asarray(meta["origin"][:2]))@np.array([[c,-s],[s,c]])
    uv=np.floor(local/float(meta["resolution"])).astype(np.int64)
    h,w=shape;uv[:,1]=h-1-uv[:,1]
    ok=(uv[:,0]>=0)&(uv[:,0]<w)&(uv[:,1]>=0)&(uv[:,1]<h)
    return uv,ok

def grid_centers(uv,meta,shape):
    h,w=shape;local=np.column_stack([uv[:,0]+.5,h-uv[:,1]-.5])*meta["resolution"]
    yaw=meta["origin"][2];c,s=np.cos(yaw),np.sin(yaw)
    return local@np.array([[c,s],[-s,c]])+np.asarray(meta["origin"][:2])

def update_grid(original,meta,added,removed,retained,rows,ids,cfg):
    # Preserve every original pixel outside measured additions and fully seen
    # vertical clearance columns. Do not smooth away thin obstacles or doorways.
    out=original.copy(); floor=cfg["floor_z"]
    lo,hi=cfg["obstacle_height_m"]
    def obstacle(p): return p[(p[:,2]>=floor+lo)&(p[:,2]<=floor+hi)]
    old=obstacle(removed);candidate_cells,ok=grid_cells(old,meta,out.shape)
    candidate_cells=np.unique(candidate_cells[ok],axis=0)
    positions=independent(rows,ids,cfg["view_baseline_m"])
    eligible=np.ones(len(candidate_cells),bool)
    clear_samples=np.zeros(len(candidate_cells),np.int32);surface_samples=clear_samples.copy();samples=0
    unknown=np.zeros(len(candidate_cells),bool)
    occupied=np.zeros(out.shape,bool)
    for points in (obstacle(retained),obstacle(added)):
        uv,ok=grid_cells(points,meta,out.shape);occupied[uv[ok,1],uv[ok,0]]=True
    if len(candidate_cells):
        xy=grid_centers(candidate_cells,meta,out.shape)
        # Nine horizontal samples per cell and vertical spacing <= voxel size.
        offsets=(-.4,0,.4)
        for dx in offsets:
            for dy in offsets:
                for height in np.unique(np.r_[np.arange(lo,hi,cfg["clear_height_step_m"]),hi]):
                    p=np.column_stack([xy+np.array([dx,dy])*meta["resolution"],np.full(len(xy),floor+height)])
                    surface,free,_=evidence(p,rows,positions,cfg["surface_tolerance_m"])
                    checked=(free>=cfg["min_clear_views"])&(surface==0)
                    eligible &= checked
                    clear_samples+=checked;surface_samples+=(surface>0);samples+=1
        eligible &= ~occupied[candidate_cells[:,1],candidate_cells[:,0]]
        # Missing headroom evidence is unknown, never free. Only observed removal seeds
        # with substantial partial clearance and no contradictory surface qualify.
        blocked=occupied[candidate_cells[:,1],candidate_cells[:,0]]
        unknown=partial_clearance_unknown(clear_samples,samples,surface_samples,blocked,eligible)
        unsure=candidate_cells[unknown]
        old_occ=original[unsure[:,1],unsure[:,0]]<128 if not meta.get("negate",0) else original[unsure[:,1],unsure[:,0]]>127
        unsure=unsure[old_occ]
        # Mid-occupancy probability lies between the configured map thresholds.
        probability=(float(meta.get("free_thresh",.196))+float(meta.get("occupied_thresh",.65)))/2
        unknown_value=int(round(255*(probability if meta.get("negate",0) else 1-probability)))
        out[unsure[:,1],unsure[:,0]]=unknown_value
        good=candidate_cells[eligible];out[good[:,1],good[:,0]]=254 if not meta.get("negate",0) else 1
    uv,ok=grid_cells(obstacle(added),meta,out.shape)
    out[uv[ok,1],uv[ok,0]]=0 if not meta.get("negate",0) else 255
    diff=out!=original
    old_prob=(original.astype(float)/255) if meta.get("negate",0) else (1-original.astype(float)/255)
    new_prob=(out.astype(float)/255) if meta.get("negate",0) else (1-out.astype(float)/255)
    old_occ=old_prob>meta.get("occupied_thresh",.65)
    new_occ=new_prob>meta.get("occupied_thresh",.65)
    new_free=new_prob<meta.get("free_thresh",.196)
    new_unknown=~new_occ&~new_free
    return out,dict(actual_added_obstacles=int((~old_occ&new_occ).sum()),
                    actual_removed_obstacles=int((old_occ&new_free).sum()),
                    occupied_to_unknown=int((old_occ&new_unknown).sum()),changed_cells=int(diff.sum()),addition_cells=len(np.unique(uv[ok],axis=0)),
                    cleared_columns=int(eligible.sum()),clearance_height_m=[lo,hi],
                    policy="preserve_unobserved; observed_removal_partial_clearance_to_unknown; full_height_clearance_to_free",
                    limitation="discrete_clearance_samples_not_collision_certification")

def partial_clearance_unknown(clear_samples,total_samples,surface_samples,blocked,free):
    """Removed-surface seeds may become unknown, never traversable by this rule."""
    return (np.asarray(clear_samples)>=max(1,total_samples*.5))&(np.asarray(surface_samples)==0)&~np.asarray(blocked,bool)&~np.asarray(free,bool)
