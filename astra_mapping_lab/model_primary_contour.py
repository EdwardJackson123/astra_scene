"""Model-primary occupancy with bounded, evidence-supported contour repair."""
import numpy as np
from scipy.ndimage import binary_closing,label
from grid_common import classes

VERSION='model_primary_contour_v1'


def split_count(before,after):
    a,_=label(before);b,_=label(after);count=0
    for k in np.unique(a[before&~after]):
        if k==0:continue
        ids=np.unique(b[(a==k)&after]);ids=ids[ids>0]
        if len(ids)>1:count+=1
    return count


def refine(before,original,allowed,meta,model,shaped,hole_added,sensor_hits,ground_views,floor_views,erasure_veto,object_scopes):
    occupied,_,_=classes(before,meta);old_occupied,_,_=classes(original,meta)
    unsupported=occupied&allowed&~old_occupied&~model&~shaped&~hole_added
    confirmed_clear=unsupported&(sensor_hits==0)&(ground_views>=3)&(floor_views>=3)&~erasure_veto
    uncertain=unsupported&~confirmed_clear
    free_value=1 if meta.get('negate',0) else 254;occ_value=255 if meta.get('negate',0) else 0
    unknown_value=50 if meta.get('negate',0) else 205
    if not classes(np.array([unknown_value],np.uint8),meta)[2][0]:
        probability=(meta.get('free_thresh',.196)+meta.get('occupied_thresh',.65))/2
        unknown_value=int(round(255*(probability if meta.get('negate',0) else 1-probability)))
        if not classes(np.array([unknown_value],np.uint8),meta)[2][0]:raise ValueError('cannot represent unknown at supplied thresholds')
    candidate=before.copy();candidate[confirmed_clear]=free_value;candidate[uncertain]=unknown_value
    preview=before.copy();preview[unsupported]=free_value
    # Preview is NOT a navigable map. It is used only for shape comparisons.
    occ,preview_free,_=classes(preview,meta);_,safe_free,_=classes(candidate,meta)
    proposals=np.zeros_like(allowed);records=[]
    for name,scope in object_scopes.items():
        scope=scope&allowed
        patch=binary_closing(occ&scope,structure=np.ones((3,3)))&scope&~occ
        patch &= erasure_veto&(sensor_hits>=3)&(ground_views<3)
        proposals|=patch;records.append(dict(id=name,proposed_cells=int(patch.sum())))
    parts,n=label(proposals,np.ones((3,3)));added=np.zeros_like(allowed);rejected=[]
    for k in range(1,n+1):
        patch=parts==k;new=added|patch
        bad=split_count(preview_free&~added,preview_free&~new)+split_count(safe_free&~added,safe_free&~new)
        if bad:rejected.append(dict(patch=k,cells=int(patch.sum()),reason='free_component_split'))
        else:added|=patch
    candidate[added]=occ_value;preview[added]=occ_value
    assert np.array_equal(candidate[~allowed],before[~allowed])
    assert not np.any(unsupported&old_occupied) and not np.any(unsupported&model)
    report=dict(version=VERSION,reclassified_cells=int(unsupported.sum()),newly_free_cells=int(confirmed_clear.sum()),
        unknown_before_repair=int(uncertain.sum()),unknown_after_repair=int((uncertain&~added).sum()),
        proposed_cells=int(proposals.sum()),added_cells=int(added.sum()),patch_rejections=rejected,objects=records,
        outside_allowed_unchanged=True,legacy_obstacles_preserved=True,model_obstacles_preserved=True,
        preview_is_not_navigation=True,robot_clearance_verified=False,published=False)
    return candidate,preview,report,dict(reclassified=unsupported,confirmed_clear=confirmed_clear,uncertain=uncertain,added=added,proposed=proposals)
