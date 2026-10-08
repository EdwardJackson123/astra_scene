"""Pure grid classification and metrics shared by all grid backends.

Keep these helpers independent of rendering, service state and semantic models.
Historical entry modules re-export the helpers for compatibility.
"""
import numpy as np

def classes(grid,meta):
    probability=grid.astype(float)/255
    if not meta.get('negate',0):probability=1-probability
    occupied=probability>meta.get('occupied_thresh',.65)
    free=probability<meta.get('free_thresh',.196)
    return occupied,free,~occupied&~free


def constrained_result(original,proposed,allowed,meta):
    if original.shape!=proposed.shape or original.shape!=allowed.shape:raise ValueError('grid shape mismatch')
    result=np.where(allowed,proposed,original).astype(np.uint8)
    old_occ,_,old_unknown=classes(original,meta);occ,free,unknown=classes(result,meta)
    return result,dict(changed_cells=int((result!=original).sum()),added_obstacles=int((occ&~old_occ).sum()),
        cleared_obstacles=int((old_occ&free).sum()),occupied_to_unknown=int((old_occ&unknown).sum()),
        unknown_to_free=int((old_unknown&free).sum()),outside_allowed_unchanged=bool(np.array_equal(result[~allowed],original[~allowed])))


def fragmentation(mask,scope):
    from scipy.ndimage import label
    a=mask&scope;labels,n=label(a,np.ones((3,3),int));sizes=np.bincount(labels.ravel())[1:]
    return dict(components=int(n),small_components_le4=int((sizes<=4).sum()),small_component_cells=int(sizes[sizes<=4].sum()),
                boundary_edges=int(np.count_nonzero(a[1:]!=a[:-1])+np.count_nonzero(a[:,1:]!=a[:,:-1])))


