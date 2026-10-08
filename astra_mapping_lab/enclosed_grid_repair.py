"""Experimental enclosed-hole repair; no global closing or dilation."""
import numpy as np
from scipy.ndimage import binary_fill_holes, label


def repair(occupied, free, allowed, hit_views, ground_views, max_cells=16):
    """Use 8-connected background so diagonal openings remain open.

    Weak hit counts support candidates, not verified physical occupancy.
    Repeated ground observations conservatively veto the entire hole; floor
    visibility alone does not prove robot-height clearance.
    """
    shape = occupied.shape
    if occupied.ndim != 2 or any(a.shape != shape for a in (free, allowed, hit_views, ground_views)):
        raise ValueError('hole repair evidence shape mismatch')
    if any(not np.isfinite(a).all() or (a < 0).any() for a in (hit_views, ground_views)):
        raise ValueError('invalid observation counts')
    holes = binary_fill_holes(occupied, structure=np.ones((3, 3))) & ~occupied
    labels, count = label(holes, np.ones((3, 3)))
    added = np.zeros_like(occupied, dtype=bool)
    records = []
    for i in range(1, count + 1):
        m = labels == i
        if not (m & allowed).any():
            continue
        size = int(m.sum())
        if size > max_cells:
            reason = 'too_large'
        elif not np.all(allowed[m] & free[m]):
            reason = 'outside_scope_or_unknown'
        elif np.any(ground_views[m] >= 3):
            reason = 'repeated_ground_observation'
        elif np.any(hit_views[m] < 1) or not np.any(hit_views[m] >= 2):
            reason = 'insufficient_obstacle_evidence'
        else:
            reason = 'candidate_fill'
            added[m] = True
        y, x = np.where(m)
        records.append(dict(id=i, cells=size, reason=reason,
                            box=[int(x.min()), int(y.min()), int(x.max()+1), int(y.max()+1)]))
    return added, records


def apply_to_grid(grid, meta, allowed, hit_views, ground_views):
    from navigation_candidate import classes
    from scipy.ndimage import distance_transform_edt
    resolution = float(meta['resolution'])
    if not np.isfinite(resolution) or resolution <= 0:
        raise ValueError('invalid resolution')
    occupied, free, _ = classes(grid, meta)
    # Preserve the tested 0.04 m² maximum across map resolutions.
    added, records = repair(occupied, free, allowed, hit_views, ground_views,
                            max_cells=max(0, int(round(.04 / resolution**2))))
    before_labels, _ = label(free)
    core = free & (distance_transform_edt(free) * resolution >= .20)
    after_free = free & ~added
    after_labels, _ = label(after_free)
    splits = []
    for key in np.unique(before_labels[core]):
        if not key: continue
        parts = np.unique(after_labels[core & (before_labels == key) & after_free])
        if len(parts[parts > 0]) > 1: splits.append(int(key))
    lost = int((core & added).sum())
    accepted = not splits and not lost
    if not accepted: added[:] = False
    result = grid.copy()
    result[added] = 255 if meta.get('negate', 0) else 0
    return result, added, dict(version='enclosed_evidence_v1', added_cells=int(added.sum()),
        holes=records, topology_accepted=accepted, split_core_components=splits,
        proposed_lost_core_cells=lost, reference_clearance_m=.20,
        robot_clearance_certified=False, max_hole_area_m2=.04)
