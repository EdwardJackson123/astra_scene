"""Evidence-owned update scopes. A grid ROI never becomes a replacement mask."""
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from map_update import roi_weights


def components(points, radius=.24, minimum_points=25):
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError('invalid component points')
    if not np.isfinite(radius) or radius <= 0 or minimum_points < 1:
        raise ValueError('invalid component policy')
    if not len(points):
        return []
    pairs = cKDTree(points).query_pairs(radius, output_type='ndarray')
    graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(points), len(points)))
    _, labels = connected_components(graph, directed=False)
    groups = [np.flatnonzero(labels == i) for i in range(labels.max() + 1)]
    return sorted((g for g in groups if len(g) >= minimum_points), key=lambda g: (-len(g), int(g[0])))


def scope_weights(points, supports, metadata, roi, core_m=.12, transition_m=.18, protected_supports=None):
    """Union of supported local volumes, intersected with permission ROI.

    Supports must already have passed change/semantic review. Bounds are never
    accepted here. Gaps wider than twice (core + transition) remain unchanged.
    """
    points = np.asarray(points, float)
    supports = np.asarray(supports, float).reshape(-1, 3)
    if not np.isfinite(supports).all() or not np.isfinite([core_m, transition_m]).all() or core_m < 0 or transition_m <= 0:
        raise ValueError('invalid scope support')
    permission = roi_weights(points, metadata, roi, transition_m)
    weights = np.zeros(len(points), float)
    allowed = np.flatnonzero(permission > 0)
    if len(supports) and len(allowed):
        # Also discard support outside the permission region before building it.
        supports = supports[roi_weights(supports, metadata, roi, transition_m) > 0]
        if len(supports):
            distance = cKDTree(supports).query(points[allowed], workers=4)[0]
            t = np.clip((core_m + transition_m - distance) / transition_m, 0, 1)
            weights[allowed] = t * t * (3 - 2*t) * permission[allowed]
    if protected_supports is not None:
        protected = np.asarray(protected_supports, float).reshape(-1,3)
        if not np.isfinite(protected).all():raise ValueError('invalid protected support')
        active = np.flatnonzero(weights>0)
        if len(protected) and len(active):
            d = cKDTree(protected).query(points[active], workers=4)[0]
            # Deferred/ignored structure wins wherever a transition touches it.
            t = np.clip((d-core_m)/transition_m,0,1)
            weights[active] *= t*t*(3-2*t)
    return weights


def validate_review(items, proposals):
    """No invented IDs/views/coordinates or failed API output can authorize edits."""
    expected = {p['id']: p for p in proposals}
    if not isinstance(items, list) or len(items) != len(expected):
        raise ValueError('incomplete review')
    if any(not isinstance(x, dict) for x in items):
        raise ValueError('invalid review item')
    if sorted(x.get('id', '') for x in items) != sorted(expected):
        raise ValueError('invalid review ids')
    for item in items:
        if item.get('decision') not in ('update', 'defer', 'ignore') or not isinstance(item.get('reason'), str):
            raise ValueError('invalid review decision')
        views = item.get('supporting_views')
        supplied = {im['view'] for im in expected[item['id']]['images']}
        if not isinstance(views, list) or any(type(v) is not int or v not in supplied for v in views):
            raise ValueError('invented evidence view')
        if item['decision'] == 'update' and (len(set(views)) < 2 or item.get('scope_complete') is not True):
            raise ValueError('update lacks multi-view complete scope')
        if any(k in item for k in ('bounds', 'coordinates', 'roi')):
            raise ValueError('review cannot create geometry')
    return items


def confirmed_stale(surface, free):
    """A transition cannot retain an obsolete surface just to blend its color.

    Strict zero-surface confirmation is kept separate from noisy majority votes.
    Navigation clearance is still checked independently over full columns.
    """
    return (np.asarray(surface)==0)&(np.asarray(free)>=6)
