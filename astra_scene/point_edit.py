"""Auditable refinement of scoped fusion. Geometry evidence is not instance truth.

Only training observations propose edits. The legacy scoped candidate remains the
fallback; rendering on training views gates adoption, never deployment.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from .io import sha256, write_json, independent

POLICY = dict(tolerance_m=.16, min_free=4, min_surface=3,
              uncovered_radius_m=.08, view_baseline_m=.15)


def decisions(surface, free, distance, policy=None):
    p = {**POLICY, **(policy or {})}
    surface, free, distance = map(np.asarray, (surface, free, distance))
    reject = (surface == 0) & (free >= p['min_free'])
    restore = (surface >= p['min_surface']) & (free == 0) & (distance > p['uncovered_radius_m'])
    return reject, restore


def accept(records):
    useful = [r for r in records if r['pixels'] >= 100]
    if len(useful) < 3:
        return False
    gain = np.array([r['before_l1'] - r['after_l1'] for r in useful])
    return bool(np.isfinite(gain).all() and gain.mean() > .0005
                and gain.min() >= -.005 and (gain > 0).mean() >= .6
                and all(r['background_delta'] <= .001
                        and r['missing_increase'] <= .02
                        and r['foreground_increase'] <= .02 for r in useful))


def refine(source, scope, baseline, lineage, output):
    """Caller owns the GPU lock; output is a new, immutable stage directory."""
    from plyfile import PlyData, PlyElement
    from .postprocess import load, rows_from_data, render_map
    from .geometry import evidence
    from .roi import contains, scope_mask
    from .quality import structural_metrics
    output = Path(output); output.mkdir(exist_ok=False)
    cfg, data, split, _ = load(source)
    rows = rows_from_data(data)
    ids = independent(rows, split['train_views'], POLICY['view_baseline_m'])
    identity = dict(schema='astra.point_edit.v1', scope=scope, policy=POLICY,
                    views=sha256(source/'job/views.npz'), baseline=sha256(baseline),
                    lineage=sha256(lineage), base=sha256(cfg['map_ply']),
                    split=sha256(source/'job/refit_report.json'),
                    code=sha256(Path(__file__)), train_views=ids)
    identity['key'] = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    write_json(output/'inputs.json', identity)
    base = PlyData.read(cfg['map_ply'], mmap=True)['vertex'].data
    candidate = PlyData.read(str(baseline), mmap=True)['vertex'].data
    with np.load(lineage) as archive:
        original_ids = archive['original_ids'].copy()
    xyz = lambda v: np.column_stack([v[n] for n in ('x','y','z')])
    fresh_ids = np.flatnonzero(original_ids < 0)
    fresh_xyz = xyz(candidate[fresh_ids])
    print('POINT_EDIT fresh evidence', len(fresh_ids), len(ids), flush=True)
    surface, free, occluded = evidence(fresh_xyz, rows, ids, POLICY['tolerance_m'])
    reject, _ = decisions(surface, free, np.zeros(len(surface)))
    base_xyz = xyz(base)
    available = np.ones(len(base), bool)
    available[original_ids[original_ids >= 0]] = False
    old_ids = np.flatnonzero(available & contains(base_xyz, scope) & (base['opacity'] > 0))
    print('POINT_EDIT old evidence', len(old_ids), flush=True)
    old_surface, old_free, old_occ = evidence(base_xyz[old_ids], rows, ids, POLICY['tolerance_m'])
    surviving_fresh = fresh_xyz[~reject]
    distance = (cKDTree(surviving_fresh).query(base_xyz[old_ids], workers=4)[0]
                if len(surviving_fresh) else np.full(len(old_ids), np.inf))
    _, restore = decisions(old_surface, old_free, distance)
    restored_ids = old_ids[restore]
    keep = np.ones(len(candidate), bool); keep[fresh_ids[reject]] = False
    restored = np.zeros(len(restored_ids), dtype=candidate.dtype)
    for name in base.dtype.names:
        restored[name] = base[name][restored_ids]
    trial = output/'proposal.ply'
    PlyData([PlyElement.describe(np.concatenate([candidate[keep], restored]), 'vertex')], text=False).write(str(trial))
    np.savez_compressed(output/'proposal_lineage.npz',
                        original_ids=np.r_[original_ids[keep], restored_ids],
                        owner=np.full(int(keep.sum())+len(restored_ids), -1, np.int64))
    np.savez_compressed(output/'evidence.npz', fresh_ids=fresh_ids, surface=surface,
                        free=free, occluded=occluded, reject=reject,
                        old_ids=old_ids, old_surface=old_surface, old_free=old_free,
                        old_occluded=old_occ, nearest_fresh_m=distance, restore=restore)
    selected_ids = [ids[i] for i in np.unique(np.linspace(0,len(ids)-1,min(8,len(ids))).astype(int))] if ids else []
    from vps.refinement.gaussian_renderer import GsplatGaussianRenderer
    import torch
    images = {}
    for name, path in [('before', baseline), ('after', trial)]:
        renderer = GsplatGaussianRenderer('cuda')
        images[name] = {i: render_map(renderer, path, rows[i]) for i in selected_ids}
        del renderer; torch.cuda.empty_cache()
    records = []
    for i in selected_ids:
        a, b = images['before'][i], images['after'][i]
        aa, bb = a['rgb'].astype(np.float32)/255, b['rgb'].astype(np.float32)/255
        mask = scope_mask(rows[i], scope) & rows[i]['valid']
        bg = rows[i]['valid'] & ~scope_mask(rows[i], scope)
        ma = structural_metrics(aa, a['depth'], a['alpha'], rows[i], scope, dict(min_pixels=100))
        mb = structural_metrics(bb, b['depth'], b['alpha'], rows[i], scope, dict(min_pixels=100))
        # Every adequately observed region must meet structural non-regression.
        pairs = list(zip(ma.get('regions',[ma]), mb.get('regions',[mb])))
        missing = [y.get('missing_surface_fraction',0)-x.get('missing_surface_fraction',0) for x,y in pairs if x['pixels']>=100]
        foreground = [y.get('foreground_disagreement_fraction',0)-x.get('foreground_disagreement_fraction',0) for x,y in pairs if x['pixels']>=100]
        records.append(dict(view=i, pixels=int(mask.sum()),
            before_l1=float(np.abs(aa-rows[i]['rgb'])[mask].mean()) if mask.any() else 0,
            after_l1=float(np.abs(bb-rows[i]['rgb'])[mask].mean()) if mask.any() else 0,
            background_delta=float(np.abs(bb-aa)[bg].mean()) if bg.any() else 0,
            missing_increase=max(missing, default=0), foreground_increase=max(foreground,default=0)))
    adopted = accept(records)
    report = dict(schema='astra.point_edit.v1', key=identity['key'], adopted=adopted,
                  rejected_new_points=int(reject.sum()), restored_old_points=len(restored_ids),
                  views=records, train_views=ids, candidate_sha256=sha256(trial),
                  evidence_sha256=sha256(output/'evidence.npz'),
                  selected='proposal' if adopted else 'baseline', deployable=False,
                  limitation='Estimated depth and Gaussian centers; no instance identity or free-space certificate. Baseline box edits outside this refinement remain.')
    write_json(output/'report.json', report)
    print('POINT_EDIT', json.dumps(report), flush=True)
    return (trial, output/'proposal_lineage.npz', report) if adopted else (baseline, lineage, report)
