"""Training-view leave-one-region-out guard before shared asset generation.

Rejecting an edit means deferred, never 'no change'. Historical bundles are not
mutated. Manual ROI behavior is deliberately untouched.
"""
import copy
import json
from pathlib import Path
import numpy as np
from .io import write_json, sha256, independent, output_path


def apply_decision(selected, ident, defer):
    """Return a new region list; an unrelated region can never be dropped."""
    if sum(r['id'] == ident for r in selected) != 1:
        raise ValueError('fusion_guard_region_identity_mismatch')
    return [r for r in selected if not (defer and r['id'] == ident)]


def select(source, regions, output):
    from .postprocess import load, rows_from_data, render_map
    from .spirula import fuse, asset
    from .roi import frame_manifest
    from .object_policy import accept_trial
    from vps.refinement.gaussian_renderer import GsplatGaussianRenderer
    import torch
    if not regions or len({r['id'] for r in regions}) != len(regions):
        raise ValueError('fusion_guard_requires_nonempty_unique_regions')
    output=output_path(output)
    asset(source, True)
    cfg,data,split,_=load(source); rows=rows_from_data(data)
    positions=independent(rows,split['train_views'],.15)
    ids=[positions[i] for i in np.unique(np.linspace(0,len(positions)-1,min(8,len(positions))).astype(int))] if positions else []
    crops={f['view']:f['crop'] for f in frame_manifest(source)['frames']}
    selected=copy.deepcopy(regions); records=[]
    identity=dict(schema='astra.fusion_guard.v1',regions=regions,
        views_sha256=sha256(source/'job/views.npz'),split_sha256=sha256(source/'job/refit_report.json'),
        reconstruction_sha256=sha256(source/'reconstruction.json'),train_views=ids,
        base_sha256=sha256(cfg['map_ply']),
        code_sha256={name:sha256(Path(__file__).parent/name) for name in
                     ['fusion_guard.py','spirula.py','object_policy.py','roi.py','postprocess.py','io.py']})
    if output.exists():
        if (output/'report.json').is_file() and json.loads((output/'inputs.json').read_text()) == identity:
            cached=json.loads((output/'report.json').read_text())
            return cached['regions'], cached
        raise ValueError('fusion_guard_incomplete_or_stale_output_choose_new_directory')
    output.mkdir()
    write_json(output/'inputs.json',identity)
    def renders(path):
        renderer=GsplatGaussianRenderer('cuda')
        result={i:render_map(renderer,path,rows[i])['rgb'] for i in ids}
        del renderer; torch.cuda.empty_cache()
        return result
    temp=output/'trial.ply'
    fuse(source,dict(mode='replace',regions=selected),temp)
    current=renders(temp)
    for region in regions:
        proposal=[r for r in selected if r['id']!=region['id']]
        if proposal:
            fuse(source,dict(mode='replace',regions=proposal),temp)
            after=renders(temp)
        else:
            after=renders(cfg['map_ply'])
        comparisons=[]
        for i in ids:
            x0,y0,x1,y1=crops[i]
            a=current[i][y0:y1,x0:x1].astype(np.float32)/255
            b=after[i][y0:y1,x0:x1].astype(np.float32)/255
            truth=data['rgb'][i,y0:y1,x0:x1]
            changed=np.abs(a-b).max(2)>2/255
            comparisons.append(dict(view=i,changed_pixels=int(changed.sum()),
                before_l1=float(np.abs(a-truth)[changed].mean()) if changed.any() else 0,
                after_l1=float(np.abs(b-truth)[changed].mean()) if changed.any() else 0))
        defer=accept_trial(comparisons)
        if defer:selected=apply_decision(selected,region['id'],True);current=after
        records.append(dict(id=region['id'],deferred=defer,views=comparisons,
                            reason='replacement_visually_regresses_on_training_views' if defer else 'retained_no_sufficient_evidence_to_defer'))
        write_json(output/'progress.json',dict(completed=len(records),total=len(regions),last=records[-1]))
        print('FUSION_GUARD',region['id'],defer,flush=True)
    temp.unlink(missing_ok=True)
    report=dict(schema='astra.fusion_guard.v1',regions=selected,decisions=records,
                train_views=ids,deferred=[r['id'] for r in records if r['deferred']],
                deployable=False,independent_validation=False,
                limitation='Training-view appearance guard; retained regions and deferred regions still need geometry and field review.')
    write_json(output/'report.json',report)
    return selected, report


if __name__ == '__main__':
    import argparse, fcntl
    from .io import ROOT
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--regions',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    with (ROOT/'runs/gpu.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        select(args.source,json.loads(args.regions.read_text()),args.output)
