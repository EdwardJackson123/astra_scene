"""Run the established Boxer detector on genuine aligned, masked RGB-D.

The adapter does not impersonate a phone/VGGT job. It keeps native image
geometry, names the sensor depth source, and uses training observations only.
Detection boxes are proposals, never instructions to erase a map volume.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import types
import numpy as np


def prepare(args):
    from connected_map_update import setup, write
    root, source, alignment, out, base, data, registration, rows, ids, held = setup(args)
    selected = [ids[j] for j in np.unique(np.linspace(0, len(ids)-1, 32).round().astype(int))]
    if set(selected) & set(held):
        raise ValueError('held-out observations cannot create proposals')
    current = dict(rgb=[], depths=[], confidence=[], poses=[], intrinsics=[])
    old = dict(rgb=[], depth=[], alpha=[])
    for i in selected:
        r = rows[i]
        image = r['rgb'].copy()
        image[~r['keep']] = 0
        for key, value in zip(current, [image, r['depth'], r['valid'], r['pose'], r['K']]):
            current[key].append(value)
        cached = np.load(args.render_cache/f'{i:03d}.npz')
        for key in old:
            value = cached[key].copy()
            if key == 'rgb' and value.dtype != np.uint8:
                raise ValueError('cached render RGB must be uint8')
            old[key].append(value)
    np.savez_compressed(out/'boxer_views.npz', **{k:np.asarray(v, dtype=(bool if k=='confidence' else np.float32)) for k,v in current.items()})
    np.savez_compressed(out/'old_renders.npz', **{k:np.asarray(v) for k,v in old.items()})
    write(out/'boxer_protocol.json', dict(
        source='masked_D455F_sensor_depth_scaled_to_verified_map_alignment',
        metric_scale_conflict=registration.get('metric_scale_conflict', True),
        depth_multiplier=data['depth_multiplier'], native_shape=list(rows[0]['depth'].shape),
        view_mapping=[dict(boxer_view=j, aligned_view=i, frame=rows[i]['frame']) for j,i in enumerate(selected)],
        train=list(range(len(selected))), validation=[], test=[],
        held_out_aligned_views=held, synthetic_old_map=True,
        publish_eligible=False, boxes_authorize_deletion=False))
    print('BOXER_INPUT_READY', len(selected), flush=True)


def detect(args):
    import fcntl
    root = Path('/home/xujie/astra_scene')
    sys.path.insert(0, str(root))
    from astra_scene.pipeline import environment
    os.environ.update(environment(args.gpu))
    os.environ['VPS_GPU_MEMORY_FRACTION'] = '.8'
    assets = root/'vendor/boxer_assets'
    sys.path.insert(0, str(assets/'boxer'))
    out = args.output.resolve()
    protocol = json.loads((out/'boxer_protocol.json').read_text())
    adapter = types.ModuleType('prepare_trial')
    adapter.EXP = out
    adapter.write_json = lambda path, x: Path(path).write_text(json.dumps(x, indent=2))
    adapter.valid_depth = lambda depth, confidence: confidence.astype(bool) & np.isfinite(depth) & (depth>.4) & (depth<6*protocol['depth_multiplier'])
    adapter.load_data = lambda: (dict(np.load(out/'boxer_views.npz')), protocol['train'], [], [])
    sys.modules['prepare_trial'] = adapter
    path = root/'vendor/phone/legacy_phone_v2/run_boxer_native_inputs.py'
    spec = importlib.util.spec_from_file_location('rgbd_boxer_native', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.HERE = assets
    module.EXP = out
    module.OUT = out/'boxer_native_game_v1'
    module.LABELS = json.loads((root/'configs/xichuang0915.json').read_text())['target_labels'].split(',')
    with (root/'runs/gpu.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        module.main()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['prepare', 'detect', 'regions', 'retry-review', 'segment', 'fuse', 'evaluate', 'geometry-review'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--alignment-dir', type=Path)
    parser.add_argument('--base-map', type=Path)
    parser.add_argument('--render-cache', type=Path)
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--build-only',action='store_true')
    parser.add_argument('--preflight',action='store_true',help='RGB-D-only stages before Gaussian training')
    parser.add_argument('--fusion-mode',choices=['geometry','protected','context'],default='geometry',
                        help='protected preserves architecture; context is the legacy mixed experiment')
    parser.add_argument('--prior',type=Path,help='optional existing Boxer ownership result to reuse for regional fusion')
    parser.add_argument('--fresh-cache',type=Path,help='optional matching training render cache for architecture protection')
    args = parser.parse_args()
    if args.phase in ('regions','retry-review','segment','fuse','evaluate','geometry-review'):
        from connected_map_update import setup
        ctx=setup(args)
        if args.phase=='regions':
            from boxer_regions import prepare as regions, review
            regions(ctx,args.render_cache.parent)
            review(ctx)
        elif args.phase=='retry-review':
            from boxer_regions import retry_failed
            retry_failed(ctx)
        elif args.phase=='segment':
            from boxer_ownership import segment
            segment(ctx,args.render_cache.parent)
        elif args.phase=='evaluate':
            from connected_map_update import evaluate_stage
            evaluate_stage(ctx,include_extra_views=False)
        elif args.phase=='geometry-review':
            from connected_map_update import geometry_review_stage
            geometry_review_stage(ctx)
        else:
            from connected_map_update import context_stage,evaluate_stage,geometry_review_stage
            if args.fusion_mode=='context':
                context_stage(ctx,ownership_policy='boxer')
            else:
                from protected_takeover import build,segment_architecture,ensure_fresh_cache
                prior=args.prior or args.output
                if not (prior/'boxer_weights.npz').exists() or (args.fusion_mode=='geometry' and not (prior/'assembly_support.npz').exists()):
                    if args.prior:raise ValueError('specified prior has no Boxer ownership evidence')
                    from boxer_ownership import ownership
                    from plyfile import PlyData
                    old=PlyData.read(args.base_map,mmap=True)['vertex'].data
                    new=PlyData.read(args.alignment_dir/'aligned_masked_sh0.ply',mmap=True)['vertex'].data
                    ownership(ctx,old,new)
                if args.render_cache is None:raise ValueError('protected fusion requires matching --render-cache of original-map views')
                fresh=args.fresh_cache or ensure_fresh_cache(ctx,prior)
                segment_architecture(ctx,prior,args.render_cache)
                build(ctx,prior,args.render_cache,fresh,geometry=args.fusion_mode=='geometry')
            if not args.build_only:
                evaluate_stage(ctx,include_extra_views=False)
                geometry_review_stage(ctx)
    else:
        (prepare if args.phase == 'prepare' else detect)(args)
