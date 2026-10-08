"""Paired held-out rendering audit of a frozen original and cleaned Gaussian model."""
import argparse,json,fcntl,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np


def run(context,cleaned,dest):
    from connected_map_update import setup,write
    from PIL import Image
    cfg=json.loads(Path(context).read_text());fusion=Path(cfg['output']);dest=Path(dest);dest.mkdir(parents=True,exist_ok=False)
    ctx=setup(SimpleNamespace(source=Path(cfg['source']),alignment_dir=Path(cfg['alignment']),output=dest/'inputs',base_map=Path(cfg['base_map']),gpu=1))
    root,source,alignment,_,base,data,registration,rows,ids,held=ctx
    from astra_scene.geometry import backproject,grid_cells
    from astra_scene.postprocess import render_map
    allowed=np.array(Image.open('/home/xujie/astra_scene/runs/surface_grid_v3_pipeline_20260929/artifacts/allowed_scope.png'))>0
    original=fusion/'artifacts/scene.ply';floor=json.loads((alignment/'artifacts/floor.json').read_text())['floor_z']
    records=[];shown=[held[0],held[len(held)//2],held[-1]]
    with (root/'runs/gpu.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        from vps.refinement.gaussian_renderer import GsplatGaussianRenderer
        renderer=GsplatGaussianRenderer('cuda')
        for i in held:
            row=rows[i];xyz,_=backproject(row);v,u=np.where(row['valid']);uv,ok=grid_cells(xyz,data['map_metadata'],allowed.shape)
            ix=np.flatnonzero(ok);ok[ix]&=allowed[uv[ix,1],uv[ix,0]];ok&=(xyz[:,2]>=floor+.15)&(xyz[:,2]<=floor+1.8)
            mask=np.zeros(row['depth'].shape,bool);mask[v[ok],u[ok]]=True
            item=dict(view=i,pixels=int(mask.sum()))
            for name,path in [('before',original),('after',cleaned)]:
                r=render_map(renderer,path,row);delta=r['depth'][mask]-row['depth'][mask]
                item[name]=dict(rgb_mae=float(np.abs(r['rgb'][mask]/255.-row['rgb'][mask]).mean()),depth_mae=float(np.abs(delta).mean()),
                    depth_median=float(np.median(np.abs(delta))),foreground_fraction=float((delta<-.20).mean()),missing_fraction=float((delta>.20).mean()))
                if i in shown:Image.fromarray(r['rgb']).save(dest/f'{i:03d}_{name}.jpg',quality=94)
            if i in shown:
                rgb=np.uint8(np.clip(row['rgb']*255,0,255));rgb[~row['keep']]=rgb[~row['keep']]//6;Image.fromarray(rgb).save(dest/f'{i:03d}_capture.jpg',quality=94)
            records.append(item);print('AUDIT',i,flush=True)
    summary={name:{metric:float(np.average([x[name][metric] for x in records],weights=[x['pixels'] for x in records])) for metric in records[0][name]} for name in ('before','after')}
    write(dest/'audit.json',dict(views=records,weighted_summary=summary,shown_views=shown,held_out_used_for_selection=False,
        limitation='same-capture depth-aligned development audit; depth alignment has unresolved scale conflict'))
    print('RESULT',json.dumps(summary),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--context',type=Path,required=True);p.add_argument('--cleaned',type=Path,required=True);p.add_argument('--destination',type=Path,required=True);a=p.parse_args();run(a.context,a.cleaned,a.destination)
