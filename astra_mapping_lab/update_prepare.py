"""Generate this job's baseline views, floor estimate and RGB-D evidence."""
from gpu_config import gpu_index
import argparse,json,os,shutil
from pathlib import Path
from types import SimpleNamespace

def prepare(cfg):
    from connected_map_update import setup,write,evidence_stage
    import numpy as np,fcntl
    from PIL import Image
    from plyfile import PlyData
    from masked_metrics import appearance
    from map_update import roi_weights
    ctx=setup(SimpleNamespace(source=Path(cfg['source']),alignment_dir=Path(cfg['alignment']),
         output=Path(cfg['output']),base_map=Path(cfg['base_map']),gpu=gpu_index(),preflight=not bool(cfg.get('fresh_ply'))))
    root,source,alignment,out,base,data,registration,rows,ids,held=ctx
    dest=alignment/'artifacts';review=dest/'review';review.mkdir(parents=True,exist_ok=True)
    early=not bool(cfg.get('fresh_ply'))
    if early:
        vertices=PlyData.read(base,mmap=True)['vertex'].data
        xyz=np.column_stack([vertices[n] for n in ('x','y','z')])
        z=xyz[(roi_weights(xyz,data['map_metadata'],data['roi'],.001)>0)&(vertices['opacity']>0),2]
        band=cfg['floor_search_z'];z=z[(z>band[0])&(z<band[1])]
        if len(z)<100:raise ValueError('没有足够的原地图地面证据')
        hist,edges=np.histogram(z,bins=np.arange(band[0],band[1]+.025,.025));peak=int(hist.argmax())
        floor=float(np.median(z[(z>=edges[peak])&(z<edges[peak+1])]))
        write(dest/'floor.json',dict(floor_z=floor,method='registered-map ROI dominant low horizontal height; not independent calibration'))
    metrics=[] if early else json.loads((dest/'metrics.json').read_text())
    from vps.refinement.gaussian_renderer import GsplatGaussianRenderer,GaussianCamera
    with (root/'runs/gpu.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        for label,path in ([('old',base)] if early else [('patch',alignment/'aligned_masked_sh0.ply')]):
            renderer=GsplatGaussianRenderer('cuda')
            for i in held:
                row=rows[i];K=row['K'];h,w=row['depth'].shape;gt=np.uint8(np.rint(row['rgb']*255))
                if (h,w)!=(480,640):raise ValueError('当前更新评估仅适配已验证的 640×480 RGB-D')
                r=renderer.render(path,row['pose'],GaussianCamera(w,h,K[0,0],K[1,1],K[0,2],K[1,2]))
                Image.fromarray(r.rgb).save(review/(row['frame']+'_'+label+'.png'))
                Image.fromarray(gt).save(review/(row['frame']+'_gt.png'))
                metrics.append(dict(frame=row['frame'],label=label,**appearance(r.rgb,gt,row['keep'])))
            del renderer
            import gc,torch
            gc.collect();torch.cuda.empty_cache()
    write(dest/'metrics.json',metrics)
    if early:evidence_stage(ctx)
    cache=out/'raw_residual_review';cache.mkdir(exist_ok=True)
    if early:shutil.copyfile(out/'evidence_points.npz',cache/'evidence_points.npz')
    # Same-run baseline, never a previous experiment's views.
    write(out/'comparison.json',dict(previous_scene=str(base),additional_scenes=dict(single_spirula=str(alignment/'aligned_masked_sh0.ply'))))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--context',type=Path,required=True);a=p.parse_args()
    prepare(json.loads(a.context.read_text()))
