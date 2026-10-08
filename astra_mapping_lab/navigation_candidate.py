"""Measured RGB-D grid candidate, independent of visual Gaussian retirement."""
from gpu_config import gpu_index
import argparse,json,time,hashlib,zipfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np


# Compatibility imports for historical scripts and snapshots.
from grid_common import classes, constrained_result


def run(context,destination=None):
    from connected_map_update import setup,write
    from PIL import Image,ImageDraw,ImageFont
    from plyfile import PlyData
    from scipy.spatial import cKDTree
    import yaml
    cfg=json.loads(Path(context).read_text());fusion=Path(cfg['output']);dest=Path(destination) if destination else fusion/'artifacts/navigation_candidate'
    if (dest/'navigation_update.json').exists():raise ValueError('grid candidate exists; preserve it and use a new output')
    dest.mkdir(parents=True,exist_ok=True);start=time.monotonic()
    ctx=setup(SimpleNamespace(source=Path(cfg['source']),alignment_dir=Path(cfg['alignment']),output=dest/'inputs',base_map=Path(cfg['base_map']),gpu=gpu_index()))
    root,source,alignment,_,base,data,registration,rows,ids,held=ctx
    from astra_scene.geometry import grid_cells,grid_centers,update_grid,evidence,backproject
    from astra_scene.io import independent
    from geometry_boundary import GeometryBoundary
    from artifact_cache import digest
    original=np.array(Image.open(source/'map.png').convert('L'));meta=data['map_metadata']
    if original.shape!=(meta['height'],meta['width']):raise ValueError('map metadata shape mismatch')
    if meta.get('mode','trinary')!='trinary' or meta['resolution']<=0:raise ValueError('unsupported map configuration')
    regions=json.loads((fusion/'artifacts/context_scope.json').read_text())['regions'];support=np.load(fusion/'assembly_support.npz')
    physical=np.load(fusion/'artifacts/supported_depth_points.npy');floor=json.loads((alignment/'artifacts/floor.json').read_text())['floor_z']
    assemblies=[support[r['id']] for r in regions if r['id'] in support]
    field=GeometryBoundary(assemblies,physical,floor)
    yy,xx=np.indices(original.shape);uv=np.column_stack([xx.ravel(),yy.ravel()]);xy=grid_centers(uv,meta,original.shape)
    ix=np.floor((xy-field.origin)/field.resolution).astype(int)
    valid=(ix[:,0]>=0)&(ix[:,1]>=0)&(ix[:,0]<field.field.shape[1])&(ix[:,1]<field.field.shape[0]);footprint=np.zeros(len(xy),bool)
    footprint[valid]=field.field[ix[valid,1],ix[valid,0]]
    roi=data['roi'];permission=(xx+.5>=roi[0])&(xx+.5<roi[2])&(yy+.5>=roi[1])&(yy+.5<roi[3]);allowed=footprint.reshape(original.shape)&permission
    def permitted(points):
        pixels,ok=grid_cells(points,meta,original.shape);mask=np.zeros(len(points),bool)
        mask[ok]=allowed[pixels[ok,1],pixels[ok,0]];return mask
    # Actual connected support, not the union's enclosing box or the full ROI.
    points=physical[permitted(physical)&(physical[:,2]>=floor+.15)&(physical[:,2]<=floor+1.8)]
    if len(points):points=points[cKDTree(np.concatenate(assemblies)).query(points,workers=4)[0]<.18]
    positions=independent(rows,ids,.20)
    if set(positions)&set(held):raise ValueError('holdout leakage into grid update')
    hit,clear,_=evidence(points,rows,positions,.12);added=points[(hit>=3)&(clear<=1)]
    # Visual retirement only nominates columns; it never proves navigable space.
    removed=np.load(fusion/'artifacts/removed_old_centers.npy');removed=removed[permitted(removed)]
    old=PlyData.read(base,mmap=True)['vertex'].data;lineage=np.load(fusion/'artifacts/lineage.npz');retained_ids=lineage['old_ids']
    retained_ids=retained_ids[old['opacity'][retained_ids]>0]
    retained=np.column_stack([old[n][retained_ids] for n in ('x','y','z')])
    params=dict(floor_z=floor,obstacle_height_m=[.15,1.8],view_baseline_m=.20,surface_tolerance_m=.12,min_clear_views=4,clear_height_step_m=.05)
    proposal,raw=update_grid(original,meta,added,removed,retained,rows,positions,params)
    result,counts=constrained_result(original,proposal,allowed,meta)
    from enclosed_grid_repair import apply_to_grid
    hit_views=np.zeros(original.shape,np.int32);ground_views=np.zeros(original.shape,np.int32)
    for position in positions:
        observed,_=backproject(rows[position],stride=2)
        pixels,ok=grid_cells(observed,meta,original.shape)
        for tally,height in ((hit_views,(observed[:,2]>=floor+.15)&(observed[:,2]<=floor+1.8)),
                             (ground_views,np.abs(observed[:,2]-floor)<.10)):
            unique=np.unique(pixels[ok&height],axis=0)
            tally[unique[:,1],unique[:,0]]+=1
    Image.fromarray(result).save(dest/'before_hole_repair.png')
    result,repaired,hole_report=apply_to_grid(result,meta,allowed,hit_views,ground_views)
    result,counts=constrained_result(original,result,allowed,meta)
    write(dest/'hole_repair.json',hole_report)
    np.savez_compressed(dest/'hole_repair_evidence.npz',sensor_hit_views=hit_views,
                        ground_views=ground_views,added=repaired,train_views=np.asarray(positions))
    Image.fromarray(original).save(dest/'original_map.png');Image.fromarray(result).save(dest/'map.png');Image.fromarray(np.uint8(allowed)*255).save(dest/'allowed_scope.png')
    mapcfg={k:meta[k] for k in ('resolution','origin','negate','occupied_thresh','free_thresh','mode') if k in meta};mapcfg['image']='map.png'
    for name in ('map.yaml','config.yaml'):(dest/name).write_text(yaml.safe_dump(mapcfg,sort_keys=False),encoding='utf-8')
    report=dict(**counts,state='CANDIDATE_READY',published=False,navigation_deployment_accepted=False,
        original_map_sha256=digest(source/'map.png'),candidate_map_sha256=digest(dest/'map.png'),base_ply_sha256=registration['base_sha256'],
        candidate_ply_sha256=digest(fusion/'artifacts/scene.ply'),source_manifest_sha256=digest(source/'manifest.json'),
        region_ids=[r['id'] for r in regions],allowed_cells=int(allowed.sum()),roi_role='permission_only',
        scope='per connected assembly footprints clipped by ROI; no whole ROI clearing',
        measured_added_points=len(added),train_views=positions,held_out_used=False,
        grid_parameters=params,raw_engine_report=raw,hole_repair=hole_report,metric_scale_conflict=registration.get('metric_scale_conflict',True),
        clearance_policy='9 samples per cell per height; every height from .15 to 1.8 requires >=4 clear views and no surface support; retained obstacles veto clearance',
        limitations=['absolute metric scale unverified','robot dimensions and clearance height not calibrated','no live navigation validation','sampled clearance is not a collision certificate','obstacle inflation left to navigation costmap'],
        elapsed_seconds=time.monotonic()-start)
    write(dest/'navigation_update.json',report)
    oc,_,_=classes(original,meta);nc,nf,nu=classes(result,meta)
    diff=np.stack([original]*3,axis=-1);diff[nc&~oc]=[230,65,55];diff[oc&nf]=[40,155,245];diff[oc&nu]=[240,170,35]
    Image.fromarray(diff).save(dest/'changes.png')
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',20);canvas=Image.new('RGB',(1800,1170),(242,246,249));draw=ImageDraw.Draw(canvas)
    box=(max(0,int(roi[0])-10),max(0,int(roi[1])-10),min(original.shape[1],int(roi[2])+10),min(original.shape[0],int(roi[3])+10))
    for j,(name,title) in enumerate([('original_map.png','Original'),('map.png','Updated candidate'),('changes.png','Red: added | Blue: cleared | Gold: unknown')]):
        im=Image.open(dest/name).convert('RGB');draw.text((j*600+5,10),title,font=font,fill=(20,35,50));canvas.paste(im,(j*600,50))
        crop=im.crop(box);crop.thumbnail((590,360),Image.Resampling.NEAREST)
        factor=min(590/(box[2]-box[0]),360/(box[3]-box[1]));crop=im.crop(box).resize((int((box[2]-box[0])*factor),int((box[3]-box[1])*factor)),Image.Resampling.NEAREST)
        canvas.paste(crop,(j*600,780))
    canvas.save(dest/'grid_comparison.png')
    text=f'<meta charset="utf-8"><title>导航栅格候选</title><style>body{{font:18px system-ui;max-width:1800px;margin:24px}}img{{width:100%}}</style><h1>最新连通区域生成的栅格候选</h1><p>新增障碍 {counts["added_obstacles"]} 格；确认清除 {counts["cleared_obstacles"]} 格；占用转未知 {counts["occupied_to_unknown"]} 格。仅生成候选，未应用于机器人；尺度与通行高度仍待独立验证。</p><p>红色：新增障碍；蓝色：确认清除；黄色：占用转未知。未知不是可通行，栅格未作机器人尺寸膨胀。</p><img src="grid_comparison.png"><p><a href="navigation_candidate.zip">下载地图及配置包</a> · <a href="navigation_update.json">证据与限制</a></p>'
    (dest/'index.html').write_text(text,encoding='utf-8')
    with zipfile.ZipFile(dest/'navigation_candidate.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name in ('map.png','map.yaml','config.yaml','navigation_update.json','hole_repair.json'):z.write(dest/name,name)
    print(json.dumps(counts),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--context',type=Path,required=True);p.add_argument('--destination',type=Path);a=p.parse_args();run(a.context,a.destination)
