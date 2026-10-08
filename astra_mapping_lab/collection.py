"""Static collection route and reconstruction package; no navigation commands."""
import heapq,json,math,shutil,zipfile
from pathlib import Path
import numpy as np
from PIL import Image
from core import clearance_mask,free_mask,world_to_pixel,pixel_to_world,finite
from replay import load_dataset,asset
from depth_io import load_depth
from masks import frame_mask


def collection_route(image,meta,plan,start_xy):
    if not isinstance(start_xy,list) or len(start_xy)!=2:raise ValueError('请填写路线起点世界坐标 x、y')
    start_xy=[finite(v,'起点') for v in start_xy]
    safe=clearance_mask(free_mask(image,meta),math.ceil(plan['preview_clearance_m']/meta['resolution']))
    p=world_to_pixel(*start_xy,meta);start=(math.floor(p[0]),math.floor(p[1]));h,w=safe.shape
    if not (0<=start[0]<w and 0<=start[1]<h and safe[start[1],start[0]]):raise ValueError('起点不满足静态栅格净空，请核对坐标')
    remaining={v['id']:v for v in plan['viewpoints']};current=start;segments=[];unreachable=[]
    while remaining:
        targets={(int(v['pixel'][0]),int(v['pixel'][1])):k for k,v in remaining.items()}
        queue=[(0,current)];dist={current:0};parent={};found=None
        while queue:
            d,node=heapq.heappop(queue)
            if d!=dist[node]:continue
            if node in targets:found=node;break
            x,y=node
            for dx,dy in [(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,1),(1,-1),(-1,-1)]:
                xx,yy=x+dx,y+dy
                if not (0<=xx<w and 0<=yy<h and safe[yy,xx]):continue
                if dx and dy and not (safe[y,xx] and safe[yy,x]):continue
                nxt=(xx,yy);nd=d+math.hypot(dx,dy)
                if nd<dist.get(nxt,float('inf')):dist[nxt]=nd;parent[nxt]=node;heapq.heappush(queue,(nd,nxt))
        if found is None:unreachable=list(remaining);break
        key=targets[found];view=remaining.pop(key);path=[found]
        while path[-1]!=current:path.append(parent[path[-1]])
        path.reverse();pixels=[[x+.5,y+.5] for x,y in path]
        segments.append(dict(view_id=key,pixels=pixels,world_xy=[pixel_to_world(x,y,meta) for x,y in pixels],length_m=dist[found]*meta['resolution'],yaw=view['yaw']))
        current=found
    return dict(start_xy=start_xy,segments=segments,unreachable=unreachable,length_m=sum(s['length_m'] for s in segments),execution_allowed=False,
                status='STATIC_ROUTE_FOR_MANUAL_COLLECTION',limitations=['仅静态栅格路径，未验证实时障碍、转向和三维覆盖。','供人工遥控补采参考；不会下发运动目标。'])


def rotation_quaternion(R):
    # Stable eigenvector conversion, scalar-first COLMAP quaternion.
    r=np.asarray(R);K=np.array([[r[0,0]-r[1,1]-r[2,2],r[1,0]+r[0,1],r[2,0]+r[0,2],r[2,1]-r[1,2]],
        [r[1,0]+r[0,1],r[1,1]-r[0,0]-r[2,2],r[2,1]+r[1,2],r[0,2]-r[2,0]],
        [r[2,0]+r[0,2],r[2,1]+r[1,2],r[2,2]-r[0,0]-r[1,1],r[1,0]-r[0,1]],
        [r[2,1]-r[1,2],r[0,2]-r[2,0],r[1,0]-r[0,1],np.trace(r)]])/3
    _,v=np.linalg.eigh(K);q=v[:,-1][[3,0,1,2]];return q if q[0]>=0 else -q


def export_spirula(dataset,target,roi,context,pipeline_profile=None):
    manifest,_,meta=load_dataset(dataset,require_map=False);local=manifest.get('world_convention')=='camera_local';target=Path(target);images=target/'dataset/images';sparse=target/'dataset/sparse/0'
    from trajectory import require_trajectory_quality
    require_trajectory_quality(manifest)
    images.mkdir(parents=True,exist_ok=False);sparse.mkdir(parents=True)
    frames=[f for f in manifest['frames'] if f['pose_valid'] and f.get('rgb')]
    if len(frames)<4:raise ValueError('重建至少需要 4 帧有效 RGB、深度及地图相机位姿')
    from keyframes import select
    selected,selection=select(dataset,frames)
    cameras=[];image_lines=[];points=[];splits=[];point_id=0
    for i,(f,split) in enumerate(selected,1):
        rgb=np.asarray(Image.open(asset(dataset,f['rgb'])).convert('RGB'));depth=load_depth(asset(dataset,f['depth']),f);h,w=depth.shape
        if rgb.shape[:2]!=(h,w):raise ValueError('RGB-D 尺寸不一致')
        K=np.array(f['K']);pose=np.array(f['pose_c2w']);w2c=np.linalg.inv(pose);q=rotation_quaternion(w2c[:3,:3]);t=w2c[:3,3]
        name=f'{i:06d}_{split}.png';Image.fromarray(rgb).save(images/name)
        keep=frame_mask(dataset,f,(h,w))
        if f.get('mask'):
            maskdir=target/'dataset/masks';maskdir.mkdir(exist_ok=True)
            Image.fromarray(keep.astype(np.uint8)*255).save(maskdir/name)
        cameras.append(f'{i} PINHOLE {w} {h} {K[0,0]} {K[1,1]} {K[0,2]} {K[1,2]}')
        image_lines.append(' '.join(map(str,[i,*q,*t,i,name])));tracks=[]
        # Only training depth initializes points; evaluation images are held out.
        if split=='train':
            for y in range(0,h,12):
                for x in range(0,w,12):
                    if not keep[y,x]:continue
                    z=float(depth[y,x])
                    if not math.isfinite(z) or not .25<z<6:continue
                    xyz=pose[:3,:3]@np.array([(x-K[0,2])*z/K[0,0],(y-K[1,2])*z/K[1,1],z])+pose[:3,3]
                    px,py=world_to_pixel(*xyz[:2],meta)
                    if not local and not (context[0]<=px<context[2] and context[1]<=py<context[3]):continue
                    point_id+=1;index=len(tracks);tracks.append(f'{x} {y} {point_id}')
                    points.append(' '.join(map(str,[point_id,*xyz,*rgb[y,x],0,i,index])))
        image_lines.append(' '.join(tracks));splits.append(dict(frame_id=f['id'],name=name,split=split))
    if len(points)<20:raise ValueError('选区及上下文内有效深度点不足，需补采')
    for name,lines in [('cameras.txt',cameras),('images.txt',image_lines),('points3D.txt',points)]:
        (sparse/name).write_text('\n'.join(lines)+'\n',encoding='utf-8')
    spec=dict(schema_version=1,backend='spirula',synthetic=manifest.get('synthetic',False),map_metadata=meta,roi_pixels=roi,context_pixels=context,
        mask_policy='ignore_black_observations' if any(f.get('mask') for f,_ in selected) else 'none',frames=splits,sfm_rgbd_report=manifest.get('sfm_rgbd_report'),keyframe_selection=selection,trajectory_quality=manifest.get('trajectory_report',{}).get('quality_gate'),reconstruction_scope='sfm_local_subset' if manifest.get('sfm_rgbd_report') else 'connected_local_subset' if manifest.get('trajectory_report',{}).get('quality_gate',{}).get('accepted') is False else 'provided_views',initial_points=len(points),coordinate_frame='camera_local' if local else 'source_map',map_aligned=not local,depth_usage='initialization_only',
        status='READY_FOR_LOCAL_RECONSTRUCTION' if local else 'READY_FOR_RECONSTRUCTION',automatic_application=False,quality_accepted=False,
        limitations=['沿用输入相机位姿，尚未做独立位姿/多视角几何验收。','RGB-D 用于初始化，未声明训练深度监督已接通。'])
    if pipeline_profile is not None:spec['pipeline_profile']=pipeline_profile
    (target/'package.json').write_text(json.dumps(spec,ensure_ascii=False,indent=2),encoding='utf-8')
    shutil.copyfile(Path(__file__).with_name('reconstruct.py'),target/'reconstruct.py')
    (target/'README.txt').write_text('运行：python reconstruct.py --package . --binary /path/to/spirula --device auto\n默认 SH0，仅生成候选。不得将其自动替换正式地图。\n',encoding='utf-8')
    archive=target/'reconstruction.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in target.rglob('*'):
            if p.is_file() and p!=archive:z.write(p,p.relative_to(target))
    return spec
