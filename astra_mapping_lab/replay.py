"""ROI-limited offline depth evidence. Additions only; never writes navigation maps."""
import argparse,hashlib,io,json,math,time,zipfile
from collections import deque
from pathlib import Path,PurePosixPath
import numpy as np
from PIL import Image,ImageDraw
from depth_io import load_depth,validate_depth_manifest
from core import finite,normalize_metadata,rectangle,pixel_to_world,polygon,free_mask,clearance_mask

DEFAULTS=dict(pixel_stride=2,min_depth_m=.25,max_depth_m=6.,min_height_m=.15,max_height_m=1.8,
              independent_baseline_m=.20,min_views=3,min_duration_s=1.,min_area_m2=.08,
              boundary_margin_m=.10)


class ReplayCancelled(RuntimeError):pass


def asset(root,name):
    if not isinstance(name,str) or not name or '\\' in name:raise ValueError('数据文件必须是相对路径')
    parts=PurePosixPath(name)
    if parts.is_absolute() or any(p in ('..','') for p in parts.parts) or ':' in name:raise ValueError('数据文件路径越界')
    result=(Path(root)/name).resolve();root=Path(root).resolve()
    if root not in result.parents or not result.is_file():raise ValueError('数据文件不存在或超出数据目录')
    if result.stat().st_size>20*1024*1024:raise ValueError('单个数据文件超过 20 MB')
    return result


def unpack_dataset(raw,target):
    target=Path(target).resolve();target.mkdir(parents=True,exist_ok=False)
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        files=[i for i in z.infolist() if not i.is_dir()]
        if len(files)>600 or sum(i.file_size for i in files)>128*1024*1024:raise ValueError('数据包最多 600 个文件、解压后 128 MB')
        seen=set()
        for info in files:
            name=info.filename;p=PurePosixPath(name)
            if p.is_absolute() or '..' in p.parts or '\\' in name or ':' in name or name in seen:raise ValueError('数据包路径无效或重复')
            if (info.external_attr>>16)&0o170000==0o120000:raise ValueError('不接受符号链接')
            if p.suffix.lower() not in ('.json','.npy','.png','.jpg','.jpeg','.pgm'):raise ValueError('数据包含不支持的文件类型')
            if info.file_size>20*1024*1024:raise ValueError('单个文件超过 20 MB')
            seen.add(name);out=(target/name).resolve()
            if target not in out.parents:raise ValueError('数据包路径越界')
            out.parent.mkdir(parents=True,exist_ok=True)
            with z.open(info) as source,out.open('wb') as dest:
                total=0
                while True:
                    chunk=source.read(65536)
                    if not chunk:break
                    total+=len(chunk)
                    if total>info.file_size:raise ValueError('文件大小与声明不符')
                    dest.write(chunk)
    return read_dataset(target)


def fingerprint(image,meta):
    return hashlib.sha256(np.asarray(image.convert('RGBA')).tobytes()+json.dumps(meta,sort_keys=True).encode()).hexdigest()


def read_dataset(root):
    root=Path(root)
    if (root/'capture_manifest.json').exists():
        from trajectory import raw_manifest
        manifest=raw_manifest(root)
        with Image.open(asset(root,manifest['map_image'])) as im:
            if im.width*im.height>4_000_000:raise ValueError('地图过大')
            image=im.convert('RGBA')
        meta=normalize_metadata(manifest['map_metadata'],image.width,image.height)
        return manifest,image,meta
    return load_dataset(root,require_map=False)


def load_dataset(root,require_map=True):
    root=Path(root);manifest=json.loads(asset(root,'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('schema_version')!=1:raise ValueError('只支持离线格式 schema_version=1')
    if require_map and (manifest.get('world_convention')!='xy_ground_z_up' or manifest.get('map_alignment',{}).get('accepted') is False):raise ValueError('数据尚未对齐旧地图，请先完成地图对齐；可先做局部重建')
    if manifest.get('world_convention') not in ('xy_ground_z_up','camera_local'):raise ValueError('未声明有效坐标系')
    validate_depth_manifest(manifest)
    required={'camera_convention':'opencv',
              'depth_aligned_to':'color','depth_rectified':True}
    for k,v in required.items():
        if manifest.get(k)!=v:raise ValueError(f'{k} 必须明确为 {v}；不自动猜测标定与坐标')
    finite(manifest.get('ground_z_m'),'ground_z_m')
    with Image.open(asset(root,manifest.get('map_image'))) as im:
        if im.width*im.height>4_000_000:raise ValueError('地图过大')
        im.load();image=im.convert('RGBA')
    meta=normalize_metadata(manifest.get('map_metadata'),image.width,image.height)
    frames=manifest.get('frames')
    if not isinstance(frames,list) or not frames:raise ValueError('至少需要 1 帧')
    seen=set();last=-math.inf
    for f in frames:
        if not isinstance(f,dict) or not isinstance(f.get('id'),str) or not f['id'] or f['id'] in seen:raise ValueError('每帧必须有唯一字符串 id')
        seen.add(f['id']);timestamp=finite(f.get('timestamp_s'),'timestamp_s')
        if timestamp<=last:raise ValueError('时间戳必须严格递增')
        last=timestamp;asset(root,f.get('depth'))
        if 'rgb' in f:asset(root,f['rgb'])
        pose=np.asarray(f.get('pose_c2w'),dtype=float);K=np.asarray(f.get('K'),dtype=float)
        if pose.shape!=(4,4) or not np.isfinite(pose).all() or not np.allclose(pose[3],[0,0,0,1],atol=1e-6):raise ValueError('pose_c2w 无效')
        R=pose[:3,:3]
        if not np.allclose(R.T@R,np.eye(3),atol=1e-3) or not np.isclose(np.linalg.det(R),1,atol=1e-3):raise ValueError('相机旋转不是有效旋转矩阵')
        if K.shape!=(3,3) or not np.isfinite(K).all() or K[0,0]<=0 or K[1,1]<=0 or not np.allclose(K[2],[0,0,1]) or abs(K[0,1])+abs(K[1,0])>1e-6:raise ValueError('K 必须是无 skew 的有效内参')
        if f.get('pose_valid') not in (True,False) or type(f.get('pose_valid')) is not bool:raise ValueError('必须显式提供 pose_valid')
    return manifest,image,meta


def components(mask):
    seen=np.zeros_like(mask,bool);h,w=mask.shape
    for y,x in zip(*np.where(mask)):
        if seen[y,x]:continue
        seen[y,x]=True;queue=deque([(int(y),int(x))]);group=[]
        while queue:
            row,col=queue.popleft();group.append((row,col))
            for dr,dc in [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]:
                rr,cc=row+dr,col+dc
                if 0<=rr<h and 0<=cc<w and mask[rr,cc] and not seen[rr,cc]:seen[rr,cc]=True;queue.append((rr,cc))
        yield np.asarray(group,dtype=int)


def parameters(overrides=None):
    out=dict(DEFAULTS)
    if overrides:
        if not isinstance(overrides,dict) or set(overrides)-set(out):raise ValueError('回放参数包含未知字段')
        out.update(overrides)
    limits=dict(pixel_stride=(1,16),min_depth_m=(.1,3),max_depth_m=(.2,15),min_height_m=(0,3),max_height_m=(.1,5),independent_baseline_m=(.05,2),min_views=(2,20),min_duration_s=(.1,120),min_area_m2=(.001,10),boundary_margin_m=(0,1))
    for k,(lo,hi) in limits.items():out[k]=finite(out[k],k,lo,hi)
    for k in ('pixel_stride','min_views'):
        if out[k]!=int(out[k]):raise ValueError(k+' 必须是整数')
        out[k]=int(out[k])
    if out['min_depth_m']>=out['max_depth_m'] or out['min_height_m']>=out['max_height_m']:raise ValueError('上下界顺序无效')
    return out


def replay(root,roi,output,overrides=None,cancelled=lambda:False):
    start=time.perf_counter();manifest,image,meta=load_dataset(root);cfg=parameters(overrides)
    roi=rectangle(roi,meta['width'],meta['height']);h,w=meta['height'],meta['width'];res=meta['resolution']
    # Grid is only a 2-D prior. Never produce deletion/free-space updates from this path.
    free=free_mask(image,meta);trusted=clearance_mask(free,math.ceil(cfg['boundary_margin_m']/res))
    support=np.zeros((h,w),np.uint16);first=np.full((h,w),np.inf);last=np.full((h,w),-np.inf)
    high=np.full((h,w),-np.inf);low=np.full((h,w),np.inf);positions=[];logs=[];ox,oy,yaw=meta['origin'];c,s=math.cos(yaw),math.sin(yaw)
    for f in manifest['frames']:
        if cancelled():raise ReplayCancelled('回放已取消，未提交结果')
        row=dict(frame_id=f['id'],timestamp_s=f['timestamp_s']);pose=np.asarray(f['pose_c2w'],float)
        if not f['pose_valid']:row['status']='SKIPPED_INVALID_POSE';logs.append(row);continue
        if any(np.linalg.norm(pose[:3,3]-p)<cfg['independent_baseline_m'] for p in positions):row['status']='SKIPPED_REPEATED_VIEW';logs.append(row);continue
        depth=load_depth(asset(root,f['depth']),f)
        if depth.ndim!=2 or depth.size>4_000_000 or depth.dtype.kind not in 'fu':raise ValueError('深度必须是最多 400 万像素的二维数值数组')
        depth=depth.astype(np.float32);dh,dw=depth.shape;K=np.asarray(f['K'],float)
        if not (0<=K[0,2]<dw and 0<=K[1,2]<dh):raise ValueError('主点超出深度图，请核实内参与尺寸')
        if 'rgb' in f:
            with Image.open(asset(root,f['rgb'])) as rgb:
                if rgb.size!=(dw,dh):raise ValueError('RGB 与已对齐深度尺寸不一致')
        from masks import frame_mask
        valid=frame_mask(root,f,depth.shape)&np.isfinite(depth)&(depth>cfg['min_depth_m'])&(depth<cfg['max_depth_m'])
        # Reject edge pixels and holes; avoid wraparound neighbours at image borders.
        clean=valid.copy();clean[[0,-1],:]=False;clean[:,[0,-1]]=False
        for dy,dx in [(1,0),(-1,0),(0,1),(0,-1)]:
            neighbor=np.roll(depth,(dy,dx),(0,1));clean&=np.roll(valid,(dy,dx),(0,1))&(np.abs(neighbor-depth)<.08+.02*depth)
        yy,xx=np.indices(depth.shape);sample=clean&(yy%cfg['pixel_stride']==0)&(xx%cfg['pixel_stride']==0);y,x=np.where(sample);z=depth[y,x]
        xyz=np.column_stack([(x-K[0,2])*z/K[0,0],(y-K[1,2])*z/K[1,1],z])@pose[:3,:3].T+pose[:3,3]
        heights=xyz[:,2]-manifest['ground_z_m'];keep=(heights>=cfg['min_height_m'])&(heights<=cfg['max_height_m']);xyz=xyz[keep];heights=heights[keep]
        dx,dy=xyz[:,0]-ox,xyz[:,1]-oy;px=(c*dx+s*dy)/res;py=h-(-s*dx+c*dy)/res
        # Continuous ROI membership precedes conversion to cell indices.
        keep=(px>=roi[0])&(px<roi[2])&(py>=roi[1])&(py<roi[3])&(px>=0)&(px<w)&(py>=0)&(py<h)
        x=np.floor(px[keep]).astype(int);y=np.floor(py[keep]).astype(int);heights=heights[keep];row['in_roi_samples']=len(x)
        candidates=trusted[y,x];x=x[candidates];y=y[candidates];heights=heights[candidates]
        if len(x):
            cells=np.unique(y*w+x);rr,cc=np.divmod(cells,w);support[rr,cc]+=1
            first[rr,cc]=np.minimum(first[rr,cc],f['timestamp_s']);last[rr,cc]=np.maximum(last[rr,cc],f['timestamp_s'])
            np.minimum.at(low,(y,x),heights);np.maximum.at(high,(y,x),heights)
        positions.append(pose[:3,3].copy());row.update(status='USED',candidate_cells=int(len(np.unique(y*w+x))),valid_depth_fraction=float(valid.mean()));logs.append(row)
    duration=np.zeros((h,w));observed=support>0;duration[observed]=last[observed]-first[observed]
    persistent=(support>=cfg['min_views'])&(duration>=cfg['min_duration_s']);regions=[];suppressed=[];accepted=np.zeros((h,w),bool)
    for group in components(persistent):
        yy,xx=group.T;area=len(group)*res*res;box=[int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)]
        if area<cfg['min_area_m2']:
            suppressed.append(dict(reason='SMALL_CHANGE',area_m2=area,bbox_pixels=box));continue
        accepted[yy,xx]=True
        regions.append(dict(id='region-'+str(len(regions)+1),kind='PERSISTENT_ADDITION_CANDIDATE',
            bbox_pixels=box,world_polygon=polygon(box,meta),evidence_area_m2=area,
            evidence_cells=len(group),support_views_min=int(support[yy,xx].min()),support_views_max=int(support[yy,xx].max()),
            duration_min_s=float(duration[yy,xx].min()),height_range_m=[float(low[yy,xx].min()),float(high[yy,xx].max())],
            status='NEEDS_REVIEW',replacement_allowed=False))
    report=dict(schema_version=1,synthetic=bool(manifest.get('synthetic',False)),dataset_name=manifest.get('name','离线数据'),
        map_fingerprint=fingerprint(image,meta),roi_pixels=roi,roi_world_polygon=polygon(roi,meta),parameters=cfg,
        frames_total=len(logs),frames_used=len(positions),frames=logs,regions=regions,suppressed_regions=suppressed,
        candidate_cells=int(observed.sum()),persistent_cells=int(persistent.sum()),accepted_cells=int(accepted.sum()),
        elapsed_s=time.perf_counter()-start,execution_allowed=False,replacement_region=None,
        capabilities=dict(addition_candidates=True,removal_detection=False,appearance_change=False,semantic_classification=False),
        limitations=['只提供旧自由空间里的新增占据候选，不代表已确认长期静态变化。',
                    '不检测删除或纹理变化，不修改原栅格；未知、遮挡与深度缺失不会被清空。',
                    '结果依赖显式提供的标定、地面高度与位姿有效性；并未完成真实设备标定。',
                    '同一次回放中持续出现的人也可能形成候选，需要复访或后续动态物体处理。'])
    if cancelled():raise ReplayCancelled('回放已取消，未提交结果')
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    overlay=np.asarray(image).copy();overlay[observed,:3]=(overlay[observed,:3]*.35+np.array([226,169,71])*.65).astype(np.uint8)
    overlay[accepted,:3]=[40,155,122];overlay[observed,3]=255
    picture=Image.fromarray(overlay);draw=ImageDraw.Draw(picture);draw.rectangle(roi,outline=(36,107,191,255),width=max(1,int(.06/res)))
    picture.save(output/'evidence.png');(output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    return report


def main():
    p=argparse.ArgumentParser();p.add_argument('dataset',type=Path);p.add_argument('--roi',type=float,nargs=4,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    print(json.dumps(replay(args.dataset,args.roi,args.output),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
