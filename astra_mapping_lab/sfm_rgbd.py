"""Convert a PINHOLE COLMAP SfM model to metric RGB-D, with held-out depth checks."""
import json, struct, shutil
from pathlib import Path
import numpy as np
from replay import asset
from depth_io import load_depth
from recording import atomic
from masks import read_mask,find_mask


def unpack(f,fmt):
    fmt='<'+fmt
    return struct.unpack(fmt,f.read(struct.calcsize(fmt)))


def rotation(q):
    w,x,y,z=np.asarray(q)/np.linalg.norm(q)
    return np.array([[1-2*y*y-2*z*z,2*x*y-2*w*z,2*x*z+2*w*y],
                     [2*x*y+2*w*z,1-2*x*x-2*z*z,2*y*z-2*w*x],
                     [2*x*z-2*w*y,2*y*z+2*w*x,1-2*x*x-2*y*y]])


def read_model(folder):
    folder=Path(folder);cameras={};images=[];points={}
    with (folder/'cameras.bin').open('rb') as f:
        for _ in range(unpack(f,'Q')[0]):
            key,model,w,h=unpack(f,'iiQQ')
            if model!=1:raise ValueError('需要 PINHOLE 模型及已去畸变图像')
            fx,fy,cx,cy=unpack(f,'dddd');cameras[key]=dict(K=[[fx,0,cx],[0,fy,cy],[0,0,1]],width=w,height=h)
    with (folder/'images.bin').open('rb') as f:
        for _ in range(unpack(f,'Q')[0]):
            row=unpack(f,'idddddddi');name=bytearray()
            while True:
                b=f.read(1)
                if not b:raise ValueError('损坏的 COLMAP 图像记录')
                if b==b'\0':break
                name.extend(b)
            obs=[unpack(f,'ddq') for _ in range(unpack(f,'Q')[0])]
            T=np.eye(4);T[:3,:3]=rotation(row[1:5]);T[:3,3]=row[5:8]
            images.append(dict(id=row[0],name=name.decode(),camera_id=row[8],w2c=T,observations=obs))
    with (folder/'points3D.bin').open('rb') as f:
        for _ in range(unpack(f,'Q')[0]):
            row=unpack(f,'QdddBBBd');points[row[0]]=dict(xyz=np.array(row[1:4]),error=row[7])
            f.seek(8*unpack(f,'Q')[0],1)
    return cameras,images,points


def depth_statistics(observed,predicted,scale):
    observed=np.asarray(observed);predicted=np.asarray(predicted)*scale
    if not len(observed):return dict(samples=0)
    error=np.abs(observed-predicted);relative=error/observed
    return dict(samples=len(observed),median_error_m=float(np.median(error)),p90_error_m=float(np.percentile(error,90)),
                median_relative_error=float(np.median(relative)),p90_relative_error=float(np.percentile(relative,90)))


def convert(source,package,model,target,mask_folder=None):
    source=Path(source);target=Path(target);target.mkdir(parents=True,exist_ok=False)
    m=json.loads((source/'manifest.json').read_text(encoding='utf-8'));spec=json.loads((Path(package)/'package.json').read_text(encoding='utf-8'))
    names={f['name']:f for f in spec['frames']};original={f['id']:f for f in m['frames']}
    cameras,images,points=read_model(model);rows=[];train_scales=[];train_obs=[];train_pred=[];eval_obs=[];eval_pred=[];per_frame=[]
    for image in sorted(images,key=lambda v:v['name']):
        entry=names[image['name']];frame=original[entry['frame_id']];camera=cameras[image['camera_id']]
        depth=load_depth(asset(source,frame['depth']),frame);K=np.asarray(camera['K']);oldK=np.asarray(frame['K'])
        if depth.shape!=(camera['height'],camera['width']):raise ValueError('图像和深度分辨率不一致')
        if max(abs(K[0,0]/oldK[0,0]-1),abs(K[1,1]/oldK[1,1]-1))>.1 or np.linalg.norm(K[:2,2]-oldK[:2,2])>10:
            raise ValueError('SfM 内参与已验证相机标定偏差过大')
        mask_path=find_mask(mask_folder,image['name']) if mask_folder else (asset(source,frame['mask']) if frame.get('mask') else None)
        keep=read_mask(mask_path,depth.shape) if mask_path else np.ones(depth.shape,bool)
        observed=[];predicted=[];T=image['w2c']
        for x,y,key in image['observations']:
            if key not in points or points[key]['error']>2:continue
            ix,iy=int(round(x)),int(round(y))
            if not (1<=ix<depth.shape[1]-1 and 1<=iy<depth.shape[0]-1):continue
            if not keep[iy-1:iy+2,ix-1:ix+2].all():continue
            patch=depth[iy-1:iy+2,ix-1:ix+2];z=float(depth[iy,ix]);q=T[:3,:3]@points[key]['xyz']+T[:3,3]
            if q[2]<=0 or not np.isfinite(patch).all() or not .25<z<6 or patch.min()<=.25 or patch.max()-patch.min()>.08+.02*z:continue
            projected=K@q;uv=projected[:2]/projected[2]
            if np.linalg.norm(uv-[x,y])>2:continue
            observed.append(z);predicted.append(q[2])
        is_train=entry['split']=='train'
        if is_train and len(observed)>=20:train_scales.append(float(np.median(np.asarray(observed)/predicted)))
        (train_obs if is_train else eval_obs).extend(observed);(train_pred if is_train else eval_pred).extend(predicted)
        per_frame.append(dict(frame_id=frame['id'],split=entry['split'],samples=len(observed),scale_median=float(np.median(np.asarray(observed)/predicted)) if observed else None))
        rows.append((frame,entry,np.linalg.inv(T),K,mask_path))
    if len(train_scales)<5:raise ValueError('至少需要 5 个训练帧提供可靠的尺度观测')
    scale=float(np.median(train_scales));train_stats=depth_statistics(train_obs,train_pred,scale);eval_stats=depth_statistics(eval_obs,eval_pred,scale)
    spread=float(np.median(np.abs(np.asarray(train_scales)/scale-1)))
    expected_eval={v['frame_id'] for v in spec['frames'] if v['split']=='eval'}
    actual_eval={f['id'] for f,e,_,_,_ in rows if e['split']=='eval'}
    reasons=[]
    if len(rows)/len(names)<.8:reasons.append('SfM 注册率低于 80%')
    if actual_eval!=expected_eval:reasons.append('缺少对比所需的留出视角')
    if spread>.1:reasons.append('不同训练帧的尺度估计不一致')
    for label,stats in [('训练',train_stats),('留出',eval_stats)]:
        if stats['samples']<100 or stats.get('median_relative_error',1)>.1 or stats.get('median_error_m',99)>.15 or stats.get('p90_relative_error',1)>.25:
            reasons.append(label+'深度几何检查未通过')
    report=dict(masked=any(row[4] is not None for row in rows),accepted=not reasons,reasons=reasons,registered=len(rows),input_images=len(names),scale_m_per_sfm_unit=scale,
                training_scale_frames=len(train_scales),scale_frame_median_relative_deviation=spread,train_depth=train_stats,heldout_depth=eval_stats,
                per_frame=per_frame,cameras=cameras,scope='局部米制 SfM；非旧地图对齐',thresholds=dict(min_registration=.8,max_scale_spread=.1,max_median_relative_depth_error=.1,max_median_depth_error_m=.15,max_p90_relative_depth_error=.25))
    atomic(target/'sfm_rgbd_report.json',report)
    if reasons:raise ValueError('；'.join(reasons))
    # One common similarity transform: preserve SfM geometry, anchor at first camera.
    anchor=rows[0][2];R0=anchor[:3,:3].T;origin=anchor[:3,3]
    frames=[]
    for frame,entry,pose,K,mask_path in rows:
        f=dict(frame);T=np.eye(4);T[:3,:3]=R0@pose[:3,:3];T[:3,3]=scale*R0@(pose[:3,3]-origin)
        for field in ('rgb','depth'):
            src=asset(source,f[field]);shutil.copyfile(src,target/src.name);f[field]=src.name
        if mask_path:
            f['mask']=f['id']+'.mask.png';shutil.copyfile(mask_path,target/f['mask'])
        f.update(K=K.tolist(),pose_c2w=T.tolist(),pose_valid=True,reconstruction_split=entry['split']);frames.append(f)
    shutil.copyfile(asset(source,m['map_image']),target/'map.png')
    result={**m,'frames':frames,'map_image':'map.png','pose_status':'LOCAL_CAMERA_TRAJECTORY','world_convention':'camera_local',
            'map_alignment':dict(accepted=False,reason='SfM 米制局部坐标尚未对齐旧地图'),'sfm_rgbd_report':report,
            'input_trajectory_report':m.get('trajectory_report'),'name':'Spirula SfM + RGB-D 米制局部轨迹'}
    result.pop('trajectory_report',None)
    atomic(target/'manifest.json',result)
    return result
