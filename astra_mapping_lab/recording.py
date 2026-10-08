"""Mode-gated recording session and quality selection independent of ROS."""
import json,math,os,shutil,time
from pathlib import Path
import numpy as np
from PIL import Image
from depth_io import save_depth,PNG,NPY


def atomic(path,value):
    path=Path(path);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,allow_nan=False,indent=2),encoding='utf-8');os.replace(tmp,path)


def validate_calibration(config):
    if config.get('verified') is not True:raise ValueError('请核实并确认深度对齐、相机外参与地图地面高度')
    T=np.asarray(config.get('camera_to_base'),float)
    if T.shape!=(4,4) or not np.isfinite(T).all() or not np.allclose(T[3],[0,0,0,1]) or not np.allclose(T[:3,:3].T@T[:3,:3],np.eye(3),atol=1e-5) or not np.isclose(np.linalg.det(T[:3,:3]),1,atol=1e-5):raise ValueError('camera_to_base 必须为有效刚体变换')
    if config.get('depth_aligned_to_color') is not True:raise ValueError('深度必须已对齐到彩色相机')
    ground=float(config.get('ground_z_m'))
    if not math.isfinite(ground):raise ValueError('地面高度无效')
    for k in ['rgb_topic','depth_topic','info_topic','pose_topic']:
        v=config.get(k)
        if not isinstance(v,str) or not v.startswith('/') or len(v)>200:raise ValueError('ROS topic 配置无效')
    return {**config,'camera_to_base':T.tolist(),'ground_z_m':ground}


def validate_capture_config(config):
    if config.get('pose_source')!='rgbd_odometry':raise ValueError('此流程使用 RGB-D 独立相机轨迹')
    source=config.get('intrinsics_source','camera_info')
    if source not in ('camera_info','static'):raise ValueError('内参来源无效')
    keys=['rgb_topic','depth_topic']+(['info_topic'] if source=='camera_info' else [])
    for key in keys:
        value=config.get(key)
        if not isinstance(value,str) or not value.startswith('/') or len(value)>200:raise ValueError('相机话题配置无效')
    result={key:config[key] for key in ['pose_source']+keys}
    result['intrinsics_source']=source
    storage=config.get('depth_storage',PNG)
    if storage not in (PNG,NPY):raise ValueError('深度存储格式无效')
    result['depth_storage']=storage
    hz=config.get('capture_hz',5)
    if isinstance(hz,bool) or not isinstance(hz,(int,float)) or not math.isfinite(hz) or not 1<=hz<=10:raise ValueError('capture_hz 必须在 1–10 Hz 之间')
    result['capture_hz']=float(hz)
    for key in ['max_frames','max_duration_s']:
        value=config.get(key,0)
        if type(value) is not int or value<0:raise ValueError(key+' 必须是非负整数，0 表示不限制')
        result[key]=value
    if source=='static':
        cal=config.get('intrinsics',{})
        K=np.asarray(cal.get('K'),float);D=np.asarray(cal.get('D'),float)
        if K.shape!=(3,3) or not np.isfinite(K).all() or K[0,0]<=0 or K[1,1]<=0 or not np.allclose(K[2],[0,0,1]):raise ValueError('静态内参 K 无效')
        pending=cal.get('D') is None and cal.get('verified') is not True
        if not pending and (D.ndim!=1 or len(D) not in (4,5,8) or not np.isfinite(D).all()):raise ValueError('静态畸变参数 D 无效')
        if cal.get('distortion_model') not in ('plumb_bob','rational_polynomial','inverse_brown_conrady'):raise ValueError('静态畸变模型无效')
        if cal.get('distortion_model')=='inverse_brown_conrady' and not pending and len(D)!=5:raise ValueError('RealSense 畸变模型需要 5 个系数')
        for key in ['width','height']:
            if type(cal.get(key)) is not int or not 1<=cal[key]<=4096:raise ValueError('内参分辨率无效')
        for key in ['serial','rgb_frame_id','depth_frame_id','provenance']:
            if not isinstance(cal.get(key),str) or not cal[key].strip():raise ValueError('静态内参缺少 '+key)
        if cal.get('depth_aligned_to_color') is not True:raise ValueError('必须明确深度已经对齐彩色')
        if cal['rgb_frame_id']!='camera_'+cal['serial'] or cal['depth_frame_id']!='camera_'+cal['serial']+'_depth':raise ValueError('静态适配器仅支持已核查的机器人相机帧命名')
        result['intrinsics']={**cal,'K':K.tolist(),'D':None if pending else D.tolist(),'verified':cal.get('verified') is True}
    return result


def require_capture_ready(config):
    cfg=validate_capture_config(config)
    if cfg['intrinsics_source']=='static' and not cfg['intrinsics']['verified']:
        if cfg['intrinsics']['D'] is None:raise ValueError('内参 K 已配置，畸变参数 D 待补充；正式录制尚未就绪')
        raise ValueError('相机内参尚未核实：需确认当前序列号、分辨率、彩色内参与畸变参数，不能直接使用未确认的 nav 内参')
    return cfg


def static_camera_info(config,rgb,depth):
    """Accept only the audited aligned-depth alias of this exact camera."""
    from types import SimpleNamespace
    cal=require_capture_ready(config)['intrinsics']
    if (rgb.width,rgb.height)!=(cal['width'],cal['height']) or (depth.width,depth.height)!=(cal['width'],cal['height']):raise ValueError('图像尺寸与静态内参不匹配')
    if rgb.header.frame_id!=cal['rgb_frame_id'] or depth.header.frame_id!=cal['depth_frame_id']:raise ValueError('相机序列号或帧标识不匹配')
    if (rgb.header.stamp.sec,rgb.header.stamp.nanosec)!=(depth.header.stamp.sec,depth.header.stamp.nanosec):raise ValueError('机器人 RGB-D 时间戳不一致')
    return SimpleNamespace(width=cal['width'],height=cal['height'],k=np.asarray(cal['K']).ravel(),d=cal['D'],distortion_model=cal['distortion_model'])


def rectification_maps(K,D,model,size):
    """Map ideal output pixels to source pixels; RGB and aligned Z share maps.

    RealSense inverse Brown uses the SDK rs2_project_point_to_pixel convention,
    which differs from OpenCV's tangential term. Do not relabel it plumb_bob.
    """
    import cv2
    if model in ('plumb_bob','rational_polynomial'):
        return cv2.initUndistortRectifyMap(K,D,None,K,size,cv2.CV_32FC1)
    if model!='inverse_brown_conrady' or len(D)!=5:raise ValueError('不支持的畸变模型')
    u,v=np.meshgrid(np.arange(size[0],dtype=np.float64),np.arange(size[1],dtype=np.float64))
    x=(u-K[0,2])/K[0,0];y=(v-K[1,2])/K[1,1];r2=x*x+y*y
    radial=1+D[0]*r2+D[1]*r2*r2+D[4]*r2*r2*r2
    x=x*radial;y=y*radial
    xd=x+2*D[2]*x*y+D[3]*(r2+2*x*x)
    yd=y+2*D[3]*x*y+D[2]*(r2+2*y*y)
    return (xd*K[0,0]+K[0,2]).astype(np.float32),(yd*K[1,1]+K[1,2]).astype(np.float32)


class RawRecorder:
    """Save rectified RGB-D before any pose or map alignment exists."""
    def __init__(self,root,record,map_image,roi,config):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=False);self.config=validate_capture_config(config)
        shutil.copyfile(map_image,self.root/'map.png');self.frames=[];self.counts={};self.last_time=None
        self.started=time.monotonic();self.state='WAITING_INPUT';self.reason='等待同步 RGB、对齐深度和内参，不需要底盘外参或定位';self.last_saved_at=None;self.last_depth_fraction=None
        self.manifest=dict(schema_version=1,kind='raw_rgbd',name='机器人原始 RGB-D',synthetic=False,
            camera_convention='opencv',depth_unit='mm' if self.config['depth_storage']==PNG else 'm',depth_aligned_to='color',depth_rectified=True,
            map_image='map.png',map_metadata=record['metadata'],source_map=record['source'],suggested_roi=roi,
            configuration=self.config,pose_status='NOT_ESTIMATED',map_alignment={'accepted':False},frames=self.frames)
        atomic(self.root/'capture_manifest.json',self.manifest);self.save_status()

    def save_status(self):
        elapsed=time.monotonic()-self.started
        atomic(self.root/'capture_status.json',dict(state=self.state,reason=self.reason,frames=len(self.frames),rejected=self.counts,elapsed_s=elapsed,updated_at=time.time(),last_saved_at=self.last_saved_at,last_depth_fraction=self.last_depth_fraction,capture_hz=self.config['capture_hz'],saved_fps=len(self.frames)/max(elapsed,1)))

    def reject(self,reason):
        self.reason=reason;self.counts[reason]=self.counts.get(reason,0)+1;self.save_status();return False

    def add(self,rgb,depth,K,stamp):
        if self.state in ('STOPPED','FAILED'):return False
        limit=self.config['max_frames']
        if limit and len(self.frames)>=limit:self.stop('达到配置的帧数上限');return False
        rgb=np.asarray(rgb);depth=np.asarray(depth,dtype=np.float32);K=np.asarray(K,float)
        if depth.ndim!=2 or rgb.shape!=(*depth.shape,3) or depth.size>4_000_000:return self.reject('RGB-D 尺寸不一致或过大')
        if K.shape!=(3,3) or not np.isfinite(K).all() or K[0,0]<=0 or K[1,1]<=0 or not np.allclose(K[2],[0,0,1]):return self.reject('相机内参无效')
        if not math.isfinite(stamp) or stamp<=0 or (self.last_time is not None and stamp<=self.last_time):return self.reject('图像时间戳未递增')
        if self.last_time is not None and stamp-self.last_time<.9/self.config['capture_hz']:return False
        valid=np.isfinite(depth)&(depth>.25)&(depth<6)
        if valid.mean()<.15:return self.reject('有效深度不足 15%')
        gray=rgb.astype(float).mean(-1)
        if np.mean((gray<3)|(gray>252))>.8:return self.reject('图像严重欠曝或过曝')
        if shutil.disk_usage(self.root).free<512*1024**2:self.stop('磁盘剩余不足 512 MB');return False
        name=f'{len(self.frames):06d}'
        try:stored=save_depth(self.root,name,depth,self.config['depth_storage'])
        except ValueError as e:return self.reject(str(e))
        Image.fromarray(rgb.astype(np.uint8)).save(self.root/(name+'.jpg'),quality=95)
        self.frames.append(dict(id=name,timestamp_s=float(stamp),rgb=name+'.jpg',**stored,K=K.tolist(),valid_depth_fraction=float(valid.mean())))
        self.last_time=stamp;self.last_saved_at=time.time();self.last_depth_fraction=float(valid.mean());self.state='RECORDING';self.reason='已保存原始 RGB-D；位姿将在停止后估计'
        atomic(self.root/'capture_manifest.json',self.manifest);self.save_status();return True

    def stop(self,reason='采集已停止'):
        self.state='STOPPED';self.reason=reason;atomic(self.root/'capture_manifest.json',self.manifest);self.save_status()


class Recorder:
    def __init__(self,root,record,map_image,roi,config):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=False);self.config=validate_calibration(config)
        shutil.copyfile(map_image,self.root/'map.png');self.last_pose=None;self.last_time=None;self.frames=[]
        self.counts={};self.started=time.monotonic();self.state='WAITING_INPUT';self.reason='等待同步 RGB、深度、内参与地图位姿'
        self.manifest=dict(schema_version=1,name='机器人 RGB-D 采集',synthetic=False,world_convention='xy_ground_z_up',camera_convention='opencv',
            depth_unit='m',depth_aligned_to='color',depth_rectified=True,ground_z_m=self.config['ground_z_m'],map_image='map.png',map_metadata=record['metadata'],
            source_map=record['source'],suggested_roi=roi,calibration=self.config,frames=self.frames)
        atomic(self.root/'manifest.json',self.manifest);self.save_status()

    def save_status(self):
        atomic(self.root/'capture_status.json',dict(state=self.state,reason=self.reason,frames=len(self.frames),rejected=self.counts,elapsed_s=time.monotonic()-self.started,updated_at=time.time()))

    def reject(self,reason):
        self.reason=reason;self.counts[reason]=self.counts.get(reason,0)+1;self.save_status();return False

    def add(self,rgb,depth,K,pose,stamp):
        if self.state in ('STOPPED','FAILED'):return False
        if len(self.frames)>=200:self.state='STOPPED';self.reason='已达到 200 帧上限';self.save_status();return False
        rgb=np.asarray(rgb);depth=np.asarray(depth,dtype=np.float32);K=np.asarray(K,float);pose=np.asarray(pose,float)
        if depth.ndim!=2 or rgb.shape!=(*depth.shape,3) or depth.size>4_000_000:return self.reject('RGB-D 尺寸不一致或过大')
        if K.shape!=(3,3) or not np.isfinite(K).all() or K[0,0]<=0 or K[1,1]<=0:return self.reject('相机内参无效')
        if pose.shape!=(4,4) or not np.isfinite(pose).all() or not np.allclose(pose[3],[0,0,0,1]) or not np.allclose(pose[:3,:3].T@pose[:3,:3],np.eye(3),atol=1e-5):return self.reject('相机位姿无效')
        if not math.isfinite(stamp) or (self.last_time is not None and stamp<=self.last_time):return self.reject('时间戳未递增')
        valid=np.isfinite(depth)&(depth>.25)&(depth<6)
        if valid.mean()<.15:return self.reject('有效深度不足 15%')
        gray=rgb.astype(float).mean(-1)
        if np.mean((gray<3)|(gray>252))>.8:return self.reject('图像严重欠曝或过曝')
        if self.last_pose is not None:
            angle=math.acos(float(np.clip((np.trace(self.last_pose[:3,:3].T@pose[:3,:3])-1)/2,-1,1)))
            if stamp-self.last_time<.5:return self.reject('关键帧间隔不足')
            if np.linalg.norm(pose[:3,3]-self.last_pose[:3,3])<.12 and angle<math.radians(10):return self.reject('视角变化不足')
        i=len(self.frames);name=f'{i:06d}';Image.fromarray(rgb.astype(np.uint8)).save(self.root/(name+'.jpg'),quality=95);np.save(self.root/(name+'.npy'),depth)
        self.frames.append(dict(id=name,timestamp_s=float(stamp),rgb=name+'.jpg',depth=name+'.npy',K=K.tolist(),pose_c2w=pose.tolist(),pose_valid=True,valid_depth_fraction=float(valid.mean())))
        self.last_pose=pose.copy();self.last_time=stamp;self.state='RECORDING';self.reason='关键帧已保存';atomic(self.root/'manifest.json',self.manifest);self.save_status();return True

    def stop(self,reason='用户停止采集'):
        self.state='STOPPED';self.reason=reason;atomic(self.root/'manifest.json',self.manifest);self.save_status()
