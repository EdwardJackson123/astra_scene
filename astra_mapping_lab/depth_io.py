"""Explicit on-disk depth encodings; algorithms always receive float32 metres."""
import os
from pathlib import Path
import numpy as np
from PIL import Image

PNG='png_u16_mm'
NPY='npy_f32_m'

def quantize_mm(depth):
    depth=np.asarray(depth)
    if depth.ndim!=2 or depth.size>4_000_000:raise ValueError('深度尺寸无效')
    valid=np.isfinite(depth)&(depth>0)
    mm=np.zeros(depth.shape,np.uint16)
    values=np.rint(depth[valid].astype(np.float64)*1000)
    if np.any((values<1)|(values>65535)):raise ValueError('深度超出毫米 uint16 可表示范围，拒绝截断')
    mm[valid]=values.astype(np.uint16)
    return mm

def save_depth(root,name,depth,encoding=PNG):
    root=Path(root)
    if encoding==PNG:
        path=root/(name+'.depth.png');temp=path.with_suffix('.tmp')
        Image.fromarray(quantize_mm(depth)).save(temp,format='PNG',compress_level=1)
        os.replace(temp,path)
        return dict(depth=path.name,depth_encoding=PNG,depth_scale_m=.001,depth_invalid_value=0)
    if encoding==NPY:
        path=root/(name+'.npy');np.save(path,np.asarray(depth,np.float32))
        return dict(depth=path.name,depth_encoding=NPY,depth_scale_m=1.)
    raise ValueError('不支持的深度存储格式')

def frame_encoding(frame):
    encoding=frame.get('depth_encoding',NPY)
    if encoding==PNG:
        if frame.get('depth_scale_m')!=.001 or frame.get('depth_invalid_value')!=0:raise ValueError('PNG 深度缺少毫米比例或无效值声明')
    elif encoding==NPY:
        if frame.get('depth_scale_m',1.)!=1.:raise ValueError('旧 NPY 深度必须为米')
    else:raise ValueError('未知深度存储格式')
    return encoding

def validate_depth_manifest(manifest):
    unit=manifest.get('depth_unit')
    if unit not in ('m','mm'):raise ValueError('未声明有效深度存储单位')
    frames=manifest.get('frames',[])
    if not isinstance(frames,list):raise ValueError('深度记录帧列表无效')
    for frame in frames:
        encoding=frame_encoding(frame)
        if (unit=='mm')!=(encoding==PNG):raise ValueError('深度单位与文件编码不一致')

def load_depth(path,frame):
    path=Path(path);encoding=frame_encoding(frame)
    if encoding==PNG:
        if path.suffix.lower()!='.png':raise ValueError('毫米 PNG 文件后缀错误')
        with Image.open(path) as image:
            if image.format!='PNG' or image.width*image.height>4_000_000 or image.mode not in ('I','I;16','I;16L','I;16B'):raise ValueError('需要 16 位单通道深度 PNG')
            values=np.asarray(image)
            if values.ndim!=2 or values.min()<0 or values.max()>65535:raise ValueError('PNG 深度值无效')
            # Divide, rather than multiply by float32(0.001), matching the source.
            depth=values.astype(np.float32)/1000.
    else:
        if path.suffix.lower()!='.npy':raise ValueError('旧深度必须是浮点米 NPY')
        depth=np.load(path,allow_pickle=False)
        if depth.dtype.kind!='f':raise ValueError('NPY 必须存储浮点米，不能默认为毫米')
        depth=np.asarray(depth,np.float32)
    if depth.ndim!=2 or depth.size>4_000_000:raise ValueError('深度尺寸无效')
    return depth
