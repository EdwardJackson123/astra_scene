"""Deterministic textured plane, solely for no-extrinsic workflow regression."""
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from core import normalize_metadata
from recording import atomic


def create_tracking_demo(raw_root,reference_root):
    raw_root=Path(raw_root);reference_root=Path(reference_root)
    raw_root.mkdir(parents=True,exist_ok=False);reference_root.mkdir(parents=True,exist_ok=False)
    rng=np.random.default_rng(20260923);texture=Image.fromarray(rng.integers(30,225,(240,420,3),dtype=np.uint8));draw=ImageDraw.Draw(texture)
    for _ in range(280):
        x,y=rng.integers(5,400),rng.integers(5,220);size=int(rng.integers(3,16));color=tuple(int(x) for x in rng.integers(20,235,3));draw.rectangle((x,y,x+size,y+size),fill=color)
    grid=Image.new('L',(100,100),255);ImageDraw.Draw(grid).rectangle((0,0,99,99),outline=0,width=2)
    meta=normalize_metadata(dict(resolution=.1,origin=[-5,-2,0]),100,100);K=[[250.,0,160],[0,250.,120],[0,0,1.]]
    T=np.eye(4);T[:3,:3]=[[1,0,0],[0,0,1],[0,-1,0]];T[:3,3]=[0,0,1]
    frames=[];reference=[]
    for i in range(8):
        rgb=texture.crop((i*10,0,i*10+320,240));name=f'{i:06d}';depth=np.full((240,320),2,np.float32)
        for target in [raw_root,reference_root]:rgb.save(target/(name+'.png'));np.save(target/(name+'.npy'),depth)
        f=dict(id=name,timestamp_s=1+i*.5,rgb=name+'.png',depth=name+'.npy',K=K);frames.append(f)
        pose=T.copy();pose[0,3]+=.08*i;reference.append(dict(f,pose_c2w=pose.tolist(),pose_valid=True))
    common=dict(schema_version=1,synthetic=True,camera_convention='opencv',depth_unit='m',depth_aligned_to='color',depth_rectified=True,
                map_image='map.png',map_metadata=meta,suggested_roi=[48,57,52,63],source_map={'kind':'synthetic'})
    raw=dict(common,kind='raw_rgbd',name='合成连续 RGB-D（无外参）',frames=frames,pose_status='NOT_ESTIMATED',map_alignment={'accepted':False})
    ref=dict(common,name='合成旧地图参考',world_convention='xy_ground_z_up',ground_z_m=0.,frames=reference,pose_status='MAP_ALIGNED')
    grid.save(raw_root/'map.png');grid.save(reference_root/'map.png');atomic(raw_root/'capture_manifest.json',raw);atomic(reference_root/'manifest.json',ref)
    return raw,ref
