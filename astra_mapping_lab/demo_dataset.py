"""Deterministic ray-cast synthetic fixture. Not a robot reconstruction benchmark."""
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw


def create_demo(root):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    image=Image.new('L',(100,100),255);draw=ImageDraw.Draw(image);draw.rectangle((0,0,99,99),outline=0,width=2);image.save(root/'map.png')
    K=np.array([[120,0,80],[0,120,60],[0,0,1]],float);R=np.array([[1,0,0],[0,0,1],[0,-1,0]],float)
    y,x=np.indices((120,160));rays=np.stack([(x-80)/120,(y-60)/120,np.ones_like(x)],-1)@R.T;frames=[]
    for n,cx in enumerate([-.5,0,.5,.5]):
        origin=np.array([cx,0,1.]);depth=np.full((120,160),5.,float)
        floor=np.divide(-1.,rays[...,2],out=np.full_like(depth,np.inf),where=rays[...,2]<-.001);depth=np.minimum(depth,floor)
        boxes=[([-.6,2.,0],[.6,2.5,1.2]),([-2.1,1.6,0],[-1.5,2.2,1.3]),([1.,2.,.7],[1.05,2.05,.75])]
        if n==0:boxes.append(([.9,3.,0],[1.3,3.4,1.4]))
        for lower,upper in boxes:
            lo=np.full_like(rays,-np.inf);hi=np.full_like(rays,np.inf);miss=np.zeros(depth.shape,bool)
            for axis in range(3):
                direction=rays[...,axis];parallel=np.abs(direction)<1e-8
                miss|=parallel&((origin[axis]<lower[axis])|(origin[axis]>upper[axis]))
                a=np.divide(lower[axis]-origin[axis],direction,out=np.full_like(depth,-np.inf),where=~parallel)
                b=np.divide(upper[axis]-origin[axis],direction,out=np.full_like(depth,np.inf),where=~parallel)
                lo[...,axis]=np.minimum(a,b);hi[...,axis]=np.maximum(a,b)
            enter=lo.max(-1);leave=hi.min(-1);hit=(leave>=np.maximum(enter,0))&~miss&(enter>.1)
            depth[hit]=np.minimum(depth[hit],enter[hit])
        # No depth means no removal evidence.
        depth[:8,:]=0;name=f'depth_{n}.npy';np.save(root/name,depth.astype(np.float32))
        pose=np.eye(4);pose[:3,:3]=R;pose[:3,3]=origin
        color=np.stack([(x*3+n*7)%200+30,(y*4)%200+30,np.clip(depth*35,20,230)],-1).astype(np.uint8);Image.fromarray(color).save(root/f'rgb_{n}.png')
        frames.append(dict(id=f'frame-{n}',timestamp_s=n*.8,rgb=f'rgb_{n}.png',depth=name,K=K.tolist(),pose_c2w=pose.tolist(),pose_valid=True))
    manifest=dict(schema_version=1,name='模拟 RGB-D 回放：新增箱体 / 瞬时物体 / 框外物体',synthetic=True,
        world_convention='xy_ground_z_up',camera_convention='opencv',depth_unit='m',depth_aligned_to='color',depth_rectified=True,
        ground_z_m=0.,map_image='map.png',map_metadata=dict(resolution=.1,origin=[-5,-2,0],negate=0,free_thresh=.196,occupied_thresh=.65),
        suggested_roi=[40,35,65,70],frames=frames)
    (root/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    return manifest
