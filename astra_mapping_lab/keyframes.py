"""Select training views AFTER tracking; hold out separate views across the trajectory."""
import math
import numpy as np
from PIL import Image
from replay import asset
from depth_io import load_depth


def select(root, frames):
    if any('reconstruction_split' in f for f in frames):
        if not all(f.get('reconstruction_split') in ('train','eval') for f in frames):raise ValueError('训练/留出划分不完整')
        train=sum(f['reconstruction_split']=='train' for f in frames);ev=len(frames)-train
        if train<3 or ev<2:raise ValueError('固定训练/留出视角不足')
        return [(f,f['reconstruction_split']) for f in frames],dict(method='preserved_comparison_split',input_frames=len(frames),train_frames=train,eval_frames=ev)

    if len(frames)<20:
        return [(f,'eval' if (i+1)%5==0 or i==len(frames)-1 else 'train') for i,f in enumerate(frames)],dict(method='short_sequence_all_views',input_frames=len(frames))
    import cv2
    good=[];rejected=[]
    for f in frames:
        gray=np.asarray(Image.open(asset(root,f['rgb'])).convert('L'))
        sharpness=float(cv2.Laplacian(gray,cv2.CV_64F).var())
        depth=load_depth(asset(root,f['depth']),f)
        valid=float(np.mean(np.isfinite(depth)&(depth>.25)&(depth<6)))
        if sharpness<20 or valid<.15:
            rejected.append(dict(frame_id=f['id'],sharpness=sharpness,depth_fraction=valid));continue
        good.append(f)
    if len(good)<8:raise ValueError('清晰且深度有效的帧不足，无法划分训练与留出视角')
    training=[good[0]]
    for f in good[1:]:
        a=np.asarray(training[-1]['pose_c2w']);b=np.asarray(f['pose_c2w'])
        distance=np.linalg.norm(a[:3,3]-b[:3,3])
        angle=math.degrees(math.acos(float(np.clip((np.trace(a[:3,:3].T@b[:3,:3])-1)/2,-1,1))))
        if distance>=.12 or angle>=6 or f['timestamp_s']-training[-1]['timestamp_s']>=1.5:training.append(f)
    if training[-1]['id']!=good[-1]['id']:training.append(good[-1])
    train_ids={f['id'] for f in training}
    # Holdouts are never used for color fitting or depth initialization.
    available=[f for f in good if f['id'] not in train_ids]
    if len(training)<3 or len(available)<2:raise ValueError('独立训练/留出视角不足')
    chosen=sorted(set(np.linspace(0,len(available)-1,min(12,len(available)),dtype=int)))
    eval_ids={available[i]['id'] for i in chosen}
    output=[(f,'train' if f['id'] in train_ids else 'eval') for f in good if f['id'] in train_ids|eval_ids]
    return output,dict(method='motion_time_keyframes_with_quality_filter',input_frames=len(frames),train_frames=len(train_ids),eval_frames=len(eval_ids),
                       redundant_frames=len(good)-len(output),rejected=rejected,thresholds=dict(translation_m=.12,rotation_deg=6,max_interval_s=1.5,min_laplacian_variance=20,min_depth_fraction=.15),
                       retained_frame_ids=[f['id'] for f,_ in output],limitations=['留出视角有独立图像，但位姿来自同一次轨迹估计；不是独立几何真值。'])
