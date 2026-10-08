"""Continuous small-baseline novel-view review; synthetic views have no ground truth."""
import json,subprocess
from pathlib import Path
import cv2,numpy as np
from .quality import target_mask
from .postprocess import load,rows_from_data
from .roi import box_geometry

def look_at(position,target):
    forward=target-position;forward/=np.linalg.norm(forward)
    right=np.cross(forward,[0,0,1.])
    if np.linalg.norm(right)<1e-6:raise ValueError('vertical_review_camera')
    right/=np.linalg.norm(right);down=np.cross(forward,right)
    pose=np.eye(4);pose[:3,:3]=np.column_stack([right,down,forward]);pose[:3,3]=position
    return pose

def review(run,before,after,out):
    from vps.refinement.gaussian_renderer import GsplatGaussianRenderer,GaussianCamera
    cfg,d,split,_=load(run);rows=rows_from_data(d)
    scope=json.loads((run/'manual_scope.json').read_text())
    regions=scope.get('regions',[scope]);out.mkdir(parents=True,exist_ok=True)
    renderer=GsplatGaussianRenderer('cuda');poses=[];segments=[]
    video=out/'continuous_review.mp4'
    cmd=['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24',
         '-s','1280x640','-r','12','-i','-','-an','-c:v','libx264','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(video)]
    with (out/'review_encoder.log').open('w') as log:
        proc=subprocess.Popen(cmd,stdin=subprocess.PIPE,stderr=log)
        try:
            for region_id,region in enumerate(regions):
                center,size,_=box_geometry(region)
                ids=sorted(split['test_views'],key=lambda i:target_mask(rows[i],region).sum(),reverse=True)[:2]
                for i in ids:
                    original=d['poses'][i];right=original[:3,0]
                    distance=np.linalg.norm(original[:3,3]-center)
                    focal=320*distance/(max(size)*.75)
                    camera=GaussianCamera(640,640,focal,focal,320,320)
                    segments.append(dict(region=region_id,source_view=i,start_frame=len(poses),frames=48))
                    for step in range(48):
                        offset=.15*np.sin(2*np.pi*step/48)
                        position=original[:3,3]+offset*right
                        pose=look_at(position,center);ims=[]
                        for label,path in [('previous',before),('candidate',after)]:
                            rgb=renderer.render(path,pose,camera).rgb.copy()
                            cv2.putText(rgb,label,(15,28),cv2.FONT_HERSHEY_SIMPLEX,.8,(255,50,30),2,cv2.LINE_AA)
                            cv2.putText(rgb,'synthetic nearby view / review only',(15,620),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,80,40),1,cv2.LINE_AA)
                            ims.append(rgb)
                        frame=np.concatenate(ims,axis=1)
                        proc.stdin.write(frame.tobytes())
                        poses.append(pose.tolist())
                        if step==12:cv2.imwrite(str(out/f'review_region_{region_id}_view_{i}.jpg'),cv2.cvtColor(frame,cv2.COLOR_RGB2BGR))
            proc.stdin.close()
            if proc.wait(timeout=60):raise ValueError('review_video_encode_failed')
        except BaseException:
            proc.kill();proc.wait();raise
    (out/'continuous_review.json').write_text(json.dumps(dict(camera_poses=poses,segments=segments,
        horizontal_offset_m=.15,reference_available=False,automatic_acceptance=False,
        limitation='small_baseline_visual_review_not_unseen_backside_validation'),indent=2))
    return video
