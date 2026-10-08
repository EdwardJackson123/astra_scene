"""Training images from original frames, with pixel-centre-correct intrinsics."""
import json
from pathlib import Path
import cv2
import numpy as np

def resize_geometry(K, source_shape, crop, target_shape):
    y0,y1,x0,x1=crop; h,w=target_shape
    sx=w/(x1-x0);sy=h/(y1-y0)
    A=np.array([[sx,0,sx*(.5-x0)-.5],[0,sy,sy*(.5-y0)-.5],[0,0,1.]])
    return A@K

def training_views(job, data, width=1024):
    report=json.loads((Path(job)/"report.json").read_text())
    capture=json.loads(Path(report["capture"]).read_text())
    out=[]
    for i,frame in enumerate(capture["frames"]):
        raw=cv2.imread(frame["image"])
        if raw is None:raise ValueError("source_rgb_missing")
        oh,ow=raw.shape[:2];sh,sw=data["depths"][i].shape
        if (sh,sw)!=(518,518):raise ValueError("unsupported_geometry_preprocessing")
        rh=round(oh*(518/ow)/14)*14
        if rh>518:raise ValueError("portrait_training_requires_original_crop_mapping")
        top=(518-rh)//2
        w=min(width,ow);h=round(oh*w/ow)
        rgb=cv2.cvtColor(cv2.resize(raw,(w,h),interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2RGB).astype(np.float32)/255
        depth=cv2.resize(data["depths"][i,top:top+rh],(w,h),interpolation=cv2.INTER_NEAREST)
        confidence=cv2.resize(data["confidence"][i,top:top+rh].astype(np.uint8),(w,h),interpolation=cv2.INTER_NEAREST).astype(bool)
        depth[~confidence]=0
        K=resize_geometry(data["intrinsics"][i],(sh,sw),(top,top+rh,0,sw),(h,w))
        out.append(dict(rgb=rgb,depth=depth,confidence=confidence,K=K))
    return out
