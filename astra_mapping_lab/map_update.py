"""Conservative, ROI-bounded weights for an independently versioned map candidate."""
import numpy as np


def fit_similarity(a,b):
    a=np.asarray(a,float);b=np.asarray(b,float)
    if a.shape!=b.shape or a.ndim!=2 or a.shape[1]!=3 or len(a)<3 or not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('无效的三维对应点')
    ac=a-a.mean(0);bc=b-b.mean(0)
    if np.linalg.svd(ac,compute_uv=False)[1]<.01:raise ValueError('对应点共线或基线不足')
    u,s,vt=np.linalg.svd(ac.T@bc);d=np.eye(3);d[2,2]=np.linalg.det(vt.T@u.T)
    R=vt.T@d@u.T;scale=float(np.sum(s*np.diag(d))/np.sum(ac*ac))
    if scale<=0 or not np.isfinite(scale):raise ValueError('无效尺度')
    return scale,R,b.mean(0)-scale*R@a.mean(0)


def roi_weights(xyz, metadata, roi, band_m=.35):
    """Fade inside the selected rectangle; no support extends outside it."""
    if not np.isfinite(band_m) or band_m<=0:raise ValueError('过渡带必须为正数')
    p=np.asarray(xyz,float);r=np.asarray(roi,float);res=float(metadata['resolution'])
    if p.ndim!=2 or p.shape[1]!=3 or r.shape!=(4,) or not np.isfinite(p).all() or not np.isfinite(r).all() or r[2]<=r[0] or r[3]<=r[1]:raise ValueError('更新区域或点坐标无效')
    yaw=float(metadata['origin'][2]);c,s=np.cos(yaw),np.sin(yaw)
    local=(p[:,:2]-metadata['origin'][:2])@np.array([[c,-s],[s,c]])
    x=local[:,0]/res;y=float(metadata['height'])-local[:,1]/res
    distance=np.minimum.reduce([x-r[0],r[2]-x,y-r[1],r[3]-y])*res
    t=np.clip(distance/band_m,0,1)
    return t*t*(3-2*t)


def fresh_weights(surface,free,region,min_surface=3):
    surface=np.asarray(surface);free=np.asarray(free);region=np.asarray(region)
    # A surface repeatedly contradicted by free-space observations is not inserted.
    supported=(surface>=min_surface)&(free<=1)
    return np.where(supported,region*np.clip((surface-min_surface+1)/3.,0,1),0.)


def old_weights(surface,free,near_supported_fresh,region,min_surface=3,min_clear=4):
    """Return removal weight. Unknown/occluded space has no deletion evidence."""
    surface=np.asarray(surface);free=np.asarray(free)
    replace=(surface>=min_surface)&np.asarray(near_supported_fresh,bool)&(free<=1)
    clear=(free>=min_clear)&(surface==0)
    strength=np.where(replace,np.clip((surface-min_surface+1)/3.,0,1),0.)
    return np.asarray(region)*np.where(clear,1.,strength)


def weighted_opacity(logits,weight):
    p=1/(1+np.exp(-np.clip(np.asarray(logits,float),-40,40)))
    p=np.clip(p*np.asarray(weight),1e-7,1-1e-7)
    return np.log(p/(1-p))
