"""Metric rigid alignment from static RGB-D / original-map render correspondences."""
from gpu_config import gpu_index
import os,sys,json,fcntl,time
from pathlib import Path
ROOT=Path('/home/xujie/astra_scene');LAB=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT),str(ROOT/'vendor/engine'),str(LAB)]
from astra_scene.pipeline import environment
os.environ.update(environment(gpu_index()))
import numpy as np,cv2
from PIL import Image
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares
from trajectory import fit_rigid,pose_error
from masks import frame_mask
from depth_io import load_depth
from astra_scene.io import write_json,sha256
from update_context import settings
CFG=settings();OUT=Path(CFG['alignment'])
SOURCE=Path(CFG['source'])
BASE=Path(CFG['base_map'])
m=json.loads((SOURCE/'manifest.json').read_text());frames={int(f['id']):f for f in m['frames']}
anchors=json.loads((OUT/'anchors/localization.json').read_text())['records']
poses=[np.asarray(a['pose'])@np.linalg.inv(frames[a['frame_index']]['pose_c2w']) for a in anchors if a['success']]
scores=[np.median([pose_error(T,Q)[0]+.04*pose_error(T,Q)[1] for Q in poses]) for T in poses]
coarse=poses[int(np.argmin(scores))];agree=[T for T in poses if pose_error(T,coarse)[0]<.6 and pose_error(T,coarse)[1]<12]
assert len(agree)>=6,('insufficient_coarse_anchors',len(agree))
coarse[:3,:3]=Rotation.from_matrix(np.array(agree)[:,:3,:3]).mean().as_matrix();coarse[:3,3]=np.median(np.array(agree)[:,:3,3],axis=0)
write_json(OUT/'coarse_alignment.json',dict(map_from_local=coarse.tolist(),anchors=len(poses),consensus=len(agree),residuals=[dict(position_m=pose_error(T,coarse)[0],rotation_deg=pose_error(T,coarse)[1]) for T in poses],acceptance='initialization_only'))
print('COARSE',len(agree),coarse.tolist(),flush=True)
cv2.setNumThreads(4);sift=cv2.SIFT_create(nfeatures=4000,contrastThreshold=.02)
def feature(rgb,dep,K,pose,valid):
    valid=valid & np.isfinite(dep)&(dep>.4)&(dep<7)
    valid=cv2.erode(valid.astype('uint8'),np.ones((5,5),'uint8'))
    span=cv2.dilate(dep,np.ones((3,3),'uint8'))-cv2.erode(dep,np.ones((3,3),'uint8'))
    valid &= (span<.10+.015*dep).astype('uint8')
    key,d=sift.detectAndCompute(cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY),valid*255)
    if d is None:return np.zeros((0,3)),np.zeros((0,128),'float32'),np.zeros((0,2))
    uv=np.array([k.pt for k in key]);ix=np.rint(uv).astype(int);ix[:,0]=ix[:,0].clip(0,dep.shape[1]-1);ix[:,1]=ix[:,1].clip(0,dep.shape[0]-1)
    z=dep[ix[:,1],ix[:,0]];q=np.c_[uv,np.ones(len(uv))]@np.linalg.inv(K).T*z[:,None]
    return q@pose[:3,:3].T+pose[:3,3],d,uv
def matches(a,b):
    if min(len(a[1]),len(b[1]))<2:return np.array([],int),np.array([],int)
    matcher=cv2.BFMatcher();f=matcher.knnMatch(a[1],b[1],k=2);back=matcher.knnMatch(b[1],a[1],k=2)
    reverse={v[0].queryIdx:v[0].trainIdx for v in back if len(v)==2 and v[0].distance<.75*v[1].distance}
    pairs=[(v[0].queryIdx,v[0].trainIdx) for v in f if len(v)==2 and v[0].distance<.75*v[1].distance and reverse.get(v[0].trainIdx)==v[0].queryIdx]
    return np.array(pairs,dtype=int).reshape(-1,2).T
with (ROOT/'runs/gpu.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX)
    from vps.refinement.gaussian_renderer import GsplatGaussianRenderer,GaussianCamera
    renderer=GsplatGaussianRenderer('cuda');aa=[];bb=[];ii=[];notes=[]
    (OUT/'alignment_views').mkdir(exist_ok=True)
    import pycolmap,torch
    from scipy.spatial import cKDTree
    refs=Path(CFG['reference_root'])/'train';scene=Path(CFG['reference_scene'])
    model=pycolmap.Reconstruction(scene/'sparse/0');model_images=list(model.images.values())
    model_matrices=np.array([im.cam_from_world().inverse().matrix() for im in model_images]);tree=cKDTree(model_matrices.reshape(-1,12))
    from vps.refinement.image_matcher import SuperPointLightGlueMatcher
    matcher=SuperPointLightGlueMatcher('cuda',max_keypoints=3000)
    refcache={}
    def reference(path):
        name=Path(path).name
        if name not in refcache:
            rp=np.loadtxt(refs/'poses'/(Path(name).stem+'.txt'));distance,index=tree.query(rp[:3].ravel());assert distance<1e-5
            im=model_images[index];cam=model.cameras[im.camera_id];assert cam.model_name=='PINHOLE'
            fx,fy,cx,cy=cam.params;K=np.array([[fx,0,cx],[0,fy,cy],[0,0,1.]])
            rgb=np.array(Image.open(refs/'rgb'/name).convert('RGB'));dep=torch.load(scene/'depths'/(Path(im.name).stem+'.pt'),map_location='cpu',weights_only=True).numpy().astype('float32')
            assert rgb.shape[:2]==dep.shape
            h,w=rgb.shape[:2];rgb=cv2.resize(rgb,(800,800),interpolation=cv2.INTER_AREA);dep=cv2.resize(dep,(800,800),interpolation=cv2.INTER_NEAREST);K[0]*=800/w;K[1]*=800/h
            refcache[name]=(rgb,dep,K,rp)
        return refcache[name]
    for i,anchor in enumerate(anchors):
        if not anchor['success']:continue
        f=frames[anchor['frame_index']];K=np.array(f['K']);local=np.array(f['pose_c2w'])
        rgb=np.array(Image.open(SOURCE/f['rgb']).convert('RGB'));dep=load_depth(SOURCE/f['depth'],f);keep=frame_mask(SOURCE,f,dep.shape)
        byquery={};localpoints={}
        def lift(uv,depth,K,pose,valid):
            xy=np.rint(uv).astype(int);good=(xy[:,0]>2)&(xy[:,0]<depth.shape[1]-3)&(xy[:,1]>2)&(xy[:,1]<depth.shape[0]-3)
            xy[:,0]=xy[:,0].clip(0,depth.shape[1]-1);xy[:,1]=xy[:,1].clip(0,depth.shape[0]-1)
            clean=cv2.erode((valid & (depth>.4)&(depth<7)).astype('uint8'),np.ones((5,5),'uint8'))>0
            span=cv2.dilate(depth,np.ones((3,3),'uint8'))-cv2.erode(depth,np.ones((3,3),'uint8'));clean &= span<.10+.015*depth
            good &= clean[xy[:,1],xy[:,0]];z=depth[xy[:,1],xy[:,0]]
            points=np.c_[uv,np.ones(len(uv))]@np.linalg.inv(K).T*z[:,None]
            return points@pose[:3,:3].T+pose[:3,3],good
        for ref in anchor['references']:
            rr,dd,kk,tt=reference(ref['image']);match=matcher.match(rgb,rr)
            aa0,ga=lift(match.query_points,dep,K,local,keep);bb0,gb=lift(match.render_points,dd,kk,tt,np.ones(dd.shape,bool))
            prior=np.linalg.norm(aa0@coarse[:3,:3].T+coarse[:3,3]-bb0,axis=1)<.8
            good=ga&gb&prior&(match.scores>.2)
            for q,a0,b0 in zip(match.query_points[good],aa0[good],bb0[good]):
                key=tuple(np.rint(q*10).astype(int));byquery.setdefault(key,[]).append(b0);localpoints[key]=a0
        accepted=[]
        for q,world in byquery.items():
            world=np.array(world);dist=np.linalg.norm(world[:,None,:]-world[None,:,:],axis=2);support=dist<.12;best=int(np.argmax(support.sum(1)))
            if support[best].sum()<2:continue
            target=np.median(world[support[best]],axis=0)
            aa.append(localpoints[q]);bb.append(target);ii.append(i);accepted.append(q)
        notes.append(dict(index=i,frame=f['id'],pairs=len(accepted),partition='holdout' if i%4==3 else 'fit',two_reference_consensus_required=True))
        print('REF_MATCH',i,f['id'],len(accepted),flush=True)
    a=np.array(aa);b=np.array(bb);ids=np.array(ii);np.savez_compressed(OUT/'learned_alignment_pairs.npz',local=a,map=b,view=ids)
    if len(a)<80:raise ValueError('insufficient static alignment correspondences')
    print('MATCHES_READY',len(a),flush=True)
