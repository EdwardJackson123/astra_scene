"""Audit residual old geometry before changing scope or optimization settings."""
import json
from pathlib import Path
import numpy as np


def audit(ctx):
    from plyfile import PlyData
    from astra_scene.context_fusion import field,context_scope
    from astra_scene.geometry import evidence,voxel_fuse,project
    from connected_regions import components
    from map_update import roi_weights
    from connected_map_update import write
    root,source,alignment,out,base,data,registration,rows,ids,held=ctx
    dest=out/'residual_audit';dest.mkdir(exist_ok=True)
    scope=json.loads((out/'artifacts/context_scope.json').read_text())
    old=PlyData.read(base,mmap=True)['vertex'].data;p=np.column_stack([old[n] for n in ('x','y','z')])
    known=np.load(out/'artifacts/lineage.npz')['old_ids'];retained=np.zeros(len(old),bool);retained[known]=True
    search=context_scope(scope,.8)
    select=retained&(field(p,search,.01)>0)&(roi_weights(p,data['map_metadata'],data['roi'])>0)&(old['opacity']>-2.2)
    ix=np.flatnonzero(select);ss,ff,oo=evidence(p[ix],rows,ids,.22)
    inside=field(p[ix],scope,.30)>0
    consensus=(ff>=6)&(ss<=2)&(ff>=6*np.maximum(ss,1))
    strict=(ff>=4)&(ss==0)
    audit=dict(retained_near_scopes=len(ix),current_scope_points=int(inside.sum()),
        strict_clearance_remaining=int(strict.sum()),strict_inside=int((strict&inside).sum()),strict_outside=int((strict&~inside).sum()),
        dominant_clearance_remaining=int(consensus.sum()),dominant_inside=int((consensus&inside).sum()),dominant_outside=int((consensus&~inside).sum()),
        unknown=int(((ss==0)&(ff==0)).sum()),conflicted=int(((ss>0)&(ff>=4)).sum()),tolerance_m=.22,preview_only=True)
    np.savez_compressed(dest/'old_evidence.npz',old_ids=ix,surface=ss,free=ff,occluded=oo,inside_scope=inside,dominant_clearance=consensus)
    q=p[ix[consensus]];q,_=voxel_fuse(q,np.zeros_like(q),.06)
    groups=components(q,.18,20);reports=[]
    for j,g in enumerate(groups):
        v=q[g];reports.append(dict(id=f'old_residual_{j:03d}',points=len(g),bounds=[v.min(0).tolist(),v.max(0).tolist()],outside_current_scope_fraction=float((field(v,scope,.3)==0).mean())))
    write(dest/'report.json',dict(**audit,components=reports));print('RESIDUAL_AUDIT',json.dumps(audit),flush=True)
    np.savez_compressed(dest/'components.npz',**{r['id']:q[g] for r,g in zip(reports,groups)})
    # Show actual residual supports on old/current images, not a new global box.
    import cv2
    from PIL import Image
    from astra_scene.postprocess import render_map
    import fcntl
    image_dir=dest/'images';image_dir.mkdir(exist_ok=True)
    with (root/'runs/gpu.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        from vps.refinement.gaussian_renderer import GsplatGaussianRenderer
        renderer=GsplatGaussianRenderer('cuda')
        for r,g in zip(reports[:8],groups[:8]):
            v=q[g];choices=[]
            for i in ids:
                index,u,y,z=project(v,rows[i]);valid=rows[i]['clear'][y,u]&(rows[i]['depth'][y,u]-z>.22)
                if valid.sum()<15:continue
                mask=np.zeros(rows[i]['depth'].shape,'uint8');mask[y[valid],u[valid]]=255;mask=cv2.dilate(mask,np.ones((7,7),'uint8'))
                choices.append((int(valid.sum()),i,mask))
            choices.sort(reverse=True,key=lambda x:x[0]);chosen=[]
            for _,i,m in choices:
                if all(np.linalg.norm(rows[i]['pose'][:3,3]-rows[k]['pose'][:3,3])>.3 for k,_ in chosen):chosen.append((i,m))
                if len(chosen)==3:break
            r['images']=[]
            for i,m in chosen:
                old_render=render_map(renderer,base,rows[i])['rgb'];gt=np.uint8(rows[i]['rgb']*255);gt[~rows[i]['keep']]=(gt[~rows[i]['keep']]*.15).astype('uint8')
                contours,_=cv2.findContours(m,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                for im in (old_render,gt):cv2.drawContours(im,contours,-1,(255,90,40),1)
                name=f'{r["id"]}_{i:03d}.jpg';Image.fromarray(np.concatenate([old_render,gt],1)).save(image_dir/name,quality=92);r['images'].append(dict(view=i,file=name))
    write(dest/'report.json',dict(**audit,components=reports))
