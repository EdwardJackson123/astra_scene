from gpu_config import gpu_index
import os,sys,json,fcntl
from pathlib import Path
ROOT=Path('/home/xujie/astra_scene');LAB=Path(__file__).resolve().parent;sys.path[:0]=[str(LAB),str(ROOT),str(ROOT/'vendor/engine')]
from astra_scene.pipeline import environment
os.environ.update(environment(gpu_index()))
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares
from map_update import fit_similarity
from astra_scene.io import write_json,sha256
from update_context import settings
CFG=settings();OUT=Path(CFG['alignment']);SOURCE=Path(CFG['source'])
BASE=Path(CFG['base_map'])
if CFG.get('apply_only'):
    report=json.loads((OUT/'similarity_alignment.json').read_text())
    if not report['accepted'] or report['base_sha256']!=sha256(BASE) or report['source_sha256']!=sha256(SOURCE/'manifest.json'):raise ValueError('alignment checkpoint mismatch')
    s=report['scale'];R=np.array(report['rotation']);t=np.array(report['translation'])
else:
    data=np.load(OUT/'learned_alignment_pairs.npz');a,b,ids=data['local'],data['map'],data['view']
    # Stable high structures and floor; movable furniture/equipment are not anchors.
    stable=(b[:,2]>CFG['stable_upper_z'])|(b[:,2]<CFG['stable_lower_z']);fit=stable&(ids%4!=3);held=stable&(ids%4==3)
    rng=np.random.default_rng(21);train=np.flatnonzero(fit);best=np.zeros(len(a),bool)
    if len(train)<30 or held.sum()<30:raise ValueError('insufficient independent static alignment evidence')
    for _ in range(5000):
        ix=rng.choice(train,3,replace=False)
        try:s,R,t=fit_similarity(a[ix],b[ix])
        except ValueError:continue
        if not .85<s<1.15:continue
        error=np.linalg.norm(s*a@R.T+t-b,axis=1);ok=(error<.15)&fit
        if ok.sum()>best.sum():best=ok
    if best.sum()<30:raise ValueError('static alignment consensus too small')
    s,R,t=fit_similarity(a[best],b[best]);x0=np.r_[np.log(s),Rotation.from_matrix(R).as_rotvec(),t]
    def residual(x):return (np.exp(x[0])*a[best]@Rotation.from_rotvec(x[1:4]).as_matrix().T+x[4:]-b[best]).ravel()
    x=least_squares(residual,x0,loss='soft_l1',f_scale=.04).x;s=float(np.exp(x[0]));R=Rotation.from_rotvec(x[1:4]).as_matrix();t=x[4:]
    error=np.linalg.norm(s*a@R.T+t-b,axis=1);notes=[]
    for i in sorted(set(ids.tolist())):
        mask=stable&(ids==i);good=mask&(error<.15)
        notes.append(dict(view=i,partition='holdout' if i%4==3 else 'fit',matches=int(mask.sum()),inliers=int(good.sum()),median_inlier_m=float(np.median(error[good])) if good.any() else None))
    def stats(mask):
        e=error[mask];good=e<.15
        return dict(matches=len(e),inliers=int(good.sum()),inlier_fraction=float(good.mean()),median_inlier_m=float(np.median(e[good])),median_all_m=float(np.median(e)),p90_all_m=float(np.percentile(e,90)))
    accepted=.85<s<1.15 and stats(held)['inlier_fraction']>=.4 and stats(held)['inliers']>=30 and stats(held)['median_inlier_m']<.1 and sum(v['inliers']>=8 for v in notes if v['partition']=='fit')>=6 and sum(v['inliers']>=8 for v in notes if v['partition']=='holdout')>=3
    report=dict(accepted=bool(accepted),scale=s,rotation=R.tolist(),translation=t.tolist(),method='single global Sim3; SuperPoint/LightGlue to original RGB/depth; >=2 reference consensus; floor and high static structures; robust fit',fit=stats(fit),holdout=stats(held),views=notes,base_sha256=sha256(BASE),source_sha256=sha256(SOURCE/'manifest.json'),metric_scale_conflict=abs(s-1)>.05,coordinate_policy='original_map_coordinate_units; scale sensor depth and local camera translations consistently',absolute_metric_accuracy_verified=False,navigation_deployment_accepted=False,holdout_note='Excluded from fitting; development regression after earlier alignment diagnostics, not independent revisit truth')
    write_json(OUT/'similarity_alignment.json',report);print(json.dumps(report),flush=True)
    if not accepted:raise ValueError('similarity_alignment_gate_failed')
if CFG.get('fresh_ply'):
    with (ROOT/'runs/gpu.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        from astra_scene.spirula import transform_ply
        transform_ply(Path(CFG['fresh_ply']),OUT/'aligned_masked_sh0.ply',s,R,t)
m=json.loads((SOURCE/'manifest.json').read_text());frames=[]
for frame in m['frames']:
    f=dict(frame);pose=np.array(f['pose_c2w']);pose[:3,3]=s*R@pose[:3,3]+t;pose[:3,:3]=R@pose[:3,:3];f['pose_c2w']=pose.tolist();frames.append(f)
write_json(OUT/'aligned_frames.json',dict(frames=frames,depth_multiplier=s,map_metadata=m['map_metadata'],roi=CFG['roi']))
