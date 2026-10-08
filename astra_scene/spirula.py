"""Reusable video reconstruction and scoped fusion, always into candidate files."""
from pathlib import Path
import math
import argparse,json,itertools,os,subprocess,shutil
import cv2,numpy as np,pycolmap
from plyfile import PlyData,PlyElement
from scipy.spatial.transform import Rotation
from .io import ROOT,write_json,sha256,output_path
from .pipeline import environment,run_stage
def ply_degree(path):
    names=PlyData.read(str(path),mmap=True)['vertex'].data.dtype.names
    count=sum(n.startswith('f_rest_') for n in names)
    degree=int(round(math.sqrt(1+count/3)-1))
    if count!=3*((degree+1)**2-1):raise ValueError('无效 SH 系数数量')
    return degree

def sh_basis(d):
    x,y,z=d.T
    return np.column_stack([np.full(len(d),.28209479177387814),-.4886025119029199*y,.4886025119029199*z,-.4886025119029199*x,
        1.0925484305920792*x*y,-1.0925484305920792*y*z,.31539156525252005*(2*z*z-x*x-y*y),
        -1.0925484305920792*x*z,.5462742152960396*(x*x-y*y)])
def transform_ply(source,dest,scale,R,t):
    v=PlyData.read(str(source),mmap=True)['vertex'].data.copy()
    p=np.column_stack([v[n] for n in ['x','y','z']]);p=scale*p@R.T+t
    for j,n in enumerate(['x','y','z']):v[n]=p[:,j]
    for j in range(3):v['scale_'+str(j)]+=np.log(scale)
    q=np.column_stack([v['rot_'+str(j)] for j in [1,2,3,0]])
    q=(Rotation.from_matrix(R)*Rotation.from_quat(q)).as_quat()
    for j,k in enumerate([1,2,3,0]):v['rot_'+str(k)]=q[:,j]
    rng=np.random.default_rng(12);dirs=rng.normal(size=(200,3));dirs/=np.linalg.norm(dirs,axis=1)[:,None]
    degree=ply_degree(source)
    if degree not in (0,1,2):raise ValueError('当前重建适配器支持 SH 0–2 阶')
    basis_count=(degree+1)**2;rest_count=3*(basis_count-1)
    Q=np.linalg.lstsq(sh_basis(dirs)[:,:basis_count],sh_basis(dirs@R)[:,:basis_count],rcond=None)[0];Q[np.abs(Q)<1e-12]=0
    coeff=np.zeros((len(v),basis_count,3),np.float32)
    for j in range(3):coeff[:,0,j]=v['f_dc_'+str(j)]
    if rest_count:
        rest=np.column_stack([v['f_rest_'+str(j)] for j in range(rest_count)])
        coeff[:,1:]=rest.reshape(-1,3,basis_count-1).transpose(0,2,1)
    rotated=np.einsum('ij,njc->nic',Q,coeff)
    predicted=sh_basis(dirs[:10])[:,:basis_count]@Q
    assert np.max(np.abs(predicted-sh_basis(dirs[:10]@R)[:,:basis_count]))<1e-10
    for j in range(3):v['f_dc_'+str(j)]=rotated[:,0,j]
    if rest_count:
        rest=rotated[:,1:].transpose(0,2,1).reshape(-1,rest_count)
        for j in range(rest_count):v['f_rest_'+str(j)]=rest[:,j]
    PlyData([PlyElement.describe(v,'vertex')],text=False).write(str(dest))
    return v

def align_dataset(source,run):
    recon=pycolmap.Reconstruction(str(run/'dataset/sparse/0'))
    manifest=json.loads((run/'dataset_manifest.json').read_text())
    old=source/'job'
    d=np.load(old/'views.npz');split=json.loads((old/'refit_report.json').read_text())
    byname={im.name:im for im in recon.images.values()}
    rows=[]
    for row in manifest['frames']:
        im=byname[row['name']]
        w2c=np.eye(4);w2c[:3]=im.cam_from_world().matrix()
        c2w=np.linalg.inv(w2c)
        rows.append({**row,'pose':c2w.tolist(),'camera_id':int(im.camera_id)})
    train=[r for r in rows if r['legacy_view'] in split['train_views']]
    x=np.array([r['pose'] for r in train])[:,:3,3]
    y=np.array([d['poses'][r['legacy_view'],:3,3] for r in train])
    def fit(x,y):
        xm=x.mean(0);ym=y.mean(0);a=x-xm;b=y-ym
        U,S,V=np.linalg.svd(b.T@a/len(x))
        fix=np.ones(3);fix[-1]=np.linalg.det(U@V)
        R=U@np.diag(fix)@V
        scale=(S*fix).sum()/np.mean(np.sum(a*a,1))
        return scale,R,ym-scale*R@xm
    def residual(transform):
        scale,R,t=transform
        p=np.linalg.norm(scale*x@R.T+t-y,axis=1)
        angle=np.array([Rotation.from_matrix(d['poses'][r['legacy_view'],:3,:3].T@R@np.array(r['pose'])[:3,:3]).magnitude() for r in train])
        return p,angle
    best=None
    for ix in itertools.combinations(range(len(train)),3):
        try:tr=fit(x[list(ix)],y[list(ix)])
        except np.linalg.LinAlgError:continue
        p,a=residual(tr);inside=(p<.15)&(a<np.deg2rad(8))
        score=(int(inside.sum()),-float(np.median(p)+np.median(a)))
        if best is None or score>best[0]:best=(score,tr,inside)
    if best[2].sum()<4:raise ValueError('insufficient_pose_alignment_consensus')
    tr=fit(x[best[2]],y[best[2]])
    scale,R,t=tr;p,a=residual(tr)
    cameras={str(k):dict(model=str(c.model.name),width=int(c.width),height=int(c.height),params=c.params.tolist()) for k,c in recon.cameras.items()}
    report=dict(method='robust_similarity_on_training_camera_anchors_from_fixed_map_bundle',scale=float(scale),rotation=R.tolist(),translation=t.tolist(),
        training_anchors=[r['legacy_view'] for r in train],position_residuals_m=p.tolist(),rotation_residuals_deg=np.rad2deg(a).tolist(),
        median_position_m=float(np.median(p)),p90_position_m=float(np.percentile(p,90)),median_rotation_deg=float(np.rad2deg(np.median(a))),
        fit_inliers=int(best[2].sum()),accepted=bool(np.percentile(p,90)<.15 and np.median(a)<np.deg2rad(5)),
        limitation='camera_anchor_alignment_not_independent_static_surface_registration',frames=rows,cameras=cameras)
    write_json(run/'alignment.json',report)
    print({k:report[k] for k in ['scale','median_position_m','p90_position_m','median_rotation_deg','fit_inliers','accepted']})
    
    if not report['accepted']:raise ValueError('Spirula 与原地图对齐未通过，不能合并')
    return report
    

def asset(source,verify=False):
    path=source/'reconstruction.json'
    if not path.exists():return None
    spec=json.loads(path.read_text())
    ply=Path(spec['aligned_ply']).resolve()
    if not ply.is_relative_to(ROOT) or not ply.is_file():raise ValueError('高质量重建文件不可用')
    if sha256(source/'job/views.npz')!=spec['views_sha256']:raise ValueError('重建对应的几何已变化，请重新重建')
    cfg=json.loads((source/'config.json').read_text())
    original=json.loads((source/'source.json').read_text())
    if spec['video_sha256']!=original['video_sha256'] or spec['base_sha256']!=original['sources']['map_ply']['sha256']:
        raise ValueError('重建与原地图或视频不匹配')
    if verify and (sha256(ply)!=spec['ply_sha256'] or sha256(cfg['map_ply'])!=spec['base_sha256']):raise ValueError('重建或原模型文件校验失败')
    return {**spec,'key':sha256(path)}

def fuse(source,scope,dest,lineage=None,expected_key=None):
    from .roi import contains
    spec=asset(source,True)
    if spec and spec.get('candidate_enabled') is False:raise ValueError('正在准备与原地图同阶的 SH 模型，请稍后预览')
    if not spec or (expected_key is not None and spec['key']!=expected_key):raise ValueError('高质量模型已变化，请重新预览')
    cfg=json.loads((source/'config.json').read_text())
    base=PlyData.read(cfg['map_ply'],mmap=True)['vertex'].data
    base_degree=ply_degree(cfg['map_ply']);fresh_degree=ply_degree(spec['aligned_ply'])
    if fresh_degree<base_degree or fresh_degree!=spec.get('output_sh_degree',base_degree):raise ValueError('候选 SH 与所选阶数不匹配')
    if spec.get('sh_selection')!='explicit' and fresh_degree!=base_degree:raise ValueError('默认 SH 必须与原地图一致，请使用同阶重建')
    fresh=PlyData.read(spec['aligned_ply'],mmap=True)['vertex'].data
    xyz=np.column_stack([fresh[n] for n in ['x','y','z']]);chosen=contains(xyz,scope)
    patch=fresh[chosen]
    if not len(patch):raise ValueError('选区内没有高质量重建内容，请调整范围')
    # Union selection; no content in gaps between independent boxes is changed.
    keep=np.ones(len(base),bool)
    if scope['mode']=='replace':
        keep=~contains(np.column_stack([base[n] for n in ['x','y','z']]),scope)
    elif scope['mode']!='add':raise ValueError('该合并仅用于新增或替换')
    if not keep.any():raise ValueError('不能替换整个原模型')
    # Preserve all original fields. Higher SH coefficients of SH0 base are exactly zero.
    fields=list(base.dtype.descr)+[(n,fresh.dtype[n]) for n in fresh.dtype.names if n not in base.dtype.names]
    dtype=np.dtype(fields)
    count=int(keep.sum());candidate=np.zeros(count+len(patch),dtype=dtype)
    for name in base.dtype.names:candidate[name][:count]=base[name][keep]
    for name in fresh.dtype.names:candidate[name][count:]=patch[name]
    output_path(dest);dest.parent.mkdir(parents=True,exist_ok=True)
    PlyData([PlyElement.describe(candidate,'vertex')],text=False).write(str(dest))
    if lineage:
        ids=np.r_[np.flatnonzero(keep),np.full(len(patch),-1,dtype=np.int64)]
        np.savez_compressed(lineage,original_ids=ids,owner=np.full(len(ids),-1,dtype=np.int64))
    return dict(backend='spirula',base_sh_degree=base_degree,candidate_sh_degree=fresh_degree,reconstruction_key=spec['key'],new_gaussians=len(patch),removed_original=int((~keep).sum()),
        reused_reconstruction=True,retrained=False,region_count=len(scope.get('regions',[scope])),
        base_fields_byte_preserved=True,extra_base_SH_coefficients_zero=True,
        limitation='Gaussian centers are scoped; footprints and floor appearance may cross the boundary')

def appearance_metrics(pred,gt,box):
    x0,y0,x1,y1=box;a=pred[y0:y1,x0:x1].astype(np.float64)/255;b=gt[y0:y1,x0:x1].astype(np.float64)/255
    mse=np.mean((a-b)**2)
    ma=cv2.GaussianBlur(a,(11,11),1.5);mb=cv2.GaussianBlur(b,(11,11),1.5)
    va=cv2.GaussianBlur(a*a,(11,11),1.5)-ma*ma;vb=cv2.GaussianBlur(b*b,(11,11),1.5)-mb*mb
    cov=cv2.GaussianBlur(a*b,(11,11),1.5)-ma*mb
    ssim=((2*ma*mb+.01**2)*(2*cov+.03**2)/((ma*ma+mb*mb+.01**2)*(va+vb+.03**2))).mean()
    return dict(rgb_l1=float(np.mean(np.abs(a-b))),psnr=float(-10*np.log10(max(mse,1e-12))),ssim=float(ssim))

def local_camera(folder,alignment,frame,width=1280):
    from vps.refinement.gaussian_renderer import GaussianCamera
    c=alignment['cameras'][str(frame['camera_id'])]
    if c['model']!='OPENCV':raise ValueError('Unsupported SfM camera model')
    fx,fy,cx,cy,*dist=c['params'];K=np.array([[fx,0,cx],[0,fy,cy],[0,0,1.]])
    raw=cv2.cvtColor(cv2.imread(str(folder/'dataset/images'/frame['name'])),cv2.COLOR_BGR2RGB)
    gt=cv2.undistort(raw,K,np.array(dist),None,K);h=int(round(c['height']*width/c['width']))
    gt=cv2.resize(gt,(width,h),interpolation=cv2.INTER_AREA);K[0]*=width/c['width'];K[1]*=h/c['height']
    return np.array(frame['pose']),GaussianCamera(width,h,K[0,0],K[1,1],K[0,2],K[1,2]),gt

def prepare(source,gpu,sh_degree='auto'):
    source=output_path(source);folder=source/'spirula';folder.mkdir(exist_ok=False)
    cfg=json.loads((source/'config.json').read_text());report=json.loads((source/'job/report.json').read_text())
    capture=json.loads(Path(report['capture']).read_text());split=json.loads((source/'job/refit_report.json').read_text())
    base_degree=ply_degree(cfg['map_ply']);degree=base_degree if sh_degree=='auto' else int(sh_degree)
    if degree not in (0,1,2) or degree<base_degree:raise ValueError('SH 必须保持原图阶数或明确提高，当前支持 0–2 阶')
    write_json(folder/'settings.json',dict(sh_degree=degree,base_sh_degree=base_degree,selection='auto' if sh_degree=='auto' else 'explicit'))
    binary=ROOT/'vendor/spirula_20260920/spirula'
    env=environment(gpu);env.update(SS_LANG='en',XDG_CONFIG_HOME=str(ROOT/'cache/spirula/config'),
        XDG_CACHE_HOME=str(ROOT/'cache/spirula'),VK_ICD_FILENAMES='/usr/share/vulkan/icd.d/nvidia_icd.json')
    uuid=subprocess.check_output(['nvidia-smi','--id',str(gpu),'--query-gpu=uuid','--format=csv,noheader'],text=True).strip().removeprefix('GPU-').replace('-','')
    device='uuid:'+uuid
    raw=folder/'raw_frames'
    is_photos=capture.get('source')=='phone_rgb_images'
    frames=capture['frames'];lookup={int(f['frame_index']):i for i,f in enumerate(frames)}
    held=set(split['validation_views']+split['test_views'])
    if not split['validation_views'] or not split['test_views']:raise ValueError('缺少独立留出的验证与测试画面')
    # Never reintroduce quarantined views through a second capped extraction.
    eligible=set(split['train_views'])|held
    chosen={int(frames[i]['frame_index']) for i in eligible}
    images=folder/'dataset/images';images.mkdir(parents=True)
    cap=None;manifest=[]
    try:
        for index in sorted(chosen):
            # Extraction already decoded this exact frame sequentially.
            # Re-seeking a VFR/long-GOP video can fail at its final valid frame.
            bgr=cv2.imread(frames[lookup[index]]['image']);ok=bgr is not None
            if not ok:raise ValueError('已提取图片读取失败：'+str(index))
            view=lookup.get(index);kind='eval' if view in held else 'train';name=f'f{index:08d}_{kind}.jpg'
            if not cv2.imwrite(str(images/name),bgr,[cv2.IMWRITE_JPEG_QUALITY,95]):raise ValueError('无法保存训练帧')
            manifest.append(dict(frame_index=index,name=name,split=kind,legacy_view=view))
    finally:
        if cap is not None:cap.release()
    write_json(folder/'dataset_manifest.json',dict(frames=manifest,source_sha256=capture['source_sha256'],holdout_policy='excluded_from_photometric_training; SfM_uses_all_frames'))
    run_stage(source,'spirula_sfm',[binary,'sfm','auto',images,'-o',folder/'dataset','--data-type','video','--quality','high','--device',device],env)
    alignment=align_dataset(source,folder)
    run_stage(source,'spirula_train',[binary,'train','3dgs','--data',folder/'dataset','--output-dir-prefix',folder,
        '--output-dir-name','training','--device',device,'--quality','high','--cap-max','1500000','--num-iterations','24000',
        '--sh-degree',str(degree),'--train-resolution-divisor','1','--eval-mode','filename','--validation-fraction','0.1',
        '--steps-per-save','8000','--save-only-latest-checkpoint','0','--save-eval-images','1','--disable-viewer','1',
        '--keep-viewer-alive','0','--cache-images','cpu'],env)
    finish_reconstruction(source,folder,alignment)

def finish_reconstruction(source,folder,alignment):
    split=json.loads((source/'job/refit_report.json').read_text())
    from vps.refinement.gaussian_renderer import GsplatGaussianRenderer
    import torch
    valid=[f for f in alignment['frames'] if f['legacy_view'] in split['validation_views']]
    checks=[]
    write_json(source/'status.json',dict(state='running',stage='spirula_validate'))
    for checkpoint in sorted((folder/'training').glob('step-*.ckpt/splat.ply')):
        renderer=GsplatGaussianRenderer('cuda');errors=[]
        for f in valid:
            pose,camera,gt=local_camera(folder,alignment,f)
            pred=renderer.render(checkpoint,pose,camera).rgb
            errors.append(appearance_metrics(pred,gt,(0,0,gt.shape[1],gt.shape[0])))
        l1=float(np.mean([r['rgb_l1'] for r in errors]));ssim=float(np.mean([r['ssim'] for r in errors]))
        checks.append(dict(path=str(checkpoint),validation_l1=l1,validation_ssim=ssim,score=l1+.25*(1-ssim)))
        del renderer;torch.cuda.empty_cache()
    if len(checks)<2 or not valid:raise ValueError('验证检查点不完整')
    selected=Path(min(checks,key=lambda r:r['score'])['path'])
    write_json(folder/'checkpoint_selection.json',dict(checkpoints=checks,selected=str(selected),selection='validation_L1_plus_0.25_times_1_minus_SSIM',test_views_used=False))
    aligned=folder/'aligned_scene.ply'
    transform_ply(selected,aligned,alignment['scale'],np.array(alignment['rotation']),np.array(alignment['translation']))
    origin=json.loads((source/'source.json').read_text())
    trained_degree=ply_degree(aligned)
    settings=json.loads((folder/'settings.json').read_text()) if (folder/'settings.json').exists() else {}
    base_degree=ply_degree(json.loads((source/'config.json').read_text())['map_ply'])
    write_json(source/'reconstruction.json',dict(schema='astra.reconstruction.v1',backend='spirula',
        aligned_ply=str(aligned),ply_sha256=sha256(aligned),video_sha256=origin['video_sha256'],
        trained_sh_degree=trained_degree,output_sh_degree=trained_degree,base_sh_degree=base_degree,sh_selection=settings.get('selection','legacy_experiment'),
        views_sha256=sha256(source/'job/views.npz'),base_sha256=origin['sources']['map_ply']['sha256'],
        alignment=str(folder/'alignment.json'),checkpoint_selection=str(folder/'checkpoint_selection.json'),
        registered_frames=len(alignment['frames']),holdout_note='SfM uses heldout images; appearance training excludes them',
        navigation_depth='VGGT estimated depth, not measured'))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--gpu',type=int,default=1);p.add_argument('--sh-degree',default='auto',choices=['auto','0','1','2']);a=p.parse_args();prepare(a.source,a.gpu,a.sh_degree)
