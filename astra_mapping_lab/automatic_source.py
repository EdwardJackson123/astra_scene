"""Prepare masked metric SfM observations on the server, without robot extrinsics."""
from gpu_config import gpu_index
import json,os,subprocess,hashlib
from pathlib import Path
from recording import atomic


def canonical_name(name):
    return hashlib.sha256(name.encode()).hexdigest()[:20]+'_'+Path(name).name


def complete_holdouts(entries,registered,timestamps):
    """Resolve missing SfM views before any depth-scale fitting or training."""
    rows=[dict(e) for e in entries];registered=set(registered)
    replacements=[]
    for row in rows:
        if row['split']!='eval' or row['name'] in registered:continue
        available=[e for e in rows if e['split']=='train' and e['name'] in registered]
        if not available:raise ValueError('无法补齐已注册的留出视角')
        selected=min(available,key=lambda e:(abs(timestamps[e['frame_id']]-timestamps[row['frame_id']]),e['name']))
        selected['split']='eval';row['split']='unregistered'
        replacements.append(dict(missing_frame=row['frame_id'],replacement_frame=selected['frame_id'],
            time_gap_s=abs(timestamps[selected['frame_id']]-timestamps[row['frame_id']]),reason='SfM did not register original held-out image'))
    held=[e for e in rows if e['split']=='eval' and e['name'] in registered]
    train=[e for e in rows if e['split']=='train' and e['name'] in registered]
    if len(held)<6 or len(train)<20:raise ValueError('SfM 已注册训练/留出视角不足')
    return rows,dict(replacements=replacements,training_frames=len(train),heldout_frames=len(held),
        policy='nearest registered timestamp; fixed before RGB-D scale fitting and 3DGS training; no appearance metrics used',
        independent_revisit=False)


def prepare(source,job,notify):
    import numpy as np,cv2,fcntl,concurrent.futures
    from PIL import Image
    from replay import read_dataset,asset
    from depth_io import load_depth
    from sfm_rgbd import convert
    m,_,meta=read_dataset(source)
    if m.get('sfm_rgbd_report',{}).get('accepted') and all(f.get('mask') for f in m['frames']):
        notify('MASKED_SFM_INPUT_VERIFIED');return source
    good=[]
    for f in m['frames']:
        rgb=np.asarray(Image.open(asset(source,f['rgb'])).convert('L'))
        depth=load_depth(asset(source,f['depth']),f)
        if rgb.shape!=(480,640) or depth.shape!=rgb.shape:raise ValueError('当前自动更新仅适配 640×480、对齐且去畸变的 D455F 数据')
        if cv2.Laplacian(rgb,cv2.CV_64F).var()>=20 and np.mean(np.isfinite(depth)&(depth>.25)&(depth<6))>=.15:good.append(f)
    if len(good)<30:raise ValueError('清晰且深度有效的帧不足 30；请补采')
    if all(f.get('reconstruction_split') in ('train','eval') for f in good):
        selected=[(f,f['reconstruction_split']) for f in good]
    else:
        # No trajectory needed to select inputs for SfM. No maximum frame cap.
        train=[good[0]]
        for f in good[1:]:
            if f['timestamp_s']-train[-1]['timestamp_s']>=.7:train.append(f)
        available=[f for f in good if f['id'] not in {r['id'] for r in train}]
        if len(available)<6:
            available=good[3::5];held={f['id'] for f in available};train=[f for f in good if f['id'] not in held]
        held={available[i]['id'] for i in np.unique(np.linspace(0,len(available)-1,min(12,len(available))).astype(int))}
        training={f['id'] for f in train};selected=[(f,'eval' if f['id'] in held else 'train') for f in good if f['id'] in training|held]
    if sum(s=='train' for _,s in selected)<20 or sum(s=='eval' for _,s in selected)<6:raise ValueError('训练和留出视角不足')
    work=job/'sfm_input';images=work/'images';masks=work/'masks'
    images.mkdir(parents=True,exist_ok=False);masks.mkdir()
    entries=[]
    for i,(f,split) in enumerate(selected):
        name=f'{i:06d}_{split}.png';Image.open(asset(source,f['rgb'])).convert('RGB').save(images/name)
        entries.append(dict(frame_id=f['id'],name=name,split=split))
    # convert reads the original observations and this exact image-to-frame mapping.
    canonical=job/'sfm_raw';canonical.mkdir()
    for f,_ in selected:
        for field in ('rgb','depth','mask'):
            if f.get(field):
                original=asset(source,f[field]);dest=canonical/canonical_name(f[field])
                if not dest.exists():os.link(original,dest)
    original=asset(source,m['map_image']);map_name=canonical_name(m['map_image'])
    if not (canonical/map_name).exists():os.link(original,canonical/map_name)
    raw={**m,'map_metadata':meta,'map_image':map_name,'schema_version':1,'ground_z_m':m.get('ground_z_m',0)}
    raw['frames']=[{**f,**{k:canonical_name(f[k]) for k in ('rgb','depth','mask') if f.get(k)}} for f,_ in selected]
    atomic(canonical/'manifest.json',raw);atomic(work/'package.json',dict(frames=entries))
    root=Path(os.environ.get('ASTRA_SCENE_ROOT','/home/xujie/astra_scene'))
    binary=os.environ.get('ASTRA_SPIRULA',str(root/'vendor/spirula_20260920/spirula'))
    gpu=subprocess.check_output(['nvidia-smi','--id',str(gpu_index()),'--query-gpu=uuid','--format=csv,noheader'],text=True).strip().replace('GPU-','').replace('-','')
    env=os.environ.copy();env.update(SS_LANG='en',VK_ICD_FILENAMES='/usr/share/vulkan/icd.d/nvidia_icd.json',OMP_NUM_THREADS='4')
    def mask_one(pair):
        from artifact_cache import sam_mask
        i,entry=pair;folder=work/'segmentation'/str(i)
        rgb=np.asarray(Image.open(images/entry['name']).convert('RGB'))
        keep,cache_info=sam_mask(rgb,root,folder,'uuid:'+gpu,'person','robot;mannequin',False,.03,env)
        person=~keep
        f=selected[i][0]
        if f.get('mask'):
            from masks import frame_mask
            keep &= frame_mask(source,f,keep.shape)
        Image.fromarray(keep.astype('uint8')*255).save(masks/entry['name'])
        return dict(masked_pixels=int(person.sum()),**cache_info)
    with (root/'runs/gpu.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);notify('PERSON_MASKING')
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:counts=list(pool.map(mask_one,enumerate(entries)))
        atomic(job/'person_mask.json',dict(enabled=True,frames=len(entries),nonempty=sum(x['masked_pixels']>0 for x in counts),cache_hits=sum(x['cache_hit'] for x in counts),cache_records=counts,method='independent SAM3 person mask, robot/mannequin negatives, dilation ratio 0.03'))
        notify('SPIRULA_SFM')
        focal=float(np.median([f['K'][0][0] for f,_ in selected]))
        command=[binary,'sfm','auto',str(images),'-o',str(job/'sfm'),'--data-type','video','--quality','high','--masks',str(masks),'--camera-mode','single','--camera-model','pinhole','--focal',str(focal),'--refine-principal-point','--audit','--threads','8','--device','uuid:'+gpu]
        atomic(job/'sfm_command.json',command)
        from stage_runtime import run as bounded_run
        with (job/'sfm.log').open('w') as log:bounded_run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=14400)
    models=list((job/'sfm').rglob('cameras.bin'))
    if len(models)!=1:raise ValueError('SfM 未产生唯一连通模型，禁止自动选取未知片段')
    from sfm_rgbd import read_model
    _,registered,_=read_model(models[0].parent)
    entries,split_report=complete_holdouts(entries,[f['name'] for f in registered],{f['id']:f['timestamp_s'] for f,_ in selected})
    atomic(work/'package.json',dict(frames=entries))
    atomic(job/'evaluation_split.json',split_report)
    notify('SFM_RGBD_CHECK');target=job/'sfm_source'
    convert(canonical,work,models[0].parent,target,mask_folder=masks)
    return target
