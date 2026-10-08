"""Metric RGB-D camera tracking and explicit map alignment, without body extrinsics."""
import json,math,shutil,time
from pathlib import Path
import numpy as np
from PIL import Image
from replay import asset,load_dataset,fingerprint
from recording import atomic
from core import world_to_pixel
from depth_io import load_depth,validate_depth_manifest


def rigid(T):
    T=np.asarray(T,float)
    if T.shape!=(4,4) or not np.isfinite(T).all() or not np.allclose(T[3],[0,0,0,1]) or not np.allclose(T[:3,:3].T@T[:3,:3],np.eye(3),atol=1e-4) or not np.isclose(np.linalg.det(T[:3,:3]),1,atol=1e-4):raise ValueError('无效刚体位姿')
    return T


def fit_rigid(a,b):
    a=np.asarray(a,float);b=np.asarray(b,float)
    if len(a)<3:raise ValueError('三维匹配不足')
    ac=a-a.mean(0);bc=b-b.mean(0)
    if np.linalg.svd(ac,compute_uv=False)[1]<.01:raise ValueError('匹配点共线或空间分布过窄')
    u,_,vt=np.linalg.svd(ac.T@bc);fix=np.eye(3);fix[2,2]=np.linalg.det(vt.T@u.T)
    R=vt.T@fix@u.T;T=np.eye(4);T[:3,:3]=R;T[:3,3]=b.mean(0)-R@a.mean(0);return T


def robust_rigid(a,b,threshold=.06,min_inliers=24):
    a=np.asarray(a,float);b=np.asarray(b,float)
    if a.shape!=b.shape or a.ndim!=2 or a.shape[1]!=3 or not np.isfinite(a).all() or not np.isfinite(b).all() or len(a)<min_inliers:raise ValueError('有效三维匹配不足')
    rng=np.random.default_rng(9);best=np.zeros(len(a),bool)
    for _ in range(160):
        ix=rng.choice(len(a),3,replace=False)
        try:T=fit_rigid(a[ix],b[ix])
        except ValueError:continue
        ok=np.linalg.norm(a@T[:3,:3].T+T[:3,3]-b,axis=1)<threshold
        if ok.sum()>best.sum():best=ok
    if best.sum()<min_inliers or best.mean()<.35:raise ValueError('三维匹配一致性不足')
    T=fit_rigid(a[best],b[best]);error=np.linalg.norm(a@T[:3,:3].T+T[:3,3]-b,axis=1);best=error<threshold
    if best.sum()<min_inliers or np.median(error[best])>.035:raise ValueError('三维配准残差过大')
    return T,dict(matches=len(a),inliers=int(best.sum()),inlier_fraction=float(best.mean()),median_error_m=float(np.median(error[best])),p90_error_m=float(np.percentile(error[best],90)))


def features(root,frame):
    import cv2
    cv2.setNumThreads(1)
    rgb=np.asarray(Image.open(asset(root,frame['rgb'])).convert('RGB'));depth=load_depth(asset(root,frame['depth']),frame)
    if depth.ndim!=2 or depth.shape!=rgb.shape[:2] or depth.size>4_000_000:raise ValueError('RGB-D 形状错误')
    K=np.asarray(frame['K'],float)
    if K.shape!=(3,3) or not np.isfinite(K).all() or K[0,0]<=0 or K[1,1]<=0:raise ValueError('内参无效')
    from masks import frame_mask
    observation_mask=frame_mask(root,frame,depth.shape)
    keypoints,desc=cv2.SIFT_create(nfeatures=1800).detectAndCompute(cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY),None)
    if desc is None:return np.zeros((0,3)),np.zeros((0,128),np.float32)
    xyz=[];descriptors=[]
    for kp,d in zip(keypoints,desc):
        x,y=kp.pt;ix,iy=int(round(x)),int(round(y))
        if not (1<=ix<depth.shape[1]-1 and 1<=iy<depth.shape[0]-1):continue
        if not observation_mask[iy-1:iy+2,ix-1:ix+2].all():continue
        patch=depth[iy-1:iy+2,ix-1:ix+2];z=float(depth[iy,ix])
        if not np.isfinite(patch).all() or not .25<z<6 or np.min(patch)<=.25 or np.max(patch)-np.min(patch)>.08+.02*z:continue
        xyz.append([(x-K[0,2])*z/K[0,0],(y-K[1,2])*z/K[1,1],z]);descriptors.append(d)
    return np.asarray(xyz).reshape(-1,3),np.asarray(descriptors,dtype=np.float32).reshape(-1,128)


def correspond(a,b):
    import cv2
    if min(len(a[0]),len(b[0]))<24:raise ValueError('纹理/深度有效特征不足')
    matcher=cv2.BFMatcher();forward=matcher.knnMatch(a[1],b[1],k=2);back=matcher.knnMatch(b[1],a[1],k=2)
    reverse={x[0].queryIdx:x[0].trainIdx for x in back if len(x)==2 and x[0].distance<.75*x[1].distance}
    pairs=[(x[0].queryIdx,x[0].trainIdx) for x in forward if len(x)==2 and x[0].distance<.75*x[1].distance and reverse.get(x[0].trainIdx)==x[0].queryIdx]
    if len(pairs)<24:raise ValueError('双向图像匹配不足，需要更慢移动或更多纹理')
    ia,ib=np.array(pairs).T;return a[0][ia],b[0][ib]


def raw_manifest(root):
    root=Path(root);m=json.loads(asset(root,'capture_manifest.json').read_text(encoding='utf-8'))
    if m.get('kind')!='raw_rgbd' or m.get('depth_aligned_to')!='color' or m.get('depth_rectified') is not True:raise ValueError('需要已对齐且已去畸变的 RGB-D 记录')
    validate_depth_manifest(m)
    if not isinstance(m.get('frames'),list) or len(m['frames'])<2:raise ValueError('至少需要 2 帧')
    last=-math.inf;seen=set()
    for f in m['frames']:
        if f['id'] in seen or not math.isfinite(f['timestamp_s']) or f['timestamp_s']<=last:raise ValueError('帧标识或时间戳无效')
        seen.add(f['id']);last=f['timestamp_s'];asset(root,f['rgb']);asset(root,f['depth'])
    return m


def pose_error(a,b):
    return (float(np.linalg.norm(a[:3,3]-b[:3,3])),
            math.degrees(math.acos(float(np.clip((np.trace(a[:3,:3].T@b[:3,:3])-1)/2,-1,1)))))


def connection_consensus(votes):
    """Require three agreeing edges and two distinct frames on EACH side."""
    best=[]
    for _,_,T in votes:
        support=[v for v in votes if pose_error(T,v[2])[0]<.10 and pose_error(T,v[2])[1]<3]
        if len(support)>len(best):best=support
    if len(best)<3 or len({v[0] for v in best})<2 or len({v[1] for v in best})<2:return None
    # Use a measured transform (medoid), never average rotations elementwise.
    chosen=min(best,key=lambda v:sum(pose_error(v[2],w[2])[0]+.02*pose_error(v[2],w[2])[1] for w in best))
    best=[v for v in best if pose_error(chosen[2],v[2])[0]<.10 and pose_error(chosen[2],v[2])[1]<3]
    if len(best)<3 or len({v[0] for v in best})<2 or len({v[1] for v in best})<2:return None
    return chosen[2],best


def trajectory_quality(frames,indices,components):
    stamps=[float(f['timestamp_s']) for f in frames];selected=sorted(indices)
    duration=stamps[-1]-stamps[0]
    coverage=len(selected)/len(frames)
    span=(stamps[selected[-1]]-stamps[selected[0]])/duration if selected and duration else 0.
    gaps=([stamps[selected[0]]-stamps[0],stamps[-1]-stamps[selected[-1]]]+
          [stamps[b]-stamps[a] for a,b in zip(selected,selected[1:])]) if selected else [duration]
    reasons=[]
    if len(selected)<4:reasons.append('可用帧少于 4')
    if coverage<.90:reasons.append('连通轨迹帧覆盖率低于 90%')
    if span<.95:reasons.append('连通轨迹时间覆盖率低于 95%')
    if max(gaps)>2:reasons.append('存在超过 2 秒的未覆盖时间段')
    return dict(accepted=not reasons,reasons=reasons,frame_coverage=coverage,time_coverage=span,
                max_gap_s=max(gaps),component_sizes=sorted(components,reverse=True),
                thresholds=dict(min_frame_coverage=.9,min_time_coverage=.95,max_gap_s=2),
                scope='轨迹完整性与片段连接检查；不代表绝对精度、闭环或地图对齐验收')


def require_trajectory_quality(m):
    """Completeness limits update scope, not the ability to build a connected local model."""
    if 'sfm_rgbd_report' in m and not m['sfm_rgbd_report'].get('accepted'):raise ValueError('SfM RGB-D 几何检查未通过')
    if m.get('kind')!='posed_rgbd' or 'trajectory_report' not in m:return
    report=m['trajectory_report'];q=report.get('quality_gate')
    if not q:raise ValueError('缺少轨迹检查报告')
    if q.get('accepted'):return
    ids={f['id'] for f in m.get('frames',[])}
    connected=any(ids.issubset(set(c['frames'])) for c in report.get('components',[]))
    if len(ids)<4 or not connected or not q.get('edge_checks') or q.get('inconsistent_edge_fraction',1)>.1:
        raise ValueError('局部重建检查未通过：有效连通帧不足或几何连接不一致')
    for f in m['frames']:rigid(f['pose_c2w'])



def track(root,target,cancelled=lambda:False):
    m=raw_manifest(root);target=Path(target);target.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(asset(root,m['map_image']),target/'map.png')
    frames=m['frames'];cache={};edges=[];logs=[];poses={};groups={};started=time.time()
    # Independent adjacent-frame segments remain useful after a tracking loss.
    # Additional temporal edges reconnect them only with multi-frame consensus.
    for i,f in enumerate(frames):
        if cancelled():raise ValueError('已退出建图模式，轨迹估计取消')
        report=dict(frame_id=f['id'],status='REJECTED');matches=[]
        try:
            feat=features(root,f)
            if len(feat[0])<24:raise ValueError('有效 RGB-D 特征不足')
            cache[i]=feat
            for j in range(max(0,i-12),i):
                if j not in cache or f['timestamp_s']-frames[j]['timestamp_s']>4:continue
                try:
                    a,b=correspond(cache[j],feat);T,r=robust_rigid(a,b)
                    angle=pose_error(np.eye(4),T)[1]
                    if np.linalg.norm(T[:3,3])>1 or angle>45:continue
                    matches.append((j,i,T,r));edges.append((j,i,T,r))
                except ValueError:continue
            adjacent=next((e for e in matches if e[0]==i-1),None)
            if adjacent:
                poses[i]=poses[i-1]@np.linalg.inv(adjacent[2]);groups[i]=groups[i-1]
                report.update(status='TRACKED',**adjacent[3])
            else:
                poses[i]=np.eye(4);groups[i]=i;report.update(status='SEGMENT_ORIGIN')
            report['initial_segment']=groups[i]
        except ValueError as e:report['reason']=str(e)
        logs.append(report)
        cache.pop(i-12,None)
        atomic(target/'processing_status.json',dict(state='TRACKING',processed=i+1,feature_frames=len(poses),total=len(frames)))
    connections=[]
    # Recompute votes after each merge: all transforms are expressed in the current components.
    while True:
        votes={}
        for a,b,T,r in edges:
            ga,gb=groups[a],groups[b]
            if ga==gb:continue
            # Transform from component b to component a.
            Q=poses[a]@np.linalg.inv(T)@np.linalg.inv(poses[b])
            if ga>gb:ga,gb=gb,ga;a,b=b,a;Q=np.linalg.inv(Q)
            votes.setdefault((ga,gb),[]).append((a,b,Q))
        candidates=[]
        for pair,v in votes.items():
            consensus=connection_consensus(v)
            if consensus:candidates.append((len(consensus[1]),pair,consensus))
        if not candidates:break
        _,(ga,gb),(Q,support)=max(candidates,key=lambda x:x[0])
        connections.append(dict(component_a=ga,component_b=gb,support_edges=[[a,b] for a,b,_ in support],
                                max_position_disagreement_m=max(pose_error(Q,v[2])[0] for v in support),
                                max_rotation_disagreement_deg=max(pose_error(Q,v[2])[1] for v in support)))
        for i in poses:
            if groups[i]==gb:poses[i]=Q@poses[i];groups[i]=ga
    components={}
    for i,g in groups.items():components.setdefault(g,[]).append(i)
    selected=max(components.values(),key=len) if components else []
    quality=trajectory_quality(frames,selected,[len(v) for v in components.values()])
    edge_checks=[]
    selected_set=set(selected)
    for a,b,T,r in edges:
        if a not in selected_set or b not in selected_set:continue
        error=pose_error(np.linalg.inv(poses[b])@poses[a],T)
        edge_checks.append(dict(a=a,b=b,position_error_m=error[0],rotation_error_deg=error[1]))
    inconsistent=sum(e['position_error_m']>.15 or e['rotation_error_deg']>5 for e in edge_checks)
    quality['edge_checks']=len(edge_checks)
    quality['inconsistent_edge_fraction']=inconsistent/max(1,len(edge_checks))
    quality['thresholds'].update(max_inconsistent_edge_fraction=.1,edge_position_m=.15,edge_rotation_deg=5)
    if quality['inconsistent_edge_fraction']>.1:
        quality['accepted']=False;quality['reasons'].append('重叠帧配准与合并轨迹不一致的比例超过 10%')
    atomic(target/'edge_checks.json',edge_checks)
    output=[]
    for i in selected:
        f=dict(frames[i])
        for k in ['rgb','depth']+(['mask'] if f.get('mask') else []):shutil.copyfile(asset(root,f[k]),target/Path(f[k]).name);f[k]=Path(f[k]).name
        f.update(pose_c2w=poses[i].tolist(),pose_valid=True);output.append(f)
    for i,log in enumerate(logs):log.update(included=i in selected,component=groups.get(i))
    report=dict(method='SIFT_RGBD_SEGMENTS_MULTI_EDGE_CONSENSUS',frames=logs,tracked=len(output),total=len(frames),
                elapsed_s=time.time()-started,connections=connections,quality_gate=quality,
                selected_origin_frame=frames[selected[0]]['id'] if selected else None,
                components=[dict(id=g,frames=[frames[i]['id'] for i in ix]) for g,ix in components.items()],
                limitations=['相对轨迹可能漂移，尚无全局闭环优化。','多帧一致性不等于绝对精度，地图对齐前不能用于旧图替换。'])
    manifest={**m,'kind':'posed_rgbd','name':'RGB-D 局部轨迹','world_convention':'camera_local','ground_z_m':0.,'map_image':'map.png',
        'pose_status':'LOCAL_CAMERA_TRAJECTORY','map_alignment':dict(accepted=False,reason='局部坐标，尚未与旧地图对齐'),
        'frames':output,'trajectory_report':report}
    atomic(target/'manifest.json',manifest);atomic(target/'quality_report.json',quality)
    return manifest


def align_anchors(root,target,spec,cancelled=lambda:False):
    m,image,meta=load_dataset(root,require_map=False)
    if m.get('world_convention')!='camera_local':raise ValueError('仅对齐独立局部轨迹')
    if spec.get('map_fingerprint')!=fingerprint(image,meta):raise ValueError('锚点与选择的旧地图不一致')
    if spec.get('world_convention')!='xy_ground_z_up' or spec.get('reference_region')!='stable_context':raise ValueError('锚点必须来自旧地图稳定上下文，坐标为 XY 地面、Z 向上')
    ground=float(spec.get('ground_z_m'))
    if not math.isfinite(ground):raise ValueError('缺少地图地面高度')
    byid={f['id']:f for f in m['frames']};anchors=spec.get('anchors',[]);transforms=[];ids=[];centres=[]
    for a in anchors:
        if a['frame_id'] not in byid or a['frame_id'] in ids:raise ValueError('锚点帧不存在或重复')
        local=rigid(byid[a['frame_id']]['pose_c2w']);world=rigid(a['pose_c2w']);transforms.append(world@np.linalg.inv(local));ids.append(a['frame_id']);centres.append(local[:3,3])
    if len(ids)<4:raise ValueError('至少需要 4 个不同帧的完整相机锚点，最后一个用于留出验证')
    if max(np.linalg.norm(a-b) for a in centres for b in centres)<.3:raise ValueError('锚点位置过于集中，请扩大采集基线')
    train=transforms[:-1];best=[]
    def residual(T,Q):
        pos=np.linalg.norm(T[:3,3]-Q[:3,3]);angle=math.degrees(math.acos(float(np.clip((np.trace(T[:3,:3].T@Q[:3,:3])-1)/2,-1,1))))
        return float(pos),angle
    for T in train:
        support=[i for i,Q in enumerate(train) if residual(T,Q)[0]<.15 and residual(T,Q)[1]<5]
        if len(support)>len(best):best=support
    if len(best)<3 or len(best)/len(train)<.7:raise ValueError('多个锚点的坐标变换不一致')
    R=sum(train[i][:3,:3] for i in best);u,_,vt=np.linalg.svd(R);fix=np.eye(3);fix[2,2]=np.linalg.det(u@vt)
    T=np.eye(4);T[:3,:3]=u@fix@vt;T[:3,3]=np.median([train[i][:3,3] for i in best],axis=0)
    holdout=residual(T,transforms[-1])
    if holdout[0]>.15 or holdout[1]>5:raise ValueError('留出锚点对齐未通过；保留局部结果，不映射旧地图')
    if cancelled():raise ValueError('已退出建图模式，未提交对齐结果')
    target=Path(target);target.mkdir(parents=True,exist_ok=False);shutil.copyfile(asset(root,m['map_image']),target/'map.png')
    frames=[]
    for original in m['frames']:
        f=dict(original);f['pose_c2w']=(T@rigid(f['pose_c2w'])).tolist()
        for k in ['rgb','depth']+(['mask'] if f.get('mask') else []):shutil.copyfile(asset(root,f[k]),target/Path(f[k]).name);f[k]=Path(f[k]).name
        frames.append(f)
    result={**m,'name':'已对齐 RGB-D 轨迹','world_convention':'xy_ground_z_up','ground_z_m':ground,'frames':frames,
        'pose_status':'MAP_ALIGNED','map_alignment':dict(accepted=True,method='robust_camera_anchors_with_holdout',map_from_local=T.tolist(),
            anchor_frames=ids,training_inliers=len(best),holdout_position_m=holdout[0],holdout_rotation_deg=holdout[1],source=spec.get('source','user_supplied'),quality_accepted=False)}
    atomic(target/'manifest.json',result);return result


def reference_anchors(root,reference,cancelled=lambda:False):
    m,image,meta=load_dataset(root,require_map=False);ref,rimage,rmeta=load_dataset(reference)
    if fingerprint(image,meta)!=fingerprint(rimage,rmeta):raise ValueError('参考 RGB-D 与目标栅格坐标绑定不一致')
    if m.get('synthetic')!=ref.get('synthetic'):raise ValueError('不能用合成参考对齐真实记录')
    if m.get('world_convention')!='camera_local':raise ValueError('输入不是局部轨迹')
    roi=m['suggested_roi'];references=[]
    for f in ref['frames'][::max(1,len(ref['frames'])//8)][:8]:
        if not f['pose_valid'] or 'rgb' not in f:continue
        xyz,d=features(reference,f);T=rigid(f['pose_c2w']);world=xyz@T[:3,:3].T+T[:3,3]
        outside=[]
        for p in world:
            x,y=world_to_pixel(*p[:2],meta);outside.append(not (roi[0]<=x<roi[2] and roi[1]<=y<roi[3]))
        outside=np.asarray(outside,bool);references.append((world[outside],d[outside]))
    anchors=[];attempts=[]
    for f in m['frames'][::max(1,len(m['frames'])//12)][:12]:
        if cancelled():raise ValueError('对齐取消')
        local,desc=features(root,f);pose=rigid(f['pose_c2w']);local=local@pose[:3,:3].T+pose[:3,3];best=None
        for ref_features in references:
            try:
                a,b=correspond((local,desc),ref_features);T,report=robust_rigid(a,b,min_inliers=30)
                if best is None or report['inliers']>best[1]['inliers']:best=(T,report)
            except ValueError:continue
        if best:
            anchors.append(dict(frame_id=f['id'],pose_c2w=(best[0]@pose).tolist()));attempts.append(dict(frame_id=f['id'],**best[1]))
    return dict(map_fingerprint=fingerprint(image,meta),world_convention='xy_ground_z_up',ground_z_m=ref['ground_z_m'],reference_region='stable_context',
                source='RGBD_reference_features_outside_ROI',anchors=anchors,matching=attempts)
