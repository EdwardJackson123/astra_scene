"""Durable server-only post-reconstruction workflow; never publishes a map."""
from gpu_config import gpu_index
import hashlib, json, os, shutil, subprocess, sys, time
from pathlib import Path
from recording import atomic
from stage_runtime import run as bounded_run
from grid_pipeline import DEFAULT_BACKEND
from instance_policy import LEGACY_FUSION_POLICY,FUSION_POLICIES

LAB=Path(__file__).resolve().parent
PROFILE='geometry_device_floor_v2_grid'


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def source_identity(source):
    from replay import asset
    from artifact_cache import signature
    manifest=json.loads((source/'manifest.json').read_text(encoding='utf-8'))
    names={f[k] for f in manifest['frames'] for k in ('rgb','depth','mask') if f.get(k)}
    if manifest.get('map_image'):names.add(manifest['map_image'])
    return dict(manifest=sha256(source/'manifest.json'),assets=signature({name:sha256(asset(source,name)) for name in sorted(names)}))


def select_map(source, registry):
    from replay import read_dataset, fingerprint
    m,image,meta=read_dataset(source)
    match=[r for r in registry['maps'] if r['map_fingerprint']==fingerprint(image,meta)]
    if len(match)!=1:raise ValueError('地图快照未唯一匹配已配置的原地图；禁止套用其他地图对齐或融合')
    selected=match[0]
    if sha256(selected['base_map'])!=selected['base_sha256']:raise ValueError('原地图版本已变化；需重新登记，禁止沿用旧配准')
    return selected


def selected_source(source, package, target, roi):
    """Exactly the views/poses used by this reconstruction, not a previous run."""
    from replay import asset
    m=json.loads((source/'manifest.json').read_text())
    spec=json.loads((package/'package.json').read_text())
    byid={f['id']:f for f in m['frames']};frames=[]
    target.mkdir(parents=True,exist_ok=False)
    def link(name):
        from automatic_source import canonical_name
        p=asset(source,name)
        # Different subdirectories often contain identical frame basenames.
        q=target/canonical_name(name)
        if not q.exists():os.link(p,q)
        return q.name
    for entry in spec['frames']:
        f=dict(byid[entry['frame_id']]);f['reconstruction_split']=entry['split']
        for field in ('rgb','depth','mask'):
            if f.get(field):f[field]=link(f[field])
        if not f.get('mask'):raise ValueError('新版自动融合要求人物遮罩，不能静默使用未遮罩观测')
        frames.append(f)
    if sum(f['reconstruction_split']=='train' for f in frames)<20 or sum(f['reconstruction_split']=='eval' for f in frames)<6:
        raise ValueError('地图更新需要至少 20 个训练视角、6 个留出视角；局部重建仍保留')
    if any(not f['id'].isdigit() for f in frames):raise ValueError('定位适配器需要数值帧 ID')
    # Downstream map adapters share the explicit map.png snapshot contract.
    # Frame assets remain namespaced, so they cannot collide with this name.
    os.link(asset(source,m['map_image']),target/'map.png')
    m.update(frames=frames,map_image='map.png',suggested_roi=roi)
    atomic(target/'manifest.json',m)


def execute(context, notify, runner=bounded_run, phase='all', checkpointing=True):
    from artifact_cache import Checkpoints,signature
    cfg=json.loads(Path(context).read_text());out=Path(cfg['output']);alignment=Path(cfg['alignment'])
    code_root=Path(cfg.get('code_root',LAB))
    py=cfg['python'];game=cfg['game_python']
    common=['--source',cfg['source'],'--alignment-dir',str(alignment),'--base-map',cfg['base_map'],'--output',str(out),'--gpu',str(gpu_index())]
    cache=['--render-cache',str(out/'old_renders')]
    early=common+['--preflight']
    unified=(code_root/'grid_pipeline.py').exists()
    grid_script='grid_pipeline.py' if unified else 'navigation_candidate.py'
    grid_outputs=[out/'artifacts/navigation_candidate'/name for name in ('map.png','map.yaml','navigation_update.json','navigation_candidate.zip')]
    if (code_root/'enclosed_grid_repair.py').exists():
        grid_outputs += [out/'artifacts/navigation_candidate'/name for name in ('hole_repair.json','hole_repair_evidence.npz')]
    steps=[
        ('MAP_LOCALIZING','early',[py,str(LAB/'update_localize.py')],[alignment/'anchors/localization.json']),
        ('MAP_MATCHING','early',[py,str(LAB/'update_matches.py')],[alignment/'learned_alignment_pairs.npz']),
        ('MAP_ALIGNING','early',[py,str(LAB/'update_alignment.py')],[alignment/'similarity_alignment.json',alignment/'aligned_frames.json']),
        ('MAP_EVIDENCE','early',[py,str(LAB/'update_prepare.py'),'--context',str(context)],[out/'evidence_points.npz',out/'artifacts/supported_depth_points.npy',alignment/'artifacts/floor.json']),
        ('BOXER_PREPARING','early',[py,str(LAB/'boxer_rgbd.py'),'--phase','prepare',*early,*cache],[out/'boxer_views.npz',out/'old_renders.npz',out/'boxer_protocol.json']),
        ('BOXER_DETECTING','early',[game,str(LAB/'boxer_rgbd.py'),'--phase','detect','--output',str(out),'--gpu',str(gpu_index())],[out/'boxer_native_game_v1/boxer_boxes.json']),
        ('REGION_REVIEW','early',[py,str(LAB/'boxer_rgbd.py'),'--phase','regions',*early,*cache],[out/'analysis/ai.json',out/'analysis/evidence.json',out/'component_support.npz']),
        ('APPLY_ALIGNMENT','late',[py,str(LAB/'update_alignment.py')],[alignment/'aligned_masked_sh0.ply']),
        ('MAP_BASELINE','late',[py,str(LAB/'update_prepare.py'),'--context',str(context)],[alignment/'artifacts/metrics.json']),
        ('REGION_SEGMENTING','late',[game,str(LAB/'boxer_rgbd.py'),'--phase','segment',*common,*cache],[out/'boxer_sam_masks.npz',out/'boxer_sam.json']),
        ('PROTECTED_FUSION','late',[py,str(LAB/'boxer_rgbd.py'),'--phase','fuse','--fusion-mode','geometry','--build-only',*common,*cache],[out/'artifacts/scene.ply',out/'artifacts/fusion.json',out/'artifacts/lineage.npz',out/'artifacts/takeover.json']),
        ('MAP_EVALUATING','late',[py,str(LAB/'boxer_rgbd.py'),'--phase','evaluate',*common],[out/'artifacts/summary.json',out/'artifacts/metrics.json']),
        ('MAP_GEOMETRY_REVIEW','late',[py,str(LAB/'boxer_rgbd.py'),'--phase','geometry-review',*common],[out/'artifacts/structural_review.json',out/'artifacts/bundle.json']),
        ('MAP_GRID_CANDIDATE','late',[py,str(LAB/grid_script),'--context',str(context)],grid_outputs),
        ('MAP_REPORT','late',[py,str(LAB/'automatic_report.py'),'--output',str(out)],[out/'artifacts/index.html']),
    ]
    env=os.environ.copy();env['ASTRA_UPDATE_CONTEXT']=str(context);env['PYTHONUNBUFFERED']='1'
    logs=out/'logs';logs.mkdir(parents=True,exist_ok=True)
    checkpoints=Checkpoints(out/'checkpoints')
    code=signature({p.name:sha256(p) for p in code_root.glob('*.py')}) if checkpointing else ''
    if checkpointing and cfg.get('code_snapshot_sha256') and code!=cfg['code_snapshot_sha256']:
        raise ValueError('任务代码快照被修改，拒绝混用版本')
    identity=dict(profile=cfg.get('profile'),source=source_identity(Path(cfg['source'])) if checkpointing else '',
                  base=cfg.get('base_sha256'),roi=cfg.get('roi'),code=code)
    identity['fusion_policy']=cfg.get('fusion_policy',LEGACY_FUSION_POLICY)
    if unified:identity['grid_backend']=cfg.get('grid_backend',DEFAULT_BACKEND)
    for stage,part,command,outputs in steps:
        if phase!='all' and phase!=part:continue
        command=[str(code_root/Path(arg).name) if Path(arg).parent==LAB and arg.endswith('.py') else arg for arg in command]
        notify(stage)
        atomic(out/'progress.json',dict(stage=stage,started_at=time.time(),profile=PROFILE))
        def action():
            if stage=='PROTECTED_FUSION' and (out/'automatic_outcome.json').exists():
                (out/'automatic_outcome.json').rename(logs/('previous_outcome_'+str(time.time_ns())+'.json'))
            timeout=float(cfg.get('stage_timeout_seconds',{}).get(stage,7200))
            with (logs/(stage.lower()+'.log')).open('a') as log:
                runner(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,cwd=code_root,timeout=timeout)
                if stage=='REGION_REVIEW':
                    runner([py,str(code_root/'boxer_rgbd.py'),'--phase','retry-review',*early,*cache],env=env,stdout=log,stderr=subprocess.STDOUT,check=True,cwd=code_root,timeout=timeout)
        mark={**identity,'stage':stage,'fresh':cfg.get('fresh_sha256') if part=='late' else None}
        if checkpointing:
            timing=checkpoints.run(stage,mark,outputs,action)
            atomic(logs/(stage.lower()+'.timing.json'),timing)
        else:action()


def initialize(cloud,key,source,package):
    from artifact_cache import signature
    p=cloud.directory(key);spec=json.loads((p/'request.json').read_text())
    from grid_pipeline import BACKENDS,DEFAULT_BACKEND
    backend=spec.get('grid_backend',DEFAULT_BACKEND)
    if backend not in BACKENDS:raise ValueError('unknown grid backend')
    fusion=spec.get('fusion_policy',LEGACY_FUSION_POLICY)
    if fusion not in FUSION_POLICIES:raise ValueError('unknown fusion policy')
    config=LAB/'map_update_registry.json';registry=json.loads(config.read_text());selected=select_map(source,registry)
    roi=spec.get('roi') or json.loads((source/'manifest.json').read_text())['suggested_roi']
    root=Path(registry['scene_root']);out=root/'runs'/('automatic_update_'+key)
    context=out/'context.json'
    if context.exists():
        cfg=json.loads(context.read_text())
        if cfg['profile']!=PROFILE or cfg['input_source_sha256']!=sha256(source/'manifest.json') or cfg['roi']!=roi or cfg.get('grid_backend','rgbd')!=backend or cfg.get('fusion_policy',LEGACY_FUSION_POLICY)!=fusion:
            raise ValueError('任务参数或源数据已变化，请建立新版本')
        return context,registry
    out.mkdir(parents=True,exist_ok=False);snapshot=out/'code';snapshot.mkdir()
    for file in LAB.glob('*.py'):shutil.copyfile(file,snapshot/file.name)
    shutil.copyfile(config,snapshot/config.name)
    target=p/'update_source';selected_source(source,package,target,roi)
    alignment=out/'alignment';alignment.mkdir()
    cfg={**registry['runtime'],**selected,'output':str(out),'alignment':str(alignment),'source':str(target),
         'fresh_ply':None,'roi':roi,'profile':PROFILE,'grid_backend':backend,'fusion_policy':fusion,'job_id':key,'code_root':str(snapshot),
         'code_snapshot_sha256':signature({p.name:sha256(p) for p in snapshot.glob('*.py')}),
         'input_source_sha256':sha256(source/'manifest.json')}
    atomic(context,cfg)
    return context,registry


def preflight(cloud,key,source,package):
    from context_rgbd import nominate
    context,registry=initialize(cloud,key,source,package);cfg=json.loads(context.read_text());out=Path(cfg['output'])
    execute(context,lambda stage:cloud.update(key,state=stage),phase='early')
    proposals=json.loads((out/'analysis/evidence.json').read_text())['proposals']
    ai=json.loads((out/'analysis/ai.json').read_text());chosen=nominate(ai['items'],proposals)
    # Identical to the existing ownership size rule, but avoid training a job
    # whose only nominated regions would all be discarded after training.
    small={p['id'] for p in proposals if p.get('oriented_box',{}).get('size') and max(p['oriented_box']['size'])<.4}
    chosen=[name for name in chosen if name not in small]
    report=dict(eligible_regions=len(chosen),nominated_ids=chosen,training_started=False,
                small_regions_skipped=sorted(small),
                scale_conflict=json.loads((Path(cfg['alignment'])/'similarity_alignment.json').read_text())['metric_scale_conflict'])
    atomic(out/'preflight.json',report)
    if not chosen:
        relevant=[x for x in ai['items'] if x.get('id') not in small]
        failures=any(x.get('ai_validation_failed') for x in relevant)
        uncertain=any(x['decision']=='defer' for x in relevant)
        outcome='SEMANTIC_UNAVAILABLE' if failures else ('NEEDS_CAPTURE' if uncertain else 'NO_SUPPORTED_CHANGE')
        message='判断服务异常，当前保留原图，可稍后重试。' if failures else ('证据或覆盖不足，当前保留原图，请补采变化区域。' if uncertain else '未发现足够支持的重要变化，保留原图。')
        (out/'artifacts/index.html').write_text('<meta charset="utf-8"><h1>自动检查结果</h1><p>'+message+'</p>',encoding='utf-8')
        cloud.update(key,state='READY',map_update=dict(state=outcome,quality_accepted=False,published=False,
            result_url=registry['results_url'].rstrip('/')+'/'+out.name+'/index.html'))
        return False
    return True


def run(cloud,key,source,result):
    cloud.update(key,map_update=dict(state='STARTING',profile=PROFILE,quality_accepted=False,published=False))
    context,registry=initialize(cloud,key,source,cloud.directory(key)/'package')
    cfg=json.loads(context.read_text());out=Path(cfg['output'])
    # Legacy callers can still enter after training; normal service preflights first.
    if not (out/'preflight.json').exists():
        if not preflight(cloud,key,source,cloud.directory(key)/'package'):return cloud.state(key)['map_update']
    cfg.update(fresh_ply=str((cloud.directory(key)/'package'/result['ply']).resolve()),apply_only=True)
    cfg['fresh_sha256']=sha256(cfg['fresh_ply']);atomic(context,cfg)
    def notify(stage):cloud.update(key,state=stage,map_update=dict(state=stage,profile=PROFILE,output=str(out),published=False,quality_accepted=False))
    try:
        execute(context,notify,phase='late')
    except subprocess.CalledProcessError:
        outcome=out/'automatic_outcome.json'
        progress=json.loads((out/'progress.json').read_text()) if (out/'progress.json').exists() else {}
        if not outcome.exists() or progress.get('stage')!='PROTECTED_FUSION':raise
        detail=json.loads(outcome.read_text())
        (out/'artifacts/index.html').write_text('<meta charset="utf-8"><h1>本次保留原地图</h1><p>变化区域完整性不足，请补采缺失或遮挡位置后重试。独立重建结果已保留。</p>',encoding='utf-8')
        return dict(state=detail['state'],profile=PROFILE,result_url=registry['results_url'].rstrip('/')+'/'+out.name+'/index.html',quality_accepted=False,published=False)

    review=json.loads((out/'artifacts/structural_review.json').read_text());url=registry['results_url'].rstrip('/')+'/'+out.name+'/index.html'
    grid=json.loads((out/'artifacts/navigation_candidate/navigation_update.json').read_text())
    return dict(state=('CANDIDATE_READY' if review['passed'] else 'QUALITY_FAILED'),profile=PROFILE,result_url=url,output=str(out),candidate_url=url.removesuffix('index.html')+'scene.ply',
                structural_review_passed=review['passed'],quality_accepted=False,published=False,navigation_updated=False,
                navigation_candidate_ready=True,navigation_candidate_url=url.removesuffix('index.html')+'navigation_candidate/index.html',
                reference_database_updated=False,grid_backend=grid.get('grid_backend',DEFAULT_BACKEND),fusion_policy=cfg.get('fusion_policy',LEGACY_FUSION_POLICY),
                navigation_assessment=grid.get('navigation_assessment',{'state':'NOT_VERIFIED','accepted':False}))
