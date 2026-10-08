"""Experimental old-surface retirement, separated from current free-space proof."""
import argparse,json,os,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np


def retire_cells(total,cleared,stable,semantic,current_obstacle,ground_views,old_occupied):
    ratio=np.divide(cleared,total,out=np.zeros_like(cleared,dtype=float),where=total>0)
    retired=old_occupied&semantic&(total>=2)&(ratio>=.85)&(stable==0)
    # Retirement is a change hypothesis; traversability requires every observed
    # old sample to be explained. The remaining 15% may contain a thin obstacle.
    free=retired&~current_obstacle&(ground_views>=3)&(cleared==total)
    unknown=retired&~current_obstacle&~free
    return free,unknown,retired


def run(context,dest,previous,all_training_views=False,reference=None):
    from connected_map_update import setup,write
    from semantic_grid_review import review
    from navigation_candidate import classes,constrained_result
    from PIL import Image,ImageDraw,ImageFont
    from scipy.spatial import cKDTree
    import yaml,zipfile
    cfg=json.loads(Path(context).read_text());fusion=Path(cfg['output']);dest=Path(dest);dest.mkdir(parents=True,exist_ok=False)
    ctx=setup(SimpleNamespace(source=Path(cfg['source']),alignment_dir=Path(cfg['alignment']),output=dest/'inputs',base_map=Path(cfg['base_map']),gpu=1))
    root,source,alignment,_,base,data,registration,rows,ids,held=ctx
    from astra_scene.geometry import backproject,voxel_fuse,grid_cells,grid_centers,evidence,project
    from astra_scene.io import independent
    from artifact_cache import digest
    meta=data['map_metadata'];original=np.array(Image.open(source/'map.png').convert('L'));shape=original.shape
    scope=json.loads((fusion/'artifacts/context_scope.json').read_text());selected={r['id'] for r in scope['regions']}
    proposals=[p for p in json.loads((fusion/'analysis/evidence.json').read_text())['proposals'] if p['id'] in selected and p['evidence_side']=='old']
    if any(im['view'] in held for p in proposals for im in p['images']):raise ValueError('semantic holdout leakage')
    sem=review(root,fusion,proposals,dest)
    # Semantic uncertainty does not negate independently measured free rays.
    # Restrict to reviewed movable objects, but require geometry below to prove retirement.
    approved={x['id'] for x in sem['items'] if x['old_state']!='present' and x['structure']=='movable' and len(x['supporting_views'])>=2 and not x.get('failed')}
    print('SEMANTIC',len(approved),'/',len(proposals),flush=True)
    allowed=np.array(Image.open(previous/'allowed_scope.png'))>0
    floor=json.loads((alignment/'artifacts/floor.json').read_text())['floor_z'];masks=np.load(fusion/'boxer_sam_masks.npz');assembly=np.load(fusion/'assembly_support.npz')
    tree=cKDTree(np.concatenate([assembly[r['id']] for r in scope['regions']]))
    protocol=json.loads((fusion/'boxer_protocol.json').read_text());views=independent(rows,[v['aligned_view'] for v in protocol['view_mapping']],.20)
    if set(views)&set(held):raise ValueError('holdout leakage')
    evidence_views=list(views)
    if all_training_views:
        # Preserve baseline independent views, then expand with training positions.
        # Semantic image selection must not limit geometric verification coverage.
        for i in ids:
            if all(np.linalg.norm(rows[i]['pose'][:3,3]-rows[j]['pose'][:3,3])>=.20 for j in evidence_views):evidence_views.append(i)
    if set(evidence_views)&set(held):raise ValueError('evidence holdout leakage')
    print('VIEWS',len(views),len(evidence_views),flush=True)
    def cells(points):
        uv,ok=grid_cells(points,meta,shape);accepted=np.flatnonzero(ok);ok[accepted]&=allowed[uv[accepted,1],uv[accepted,0]];return uv,ok
    current=[];oldparts=[];parts_by_region={k:[] for k in approved}
    for i in evidence_views:
        row=rows[i];p,_=backproject(row,stride=2);uv,ok=cells(p);p=p[ok];p=p[(p[:,2]>=floor-.12)&(p[:,2]<=floor+1.8)]
        current.append(p)
    for i in views:
        row=rows[i]
        with np.load(fusion/'old_renders'/f'{i:03d}.npz') as old:
            rr=dict(row,depth=old['depth'],valid=(old['alpha']>.85)&(old['depth']>.4)&(old['depth']<7))
            # All historical visible obstacle surfaces contribute to the denominator.
            p,_=backproject(rr,stride=2);uv,ok=cells(p);p=p[ok];p=p[(p[:,2]>=floor+.15)&(p[:,2]<=floor+1.8)];oldparts.append(p)
            for key in approved:
                name=key+'__'+str(i)
                if name not in masks:continue
                p,_=backproject(rr,masks[name],stride=2);uv,ok=cells(p);p=p[ok];p=p[(p[:,2]>=floor+.15)&(p[:,2]<=floor+1.8)]
                if len(p):p=p[cKDTree(assembly[key]).query(p,workers=4)[0]<.18]
                parts_by_region[key].append(p)
    def vox(parts):
        p=np.concatenate(parts) if parts else np.empty((0,3))
        return voxel_fuse(p,np.zeros_like(p),.025)[0] if len(p) else p
    current=vox(current);old=vox(oldparts);print('SURFACES',len(current),len(old),flush=True)
    hit,clear,_=evidence(current,rows,evidence_views,.10);current=current[(hit>=3)&(clear<=1)]
    # Retain positive obstacle evidence already verified by the baseline's
    # independent training views, even if this cached-view subset misses it.
    prior_physical=np.load(fusion/'artifacts/supported_depth_points.npy');_,prior_ok=cells(prior_physical)
    current=np.concatenate([current,prior_physical[prior_ok]])
    obs=current[(current[:,2]>=floor+.15)&(current[:,2]<=floor+1.8)]
    # Current obstacle returns veto erasure even outside a movable-instance mask.
    occupied=np.zeros(shape,bool);uv,ok=cells(obs);occupied[uv[ok,1],uv[ok,0]]=True
    semantic=np.zeros(shape,bool)
    for key,parts in parts_by_region.items():
        p=vox(parts);uv,ok=cells(p);semantic[uv[ok,1],uv[ok,0]]=True
    support,clear,occluded=evidence(old,rows,evidence_views,.12);uv,ok=cells(old);flat=uv[ok,1]*shape[1]+uv[ok,0]
    total=np.bincount(flat,minlength=original.size).reshape(shape)
    cleared=np.bincount(flat,weights=((clear[ok]>=4)&(support[ok]==0)),minlength=original.size).reshape(shape)
    stable=np.bincount(flat,weights=(support[ok]>=2),minlength=original.size).reshape(shape)
    oc,_,_=classes(original,meta);eligible=oc&semantic&(total>=2)&(stable==0)&(cleared/np.maximum(total,1)>=.85)
    y,x=np.where(eligible);xy=grid_centers(np.column_stack([x,y]),meta,shape);ground=np.column_stack([xy,np.full(len(x),floor)])
    ground_views=np.zeros(shape,int)
    for i in evidence_views:
        ix,u,v,z=project(ground,rows[i]);row=rows[i]
        # Ground must be measured, not inferred from an absent obstacle return.
        good=row['valid'][v,u]&row['keep'][v,u]&(np.abs(row['depth'][v,u]-z)<.10)
        ground_views[y[ix],x[ix]]+=good
    free,unknown,retired=retire_cells(total,cleared,stable,semantic,occupied,ground_views,oc)
    # Preserve the earlier candidate's supported additions and unknown evidence.
    # Failure/uncertainty in semantic review must not restore stale occupancy.
    proposed=np.array(Image.open(previous/'map.png').convert('L'));negate=meta.get('negate',0);prob=(meta.get('free_thresh',.196)+meta.get('occupied_thresh',.65))/2
    proposed[unknown]=round(255*(prob if negate else 1-prob));proposed[free]=1 if negate else 254
    # Only admit additions close to the reviewed connected assembly geometry.
    additions=obs[tree.query(obs,workers=4)[0]<.18];uv,ok=cells(additions);proposed[uv[ok,1],uv[ok,0]]=255 if negate else 0
    result,counts=constrained_result(original,proposed,allowed,meta)
    # Frozen holdout audit: never used to change the produced map.
    held_conflict=np.zeros(shape,bool);held_floor=np.zeros(shape,bool)
    for i in held:
        p,_=backproject(rows[i],stride=2);uv,ok=cells(p);inside=(p[:,2]>=floor+.15)&(p[:,2]<=floor+1.8)&ok
        held_conflict[uv[inside,1],uv[inside,0]]=True
        near=(np.abs(p[:,2]-floor)<.10)&ok;held_floor[uv[near,1],uv[near,0]]=True
    _,actual_free,_=classes(result,meta);actual_cleared=oc&actual_free
    report=dict(**counts,semantic_approved=sorted(approved),semantic_regions=len(proposals),retired_old_cells=int(retired.sum()),
        retired_but_currently_occupied=int((retired&occupied).sum()),cleared_with_holdout_obstacle=int((actual_cleared&held_conflict).sum()),
        cleared_with_holdout_ground=int((actual_cleared&held_floor).sum()),held_out_used_for_selection=False,
        candidate_audit_passed=not bool((actual_cleared&held_conflict).any()),
        incomplete_retired_surface_cells=int((retired&~occupied&(cleared<total)).sum()),
        old_surface_points=len(old),measured_current_points=len(current),resolution=meta['resolution'],
        surface_source_views=views,geometric_evidence_views=evidence_views,held_out_views=held,
        policy='reviewed movable scope; >=85% old surface cleared in >=4 views and no stable surface nominates retirement only; current obstacles veto; ALL observed old samples cleared plus >=3 measured ground views required for free; incomplete retired surface becomes UNKNOWN',
        published=False,navigation_deployment_accepted=False,metric_scale_conflict=registration.get('metric_scale_conflict',True),
        limitations=['observed old-surface coverage is not complete volume clearance','sparse or occluded current obstacles may be missed','scale and robot dimensions unverified'],
        map_id=cfg.get('map_id','unspecified'),original_sha256=digest(source/'map.png'))
    report['policy_version']='surface_grid_v3_complete_observed'
    write(dest/'navigation_update.json',report)
    np.savez_compressed(dest/'cell_evidence.npz',total=total,cleared=cleared,stable=stable,semantic=semantic,current_obstacle=occupied,ground_views=ground_views,retired=retired,free=free,unknown=unknown,held_obstacle=held_conflict,held_floor=held_floor)
    for name,arr in [('map.png',result),('original_map.png',original),('allowed_scope.png',np.uint8(allowed)*255)]:Image.fromarray(arr).save(dest/name)
    mapcfg={k:meta[k] for k in ('resolution','origin','negate','occupied_thresh','free_thresh','mode') if k in meta};mapcfg['image']='map.png'
    (dest/'map.yaml').write_text(yaml.safe_dump(mapcfg,sort_keys=False))
    with zipfile.ZipFile(dest/'navigation_candidate.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name in ('map.png','map.yaml','navigation_update.json','semantic_review.json'):z.write(dest/name,name)
    diff=np.stack([original]*3,axis=-1);nc,nf,nu=classes(result,meta);diff[nc&~oc]=[230,65,55];diff[oc&nf]=[30,155,245];diff[oc&nu]=[240,170,35]
    Image.fromarray(diff).save(dest/'changes.png')
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',19);canvas=Image.new('RGB',(1800,1050),(242,246,249));draw=ImageDraw.Draw(canvas)
    roi=data['roi'];box=(int(roi[0])-10,max(0,int(roi[1])-10),min(shape[1],int(roi[2])+10),min(shape[0],int(roi[3])+20))
    for j,(im,title) in enumerate([(Image.open(previous/'map.png'),'Previous: full-column gate'),(Image.fromarray(result),'New: old-surface evidence + DeepSeek'),(Image.fromarray(diff),'Changes from original')]):
        draw.text((j*600+5,12),title,font=font,fill=(20,35,50));canvas.paste(im.convert('RGB').resize((505,595),Image.Resampling.NEAREST),(j*600,45));crop=im.crop(box);crop=crop.resize((590,int(crop.height*590/crop.width)),Image.Resampling.NEAREST);canvas.paste(crop.convert('RGB'),(j*600,650))
    canvas.save(dest/'grid_comparison.png')
    (dest/'index.html').write_text('<meta charset="utf-8"><title>旧障碍更新实验</title><style>body{font:17px system-ui;margin:25px}img{max-width:100%}</style><h1>DeepSeek + 旧表面证据栅格实验（未部署）</h1><p>红色新增，蓝色清除，黄色转未知。当前地面观测不等于完整通行体积验证。</p><pre>'+json.dumps(report,ensure_ascii=False,indent=2)+'</pre><img src="grid_comparison.png">',encoding='utf-8')
    from grid_review_report import build
    build(dest,reference or previous)
    from confirmed_grid import build as export_confirmed
    export_confirmed(dest,dest/'confirmed')
    print('RESULT',json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--context',type=Path,required=True);p.add_argument('--destination',type=Path,required=True);p.add_argument('--previous',type=Path,required=True);p.add_argument('--all-training-views',action='store_true');p.add_argument('--reference',type=Path,help='Comparison report only, never decision input');a=p.parse_args();run(a.context,a.destination,a.previous,a.all_training_views,a.reference)
