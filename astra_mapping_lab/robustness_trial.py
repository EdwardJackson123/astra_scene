"""Offline region rollback experiment against a frozen accepted reconstruction."""
import argparse,json,os,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from robustness import alignment_review,classify_rays,decision


def trial(baseline,out):
    from connected_map_update import setup,write,evaluate_stage,geometry_review_stage
    from scipy.spatial import cKDTree
    from plyfile import PlyData,PlyElement
    cfg=json.loads((baseline/'context.json').read_text())
    out.mkdir(parents=True,exist_ok=False)
    ctx=setup(SimpleNamespace(source=Path(cfg['source']),alignment_dir=Path(cfg['alignment']),output=out,base_map=Path(cfg['base_map']),gpu=1))
    root,source,alignment,out,base,data,registration,rows,ids,held=ctx
    from astra_scene.geometry import project
    dest=out/'artifacts';start=time.monotonic()
    pairs=np.load(alignment/'learned_alignment_pairs.npz')
    ar=alignment_review(pairs['local'],pairs['map'],pairs['view'],registration)
    write(dest/'alignment_robustness.json',ar)
    print('ALIGNMENT',ar['candidate_alignment_accepted'],ar['worst_displacement'],flush=True)
    regions=json.loads((baseline/'artifacts/context_scope.json').read_text())['regions']
    support=np.load(baseline/'assembly_support.npz');regions=[r for r in regions if r['id'] in support]
    surfaces=np.concatenate([support[r['id']] for r in regions])
    labels=np.concatenate([np.full(len(support[r['id']]),i,int) for i,r in enumerate(regions)])
    tree=cKDTree(surfaces)
    lineage=np.load(baseline/'artifacts/lineage.npz')
    old=PlyData.read(base,mmap=True)['vertex'].data
    new=PlyData.read(alignment/'aligned_masked_sh0.ply',mmap=True)['vertex'].data
    xyz=lambda v:np.column_stack([v[n] for n in ('x','y','z')])
    removed=np.ones(len(old),bool);removed[lineage['old_ids']]=False;oldids=np.flatnonzero(removed);newids=lineage['new_ids']
    oldpoints=xyz(old[oldids]);newpoints=xyz(new[newids])
    oldgroup=labels[tree.query(oldpoints,workers=4)[1]];newgroup=labels[tree.query(newpoints,workers=4)[1]]
    views=[x['aligned_view'] for x in json.loads((baseline/'boxer_protocol.json').read_text())['view_mapping']]
    if set(views)&set(held):raise ValueError('holdout leakage into decision')
    report=[];accepted=[]
    for j,region in enumerate(regions):
        oi=np.flatnonzero(oldgroup==j);ni=np.flatnonzero(newgroup==j)
        sample=lambda a:a[np.linspace(0,len(a)-1,min(2000,len(a)),dtype=int)] if len(a) else a
        oq=oldpoints[sample(oi)];nq=newpoints[sample(ni)]
        oc=classify_rays(oq,rows,views,project);nc=classify_rays(nq,rows,views,project)
        near=cKDTree(newpoints[ni]).query(oq,workers=4)[0]<.10 if len(ni) else np.zeros(len(oq),bool)
        # A same-surface replacement still requires current depth support.
        colocated=near&(oc['support']>=2)
        d=decision(oc,nc,colocated,ar['candidate_alignment_accepted'])
        report.append(dict(id=region['id'],bounds=region['bounds'],**d,old_points=len(oi),new_points=len(ni),old_samples=len(oq),new_samples=len(nq)))
        if d['accepted']:accepted.append(j)
        print('REGION',j,region['id'],d['accepted'],d['reasons'],flush=True)
    write(dest/'region_robustness.json',dict(regions=report,accepted=[regions[i]['id'] for i in accepted],training_only=True,
        holdouts_used_for_decisions=False,point_assignment='nearest connected assembly; one owner per point',
        evidence='deterministic samples, at most 2000 old and 2000 new per region',
        policy='experimental whole-region rollback; residual unverified removal is reported, not free-space certification',
        elapsed_seconds=time.monotonic()-start,published=False))
    write(dest/'capture_targets.json',dict(regions=[dict(id=r['id'],bounds=r['bounds'],reasons=r['reasons'],
        purpose='independent revisit and additional side views; check depth validity before authorizing erasure') for r in report if not r['accepted']],
        coordinate_frame='original map',absolute_metric_scale_verified=False,robot_motion_authorized=False,
        status='diagnostic targets only; overlapping detections require consolidation before route planning'))
    takeold=np.isin(oldgroup,accepted);takenew=np.isin(newgroup,accepted)
    keep=np.ones(len(old),bool);keep[oldids[takeold]]=False;retained=np.flatnonzero(keep)
    result=np.empty(len(retained)+int(takenew.sum()),dtype=old.dtype);result[:len(retained)]=old[retained]
    # Copy the original candidate's already-corrected floor colors exactly.
    previous=PlyData.read(baseline/'artifacts/scene.ply',mmap=True)['vertex'].data
    result[len(retained):]=previous[len(lineage['old_ids']):][takenew]
    PlyData([PlyElement.describe(result,'vertex')],text=False).write(dest/'scene.ply')
    write(dest/'bundle.json',dict(published=False,quality_accepted=False,experimental=True,scene='scene.ply'))
    for name in ('context_scope.json','replacement_support.npy'):
        os.link(baseline/'artifacts'/name,dest/name)
    (out/'context_refinement').mkdir();os.link(baseline/'artifacts/scene.ply',out/'context_refinement/initial.ply')
    write(out/'comparison.json',dict(previous_scene=str(baseline/'artifacts/scene.ply')))
    write(out/'context.json',dict(cfg,output=str(out)))
    evaluate_stage(ctx,include_extra_views=False);geometry_review_stage(ctx)
    from PIL import Image,ImageDraw,ImageFont
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18)
    for batch in range(0,len(held),3):
        selected=held[batch:batch+3];canvas=Image.new('RGB',(1440,390*len(selected)),(245,247,250));draw=ImageDraw.Draw(canvas)
        for row,i in enumerate(selected):
            frame=rows[i]['frame']
            for col,(path,title) in enumerate([(baseline/'artifacts/review'/(frame+'_candidate.png'),'Previous candidate'),(dest/'review'/(frame+'_candidate.png'),'Region rollback trial'),(dest/'review'/(frame+'_gt.png'),'Current RGB')]):
                draw.text((col*480+8,row*390+5),frame+' | '+title,font=font,fill=(20,35,50))
                canvas.paste(Image.open(path).resize((480,360)),(col*480,row*390+30))
        canvas.save(dest/f'robot_views_{batch//3+1}.jpg',quality=92)
    summary=json.loads((dest/'structural_review.json').read_text())['summary']
    body='<meta charset="utf-8"><title>Robustness trial</title><style>body{font:16px system-ui;max-width:1450px;margin:24px auto}img{width:100%}td,th{padding:8px;border:1px solid #ddd}table{border-collapse:collapse}</style><h1>逐区域鲁棒性实验（未切换默认流程）</h1>'
    body+=f'<p>区域通过 {len(accepted)}/{len(regions)}；配准稳定性 {ar["candidate_alignment_accepted"]}。只用训练视角决策；保持原评估区域，12 个留出视角展示，几何评估仍为原来的 6 个视角。深度无效不代表空闲。</p>'
    body+='<pre>'+json.dumps(summary,indent=2)+'</pre>'
    body+=''.join(f'<img src="robot_views_{b//3+1}.jpg">' for b in range(0,len(held),3))
    body+='<table><tr><th>区域</th><th>结果</th><th>原因</th><th>删除未证实比例</th><th>新增未支持比例</th></tr>'
    for r in report:body+=f'<tr><td>{r["id"]}</td><td>{"更新" if r["accepted"] else "保留原图"}</td><td>{", ".join(r["reasons"])}</td><td>{r["unknown_removal_fraction"]:.1%}</td><td>{r["unsupported_new_fraction"]:.1%}</td></tr>'
    (dest/'index.html').write_text(body+'</table>',encoding='utf-8')
    print('COMPLETE',len(accepted),len(regions),json.dumps(summary),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--baseline',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    trial(a.baseline,a.output)
