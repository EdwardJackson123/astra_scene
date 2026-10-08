"""Frozen reconstruction: group dependencies and trim proven seam conflicts."""
import argparse,json,os,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np


def trial(baseline,out):
    from connected_map_update import setup,write,evaluate_stage,geometry_review_stage
    from change_groups import build_groups,removal_masks
    from robustness import classify_rays
    from scipy.spatial import cKDTree
    from plyfile import PlyData,PlyElement
    cfg=json.loads((baseline/'context.json').read_text());out.mkdir(parents=True,exist_ok=False)
    ctx=setup(SimpleNamespace(source=Path(cfg['source']),alignment_dir=Path(cfg['alignment']),output=out,base_map=Path(cfg['base_map']),gpu=1))
    root,source,alignment,out,base,data,registration,rows,ids,held=ctx
    from astra_scene.geometry import project
    from map_update import roi_weights
    dest=out/'artifacts';start=time.monotonic()
    regions=json.loads((baseline/'artifacts/context_scope.json').read_text())['regions'];support=np.load(baseline/'assembly_support.npz')
    proposals={p['id']:p for p in json.loads((baseline/'analysis/evidence.json').read_text())['proposals']}
    reviews={p['id']:p for p in json.loads((baseline/'analysis/ai.json').read_text())['items']}
    floor=json.loads((alignment/'artifacts/floor.json').read_text())['floor_z'];masks=np.load(baseline/'boxer_sam_masks.npz')
    groups=build_groups(regions,support,proposals,reviews,masks,floor);write(dest/'change_groups.json',groups)
    print('GROUPS',len(regions),len(groups['groups']),len(groups['edges']),flush=True)
    lineage=np.load(baseline/'artifacts/lineage.npz');oldids=lineage['old_ids'];newids=lineage['new_ids']
    previous=PlyData.read(baseline/'artifacts/scene.ply',mmap=True)['vertex'].data
    xyz=lambda v:np.column_stack([v[n] for n in ('x','y','z')])
    op=xyz(previous[:len(oldids)]);np_=xyz(previous[len(oldids):]);tree=cKDTree(np_)
    d=tree.query(op,workers=4)[0];assembly=cKDTree(np.concatenate([support[r['id']] for r in regions]))
    allowed=(d<.18)&(op[:,2]>floor+.18)&(roi_weights(op,data['map_metadata'],data['roi'],.001)>0)
    allowed&=assembly.query(op,workers=4)[0]<.20
    protected=np.isin(oldids,lineage['protected_old_ids']);allowed&=~protected
    oldprobe=np.flatnonzero(allowed);newprobe=np.flatnonzero((cKDTree(op).query(np_,workers=4)[0]<.18)&(np_[:,2]>floor+.18))
    views=[v['aligned_view'] for v in json.loads((baseline/'boxer_protocol.json').read_text())['view_mapping']]
    if set(views)&set(held):raise ValueError('held-out views cannot determine cleanup')
    # Collapse nearby capture locations before counting support; slow motion is not independent evidence.
    selected=[]
    for i in views:
        if all(np.linalg.norm(rows[i]['pose'][:3,3]-rows[j]['pose'][:3,3])>=.25 for j in selected):selected.append(i)
    oc=classify_rays(op[oldprobe],rows,selected,project);nc=classify_rays(np_[newprobe],rows,selected,project)
    visible=np.zeros(len(oldprobe),int);replacement=np.zeros(len(oldprobe),int)
    for i in selected:
        row=rows[i];ix,u,v,z=project(op[oldprobe],row)
        with np.load(baseline/'old_renders'/f'{i:03d}.npz') as old, np.load(baseline/'fresh_renders'/f'{i:03d}.npz') as fresh:
            valid=row['valid'][v,u]&row['keep'][v,u]
            seen=(old['alpha'][v,u]>.7)&(np.abs(old['depth'][v,u]-z)<.15)
            visible[ix]+=seen&valid
            replacement[ix]+=seen&valid&(fresh['alpha'][v,u]>.90)&(np.abs(fresh['depth'][v,u]-row['depth'][v,u])<.08)&(np.abs(fresh['depth'][v,u]-z)<.08)
    axes=np.sort(np.exp(np.column_stack([previous['scale_'+str(k)][len(oldids)+newprobe] for k in range(3)])),axis=1)
    rmold,_=removal_masks(oc['clear'],oc['support'],replacement,visible,np.zeros(len(oldprobe)))
    _,rmnew=removal_masks(nc['clear'],nc['support'],np.zeros(len(newprobe)),np.zeros(len(newprobe)),axes[:,1])
    keep=np.ones(len(previous),bool);keep[oldprobe[rmold]]=False;keep[len(oldids)+newprobe[rmnew]]=False
    assert keep[:len(oldids)][protected].all()
    candidate=previous[keep];PlyData([PlyElement.describe(candidate,'vertex')],text=False).write(dest/'scene.ply')
    np.savez_compressed(dest/'cleanup_evidence.npz',old_probe_ids=oldids[oldprobe],new_probe_ids=newids[newprobe],old_clear=oc['clear'],old_support=oc['support'],old_visible=visible,replacement=replacement,new_clear=nc['clear'],new_support=nc['support'],removed_old=oldids[oldprobe[rmold]],removed_new=newids[newprobe[rmnew]])
    write(dest/'cleanup.json',dict(old_probes=len(oldprobe),new_probes=len(newprobe),removed_old=int(rmold.sum()),removed_new=int(rmnew.sum()),
        old_verified_clear=int((rmold&(oc['clear']>=4)).sum()),old_colocated_replaced=int((rmold&(replacement>=3)).sum()),
        train_views=selected,held_out_used=False,restored_stale_points=0,geometry_or_color_optimization=False,
        original_scope_preserved=True,all_retained_points_byte_identical=True,elapsed_seconds=time.monotonic()-start))
    write(dest/'bundle.json',dict(published=False,quality_accepted=False,experimental=True,scene='scene.ply'))
    for name in ('context_scope.json','replacement_support.npy'):os.link(baseline/'artifacts'/name,dest/name)
    (out/'context_refinement').mkdir();os.link(baseline/'artifacts/scene.ply',out/'context_refinement/initial.ply')
    write(out/'comparison.json',dict(previous_scene=str(baseline/'artifacts/scene.ply')))
    write(out/'context.json',dict(cfg,output=str(out)))
    evaluate_stage(ctx,include_extra_views=False);geometry_review_stage(ctx)
    from PIL import Image,ImageDraw,ImageFont
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18)
    for batch in range(0,len(held),3):
        canvas=Image.new('RGB',(1440,1170),(245,247,250));draw=ImageDraw.Draw(canvas)
        for row,i in enumerate(held[batch:batch+3]):
            frame=rows[i]['frame']
            for col,(directory,suffix,title) in enumerate([(baseline/'artifacts/review','candidate','Previous'),(dest/'review','candidate','Seam cleanup'),(dest/'review','gt','Current RGB')]):
                draw.text((col*480+8,row*390+5),frame+' | '+title,font=font,fill=(20,35,50));canvas.paste(Image.open(directory/(frame+'_'+suffix+'.png')).resize((480,360)),(col*480,row*390+30))
        canvas.save(dest/f'robot_views_{batch//3+1}.jpg',quality=93)
    print('DONE',json.dumps(json.loads((dest/'cleanup.json').read_text())),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--baseline',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();trial(a.baseline,a.output)
