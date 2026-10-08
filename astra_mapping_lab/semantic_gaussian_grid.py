"""Refine a 3DGS grid with measured depth and bounded semantic contour operations."""
from gpu_config import gpu_index
import argparse,json,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from grid_common import classes, constrained_result, fragmentation


def close_object_gaps(occupied,scope,solid=False):
    import cv2
    from scipy.ndimage import binary_fill_holes,label
    seed=occupied&scope
    closed=cv2.morphologyEx(seed.astype(np.uint8),cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))).astype(bool)
    added=closed&scope&~occupied
    if not solid:
        added &= ~(binary_fill_holes(seed)&~seed)
    if solid:
        filled=binary_fill_holes(seed|added)&~(seed|added);labels,_=label(filled);n=np.bincount(labels.ravel())
        added|=filled&(n[labels]<=4)&scope
    # This operator never erases an occupied pixel or writes outside this object.
    return occupied|added,added


def run(context,seed,dest):
    from connected_map_update import setup,write
    from grid_shape_semantics import review
    from PIL import Image,ImageDraw,ImageFont
    from scipy.ndimage import binary_dilation
    import cv2,yaml
    started=time.monotonic();cfg=json.loads(Path(context).read_text());fusion=Path(cfg['output']);dest=Path(dest);dest.mkdir(parents=True,exist_ok=False)
    ctx=setup(SimpleNamespace(source=Path(cfg['source']),alignment_dir=Path(cfg['alignment']),output=dest/'inputs',base_map=Path(cfg['base_map']),gpu=gpu_index()))
    root,source,alignment,_,base,data,registration,rows,ids,held=ctx
    from astra_scene.geometry import grid_cells,backproject,evidence
    from artifact_cache import digest
    seed_report=json.loads((seed/'navigation_update.json').read_text());views=seed_report['render_views']
    if set(views)&set(held):raise ValueError('holdout leakage')
    meta=data['map_metadata'];original=np.array(Image.open(seed/'original_map.png').convert('L'));candidate=np.array(Image.open(seed/'map.png').convert('L'));shape=original.shape
    allowed=np.array(Image.open(seed/'allowed_scope.png'))>0;floor=json.loads((alignment/'artifacts/floor.json').read_text())['floor_z']
    def cells(p):
        uv,ok=grid_cells(p,meta,shape);ix=np.flatnonzero(ok);ok[ix]&=allowed[uv[ix,1],uv[ix,0]];return uv,ok
    hit_views=np.zeros(shape,np.int16);ground_views=np.zeros(shape,np.int16)
    for i in views:
        p,_=backproject(rows[i],stride=2);uv,ok=cells(p)
        m=ok&(p[:,2]>=floor+.15)&(p[:,2]<=floor+1.8);u=np.unique(uv[m],axis=0);hit_views[u[:,1],u[:,0]]+=1
        m=ok&(np.abs(p[:,2]-floor)<.10);u=np.unique(uv[m],axis=0);ground_views[u[:,1],u[:,0]]+=1
    measured=hit_views>=3;weak=hit_views>=1
    # A one-cell clearance-only guard accounts for discretization at obstacle
    # edges. It does not inflate the exported occupied obstacles globally.
    veto=binary_dilation(weak,structure=np.ones((3,3),bool))&allowed
    cache=np.load(seed/'model_surfaces.npz');points=cache['points'][cache['selected_indices']]
    hit,free,_=evidence(points,rows,views,.10)
    contradicted=(free>=4)&(hit==0);uv,ok=cells(points);flat=uv[ok,1]*shape[1]+uv[ok,0]
    total=np.bincount(flat,minlength=original.size).reshape(shape)
    rejected=np.bincount(flat,weights=contradicted[ok],minlength=original.size).reshape(shape)
    disproven=(total>0)&(total==rejected)&~weak&(ground_views>=3)
    original_occ,_,_=classes(original,meta);seed_occ,seed_free,_=classes(candidate,meta)
    desired_free=original_occ&seed_free&allowed
    confirmed_free=desired_free&~veto&(ground_views>=3)
    # Do not inherit unvalidated deletion or retain physically contradicted new speckles.
    clean_model=seed_occ&~disproven
    occupied=(original_occ&~confirmed_free)|clean_model|measured
    # Only remove model-only new occupied pixels, never legacy occupied pixels here.
    removed_model_speckles=seed_occ&~original_occ&disproven
    occupied[removed_model_speckles]=False
    before_shape=occupied.copy()
    regions=json.loads((fusion/'artifacts/context_scope.json').read_text())['regions'];selected={r['id'] for r in regions}
    proposals=[p for p in json.loads((fusion/'analysis/evidence.json').read_text())['proposals'] if p['id'] in selected and p['evidence_side']=='current']
    if any(im['view'] in held for p in proposals for im in p['images']):raise ValueError('semantic holdout leakage')
    assembly=np.load(fusion/'assembly_support.npz');object_scopes={};preview_dir=dest/'shape_previews';preview_dir.mkdir()
    review_proposals=[];skipped=[]
    for item in proposals:
        p=assembly[item['id']];uv,ok=cells(p);uv=np.unique(uv[ok],axis=0).astype(np.int32)
        if len(uv)<3:skipped.append(item['id']);continue
        scope=np.zeros(shape,np.uint8);cv2.fillConvexPoly(scope,cv2.convexHull(uv),1);scope=scope.astype(bool)&allowed;object_scopes[item['id']]=scope
        b,badd=close_object_gaps(before_shape,scope,solid=False);c,cadd=close_object_gaps(before_shape,scope,solid=True)
        if not cadd.any() and not badd.any():skipped.append(item['id']);continue
        yy,xx=np.where(scope);box=(max(0,int(xx.min())-5),max(0,int(yy.min())-5),min(shape[1],int(xx.max())+6),min(shape[0],int(yy.max())+6))
        preview=Image.new('RGB',(1200,440),'white');draw=ImageDraw.Draw(preview);font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18)
        contours,_=cv2.findContours(scope.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        for j,(a,add,title) in enumerate([(before_shape,np.zeros(shape,bool),'A Existing grid'),(b,badd,'B Close small gaps'),(c,cadd,'C Also fill small holes')]):
            rgb=np.repeat(np.where(a,0,255).astype(np.uint8)[:,:,None],3,axis=2);rgb[add]=[220,30,70];cv2.drawContours(rgb,contours,-1,(40,120,220),1)
            im=Image.fromarray(rgb).crop(box);im.thumbnail((380,380),Image.Resampling.NEAREST)
            # Enlarge while preserving square cells and aspect ratio.
            scale=min(380/im.width,380/im.height);im=im.resize((round(im.width*scale),round(im.height*scale)),Image.Resampling.NEAREST)
            draw.text((j*400+5,8),title,font=font,fill='black');preview.paste(im,(j*400+5,45))
        name=item['id']+'.png';preview.save(preview_dir/name);review_proposals.append(dict(item,grid_preview=name))
    sem=review(root,fusion,review_proposals,dest);sem['no_op_or_missing_scope']=skipped;write(dest/'shape_semantics.json',sem);print('SEMANTICS',len(sem['items']),flush=True)
    edits=[];shaped=np.zeros(shape,bool)
    for item in sem['items']:
        if item['operation']!='close_small_gaps' or item.get('failed'):continue
        scope=object_scopes[item['id']]
        # The hull limits the operation; it is never filled into occupancy itself.
        _,added=close_object_gaps(before_shape,scope,solid=item['footprint']=='solid_body')
        added &= ~occupied
        occupied|=added;shaped|=added;edits.append(dict(id=item['id'],cells=int(added.sum()),footprint=item['footprint']))
    proposed=original.copy();negate=meta.get('negate',0)
    proposed[confirmed_free&~occupied]=1 if negate else 254;proposed[occupied&allowed]=255 if negate else 0
    result,counts=constrained_result(original,proposed,allowed,meta)
    from enclosed_grid_repair import apply_to_grid
    Image.fromarray(result).save(dest/'before_hole_repair.png')
    result,hole_added,hole_report=apply_to_grid(result,meta,allowed,hit_views,ground_views)
    result,counts=constrained_result(original,result,allowed,meta)
    write(dest/'hole_repair.json',hole_report)
    np.savez_compressed(dest/'hole_repair_evidence.npz',sensor_hit_views=hit_views,ground_views=ground_views,added=hole_added,train_views=np.asarray(views))
    # Final occupancy is model-primary; depth-only conflicts become UNKNOWN,
    # never silently turn into traversable free space.
    from model_primary_contour import refine
    surface_evidence=np.load(seed/'grid_evidence.npz')
    Image.fromarray(result).save(dest/'before_model_primary.png')
    result,model_preview,contour_report,contour_edits=refine(result,original,allowed,meta,
        surface_evidence['occupied'],shaped,hole_added,hit_views,ground_views,
        surface_evidence['floor_views'],surface_evidence['erasure_veto'],object_scopes)
    result,counts=constrained_result(original,result,allowed,meta)
    Image.fromarray(model_preview).save(dest/'model_occupancy_preview_NOT_NAVIGATION.png')
    write(dest/'contour_cleanup.json',contour_report)
    np.savez_compressed(dest/'contour_cleanup_evidence.npz',**contour_edits)
    held_obstacle=np.zeros(shape,bool);held_floor=np.zeros(shape,bool)
    for i in held:
        p,_=backproject(rows[i],stride=2);uv,ok=cells(p);m=ok&(p[:,2]>=floor+.15)&(p[:,2]<=floor+1.8);held_obstacle[uv[m,1],uv[m,0]]=True
        m=ok&(np.abs(p[:,2]-floor)<.10);held_floor[uv[m,1],uv[m,0]]=True
    out_occ,out_free,out_unknown=classes(result,meta);_,original_free,_=classes(original,meta);cleared=original_occ&out_free;conflict=cleared&held_obstacle
    report=dict(**counts,source='3DGS model-primary occupancy + unknown conflict cells + bounded supported contour repair',
        model_seed=str(seed),source_ply_sha256=seed_report['source_ply_sha256'],
        sensor_protected_previous_deletions=int((desired_free&~confirmed_free).sum()),
        sensor_rejected_model_additions=int(removed_model_speckles.sum()),measured_obstacle_cells=int(measured.sum()),
        free_to_unknown=int((original_free&out_unknown&allowed).sum()),unknown_cells_in_scope=int((out_unknown&allowed).sum()),
        semantic_additions=int(shaped.sum()),semantic_edits=edits,hole_repair=hole_report,contour_cleanup=contour_report,
        fragmentation_seed=fragmentation(seed_occ,allowed),fragmentation_before_shape=fragmentation(before_shape,allowed),fragmentation_final=fragmentation(out_occ,allowed),
        cleared_with_holdout_obstacle=int(conflict.sum()),cleared_with_holdout_ground=int((cleared&held_floor).sum()),
        candidate_audit_passed=not bool(conflict.any()),held_out_used_for_selection=False,
        published=False,navigation_deployment_accepted=False,metric_scale_conflict=registration.get('metric_scale_conflict',True),
        policy='3DGS model-primary occupancy; depth-only conflicts become unknown; measured depth vetoes unsafe clearances; bounded supported gap repair must preserve free-space connectivity; model preview is not navigation',
        limitations=['conservative depth guard may retain stale obstacles','new contour occupancy may narrow passages; not robot-clearance validated','same-capture development audit, not independent revisit'],elapsed_s=time.monotonic()-started)
    for name,a in [('map.png',result),('original_map.png',original),('before_shape.png',np.where(allowed,np.where(before_shape,255 if negate else 0,original),original).astype(np.uint8)),('allowed_scope.png',np.uint8(allowed)*255)]:Image.fromarray(a).save(dest/name)
    # Save the true pre-shape grid, including confirmed frees, for honest ablation.
    pre=original.copy();pre[confirmed_free&~before_shape]=1 if negate else 254;pre[before_shape&allowed]=255 if negate else 0;Image.fromarray(pre).save(dest/'before_shape.png')
    config={k:meta[k] for k in ('resolution','origin','negate','occupied_thresh','free_thresh','mode') if k in meta};config['image']='map.png'
    (dest/'map.yaml').write_text(yaml.safe_dump(config,sort_keys=False));write(dest/'navigation_update.json',report)
    np.savez_compressed(dest/'grid_evidence.npz',sensor_hit_views=hit_views,ground_views=ground_views,veto=veto,measured=measured,disproven=disproven,confirmed_free=confirmed_free,shaped=shaped,hole_added=hole_added,held_obstacle=held_obstacle,conflict=conflict)
    print('RESULT',json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--context',type=Path,required=True);p.add_argument('--seed',type=Path,required=True);p.add_argument('--destination',type=Path,required=True);a=p.parse_args();run(a.context,a.seed,a.destination)
