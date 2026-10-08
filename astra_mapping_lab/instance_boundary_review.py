"""DeepSeek reviews projected complete-instance volumes before takeover."""
import json,concurrent.futures
import numpy as np


def accepted(answer):
    return (answer.get('decision')=='update' and answer.get('scope_complete') is True
            and answer.get('change_type') in ('layout','add','move','remove')
            and len(set(answer.get('supporting_views',[])))>=2
            and not answer.get('missing_parts') and not answer.get('ai_validation_failed'))


def review(ctx,regions,assemblies,proposals,old_cache):
    import cv2
    from PIL import Image
    from semantic_router import SemanticClient
    from recording import atomic
    root,source,alignment,out,base,data,registration,rows,ids,held=ctx
    dest=out/'instance_boundary_review';dest.mkdir(exist_ok=True)
    client=SemanticClient(root,out)
    def work(region):
        name=region['id'];p=proposals[name];q=assemblies[name];images=[]
        for im in p['images'][:3]:
            i=im['view']
            if i in held:raise ValueError('held-out view used in boundary review')
            row=rows[i];path=old_cache/f'{i:03d}.npz'
            if not path.exists():continue
            before=np.load(path)['rgb'];current=np.uint8(np.clip(row['rgb']*255,0,255))
            pair=np.concatenate([before,current],axis=1).copy()
            camera=(q-row['pose'][:3,3])@row['pose'][:3,:3];camera=camera[camera[:,2]>.1]
            if len(camera)<4:continue
            uv=camera@row['K'].T;uv=uv[:,:2]/uv[:,2,None]
            h,w=before.shape[:2];uv=np.clip(uv,[-w,-h],[2*w,2*h])
            if p['evidence_side']=='current':uv[:,0]+=w
            polygon=cv2.convexHull(np.rint(uv).astype('int32'))
            cv2.polylines(pair,[polygon],True,(0,160,255),2)
            file=dest/f'{name}_{i}.jpg';Image.fromarray(pair).save(file,quality=93)
            images.append((im,file))
        context=dict(stage='complete_instance_boundary',evidence_side=p['evidence_side'],
            instructions='蓝色轮廓是实际拟接管的完整三维实例体积投影，左旧图右当前采集。请检查该侧设备顶部横梁、竖杆、底脚是否完整；窗户、窗外、墙、天花板是否明显混入。二维轮廓内的背景不自动属于实例，但明显跨到无关建筑应拒绝。只在至少两个视角支持完整设备且边界合理时 update/scope_complete=true；缺顶部或底脚、明显混入建筑时 defer/scope_complete=false，明确列 missing_parts。此阶段不再接受半个设备的局部更新。禁止猜坐标。')
        answer,audit=client.call(p,images,context,stage='complete_instance_boundary_v1')
        return dict(id=name,accepted=accepted(answer),answer=answer,call=audit,images=[str(f.relative_to(out)) for _,f in images])
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:records=list(pool.map(work,regions))
    names={r['id'] for r in records if r['accepted']}
    atomic(out/'artifacts/instance_boundary_review.json',dict(records=records,accepted=sorted(names),
        policy='complete instance or retain; no pointwise partial fallback',held_out_used=False))
    if not names:
        atomic(out/'automatic_outcome.json',dict(state='NEEDS_CAPTURE',reason='instance_boundaries_incomplete',
            detail='Complete-instance review did not accept a boundary; original map retained',published=False))
        raise ValueError('No complete instance boundaries accepted; retain original map')
    return [r for r in regions if r['id'] in names]
