"""Conservative block regularization with an explicit topology audit."""
import argparse,json,zipfile
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from navigation_candidate import classes,constrained_result
from semantic_gaussian_grid import fragmentation


def conservative_blocks(occupied,allowed,block=2):
    if not isinstance(block,int) or block<1:raise ValueError('invalid block')
    h,w=occupied.shape
    pad=np.pad(occupied&allowed,((0,(-h)%block),(0,(-w)%block)),constant_values=False)
    pooled=pad.reshape(pad.shape[0]//block,block,pad.shape[1]//block,block).any(axis=(1,3))
    expanded=pooled.repeat(block,0).repeat(block,1)[:h,:w]
    return occupied|(expanded&allowed)


def topology_audit(before_free,after_free,resolution):
    from scipy.ndimage import label,distance_transform_edt
    labels,n=label(before_free);after_labels,_=label(after_free)
    # Compare the connectivity of pre-existing free-space cores at >=20cm
    # clearance. This does not certify clearance for a particular robot.
    core=before_free&(distance_transform_edt(before_free)*resolution>=.20)
    splits=[]
    for i in range(1,n+1):
        anchor=core&(labels==i);ids=np.unique(after_labels[anchor&after_free]);ids=ids[ids>0]
        if len(ids)>1:splits.append(dict(original_component=i,new_components=len(ids),core_cells=int(anchor.sum())))
    return dict(split_free_space_cores=splits,lost_core_cells=int((core&~after_free).sum()),reference_clearance_m=.20,robot_clearance_certified=False)


def build(source,reference,dest):
    import yaml
    source=Path(source);dest=Path(dest);dest.mkdir(parents=True,exist_ok=True)
    if any(dest.iterdir()):raise ValueError('destination contains artifacts; use a new directory')
    meta=yaml.safe_load((source/'map.yaml').read_text(encoding='utf-8'));resolution=meta['resolution']
    if abs(resolution-.05)>1e-8:raise ValueError('this experiment uses 5cm input and 10cm blocks')
    original=np.array(Image.open(source/'original_map.png'));before=np.array(Image.open(source/'map.png'));allowed=np.array(Image.open(source/'allowed_scope.png'))>0
    occ,free,unknown=classes(before,meta);structured=conservative_blocks(occ,allowed,2)
    result=before.copy();result[structured&allowed]=255 if meta.get('negate',0) else 0
    result,counts=constrained_result(original,result,allowed,meta);_,newfree,_=classes(result,meta)
    audit=topology_audit(free,newfree,resolution)
    e=np.load(source/'grid_evidence.npz');oc,_,_=classes(original,meta);cleared=oc&newfree;conflicts=cleared&e['held_obstacle']
    report=dict(**counts,method='conservative 10cm blocks inside update permission; retain 5cm map geometry',
        source=str(source.resolve()),block_size_m=.10,added_by_regularization=int((structured&~occ).sum()),
        occupied_pixels_removed=0,
        fragmentation_before=fragmentation(occ,allowed),fragmentation_after=fragmentation(structured,allowed),
        topology_audit=audit,cleared_with_holdout_obstacle=int(conflicts.sum()),
        candidate_audit_passed=not conflicts.any() and not audit['split_free_space_cores'],
        published=False,navigation_deployment_accepted=False,
        limitations=['conservative blocks may narrow gaps and overestimate obstacles','20cm free-core topology check is not robot-specific clearance','independent revisit and metric scale unverified'])
    report['candidate_audit_passed']=bool(report['candidate_audit_passed'])
    for name,a in [('map.png',result),('evidence_map.png',before),('original_map.png',original),('allowed_scope.png',np.uint8(allowed)*255)]:Image.fromarray(a).save(dest/name)
    meta['image']='map.png';(dest/'map.yaml').write_text(yaml.safe_dump(meta,sort_keys=False),encoding='utf-8')
    (dest/'navigation_update.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    semantic=json.loads((source/'shape_semantics.json').read_text(encoding='utf-8'));(dest/'shape_semantics.json').write_text(json.dumps(semantic,ensure_ascii=False,indent=2),encoding='utf-8')
    y,x=np.where(allowed);box=(max(0,int(x.min())-8),max(0,int(y.min())-8),min(result.shape[1],int(x.max())+9),min(result.shape[0],int(y.max())+9))
    h=round((box[3]-box[1])*570/(box[2]-box[0]));canvas=Image.new('RGB',(1800,h+105),'white');d=ImageDraw.Draw(canvas)
    font_path=next(p for p in [Path('C:/Windows/Fonts/msyh.ttc'),Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')] if p.exists())
    f=ImageFont.truetype(str(font_path),24);cn='msyh' in str(font_path)
    titles=['上一版：仅 3DGS','深度保护与语义复核','10cm 保守轮廓候选'] if cn else ['Previous: 3DGS only','Depth guard + semantic review','Conservative 10cm contours']
    for j,(a,title) in enumerate(zip([np.array(Image.open(Path(reference)/'map.png')),before,result],titles)):
        d.text((j*600+10,12),title,font=f,fill='black');canvas.paste(Image.fromarray(a).convert('RGB').crop(box).resize((570,h),Image.Resampling.NEAREST),(j*600+10,65))
    canvas.save(dest/'comparison.png')
    n=report['fragmentation_after'];b=report['fragmentation_before'];calls=semantic['calls'];approved=[v for v in semantic['items'] if v['operation']=='close_small_gaps']
    src_report=json.loads((source/'navigation_update.json').read_text(encoding='utf-8'))
    page=f'''<!doctype html><meta charset="utf-8"><title>结构化栅格候选</title><style>body{{font:18px system-ui;max-width:1800px;margin:28px auto;padding:0 20px;color:#222}}p{{line-height:1.7}}img{{max-width:100%}}td,th{{padding:12px;border-bottom:1px solid #ddd}}</style>
    <h1>3DGS 栅格：深度保护与保守轮廓整理</h1><p>黑色障碍、白色空闲，不用黄色。左图是上轮仅 3DGS 结果；中图是加入实际深度保护的证据图；右图是更规整的候选。右图通过增加局部占用整理轮廓，没有擦掉已有障碍；保留 5cm 地图分辨率、原点与尺寸，局部按 10cm 块合并。</p><img src="comparison.png">
    <table><tr><th>轮廓统计</th><th>中图</th><th>右图</th></tr><tr><td>连通障碍片段</td><td>{b['components']}</td><td>{n['components']}</td></tr><tr><td>不超过4格的小片段</td><td>{b['small_components_le4']}</td><td>{n['small_components_le4']}</td></tr><tr><td>边界栅格边数</td><td>{b['boundary_edges']}</td><td>{n['boundary_edges']}</td></tr></table>
    <p>右图相对原地图清空 {counts['cleared_obstacles']} 格、新增 {counts['added_obstacles']} 格；轮廓整理额外增加 {report['added_by_regularization']} 格占用。这是更保守的障碍外形，不代表物体真实外形变大，也不能把减少的小片段都称为已证实的噪点。</p>
    <h2>DeepSeek 实际做了什么</h2><p>复核 {len(semantic['items'])} 个设备的多视角图像与 A/B/C 栅格候选，{len(approved)} 个对象批准小断口处理，实际新增 {src_report['semantic_additions']} 格。模型没有直接修改坐标，也没有批准删除。主要视觉变化来自独立的 10cm 保守结构化步骤，不能归功于 DeepSeek。<a href="shape_semantics.json">完整模型答复</a></p>
    <h2>检查结果与限制</h2><p>深度保护将中图的清空冲突降到 {src_report['cleared_with_holdout_obstacle']} 格；右图为 {report['cleared_with_holdout_obstacle']} 格。半径20cm参考的自由空间核心，连通分裂 {len(audit['split_free_space_cores'])} 处、占用核心 {audit['lost_core_cells']} 格。此检查不是机器人尺寸校验，保守轮廓仍可能收窄通道。未应用到机器人，未切换默认流程，仍需独立采集与米制尺度验证。</p>
    <p><a href="structured_candidate.zip">下载右图候选</a> · <a href="evidence_map.png">中图细粒度证据图</a> · <a href="navigation_update.json">完整统计</a></p>'''
    (dest/'index.html').write_text(page,encoding='utf-8')
    with zipfile.ZipFile(dest/'structured_candidate.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name in ['map.png','map.yaml','navigation_update.json']:z.write(dest/name,name)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--reference',type=Path,required=True);p.add_argument('--destination',type=Path,required=True);a=p.parse_args();print(json.dumps(build(a.source,a.reference,a.destination)))
