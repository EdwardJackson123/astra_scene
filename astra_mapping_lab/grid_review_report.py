"""Explain candidate grid decisions and export observation targets, never nav goals."""
import argparse,collections,hashlib,html,json,zipfile
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from navigation_candidate import classes

REASONS={
    1:('未进入可移动对象复核范围',(125,135,150)),
    2:('当前深度仍支持障碍',(70,125,80)),
    3:('旧表面仍有多视角支持',(140,85,160)),
    4:('缺少历史表面样本',(160,160,160)),
    5:('旧表面穿透证据不足',(215,105,65)),
    6:('已变化但仍有未解释表面：未知',(240,175,40)),
    7:('其他待确认：未知',(210,195,100)),
    8:('当前规则确认清空',(35,155,245)),
    9:('新增障碍',(225,55,55)),
}


def decision_reasons(original,result,allowed,meta,e):
    old,_,_=classes(original,meta);occupied,free,unknown=classes(result,meta)
    labels=np.zeros(original.shape,np.uint8);remaining=old&occupied&allowed
    # Positive current evidence takes priority over a semantic explanation.
    for code,condition in [(2,e['current_obstacle']),(1,~e['semantic']),(4,e['total']<2),
                           (3,e['stable']>0),(5,np.ones(original.shape,bool))]:
        choose=remaining&condition;labels[choose]=code;remaining&=~choose
    labels[old&unknown&allowed]=7
    labels[old&unknown&allowed&e['retired']&(e['cleared']<e['total'])]=6
    labels[old&free&allowed]=8;labels[~old&occupied&allowed]=9
    return labels


def observation_targets(labels,e,allowed,meta):
    from scipy.ndimage import label
    # Unknown or insufficient geometry only; never ask to erase stable buildings.
    pending=np.isin(labels,[4,5,6,7])&allowed
    groups,n=label(pending,np.ones((3,3),int));targets=[]
    h=labels.shape[0];yaw=float(meta['origin'][2]);c,s=np.cos(yaw),np.sin(yaw)
    for k in range(1,n+1):
        y,x=np.where(groups==k)
        if len(x)<3:continue
        xy=np.array([x.mean()+.5,h-y.mean()-.5])*meta['resolution']
        xy=xy@np.array([[c,s],[-s,c]])+np.asarray(meta['origin'][:2])
        unresolved=int(np.isin(labels[y,x],[6,7]).sum())
        targets.append(dict(cells=len(x),unknown_cells=unresolved,
            bounds_pixels=[int(x.min()),int(y.min()),int(x.max()+1),int(y.max()+1)],
            centroid_pixels=[float(x.mean()),float(y.mean())],target_map_xy=xy.tolist(),
            reasons=dict(collections.Counter(str(int(v)) for v in labels[y,x])),
            priority_score=len(x)+unresolved,
            instruction='补充能看到旧障碍残余表面与地面的侧向视角；目标中心可能位于障碍内，不能作为导航落脚点'))
    targets.sort(key=lambda t:-t['priority_score'])
    return dict(targets=[dict(id=f'coverage_{i+1:02d}',**t) for i,t in enumerate(targets)],
                executable_navigation_goals=False,metric_coordinates_verified=False,
                rule='8-connected unresolved cells, >=3 cells; sorted by area plus unknown count; no held-out evidence used')


def build(dest,reference):
    import yaml
    read=lambda p:json.loads(p.read_text(encoding='utf-8'))
    report=read(dest/'navigation_update.json');before=read(reference/'navigation_update.json')
    meta=yaml.safe_load((dest/'map.yaml').read_text(encoding='utf-8'))
    original=np.array(Image.open(dest/'original_map.png').convert('L'));result=np.array(Image.open(dest/'map.png').convert('L'))
    allowed=np.array(Image.open(dest/'allowed_scope.png'))>0;e=np.load(dest/'cell_evidence.npz')
    old,_,_=classes(original,meta);occ,free,unknown=classes(result,meta);cleared=old&free
    outside_ok=bool(np.array_equal(original[~allowed],result[~allowed]))
    conflicts=int((cleared&e['held_obstacle']).sum())
    audit=dict(outside_scope_unchanged=outside_ok,cleared_cells=int(cleared.sum()),
        cleared_with_holdout_obstacle=conflicts,
        minimum_ground_views=int(e['ground_views'][cleared].min()) if cleared.any() else None,
        unexplained_old_surface_in_free_cells=int((cleared&(e['cleared']<e['total'])).sum()),
        current_obstacle_in_free_cells=int((cleared&e['current_obstacle']).sum()),
        map_sha256=hashlib.sha256((dest/'map.png').read_bytes()).hexdigest(),
        passed=outside_ok and conflicts==0 and not bool((cleared&((e['cleared']<e['total'])|e['current_obstacle']|(e['ground_views']<3))).any()),
        navigation_accepted=False)
    assert audit['cleared_cells']==report['cleared_obstacles'],'report/map count mismatch'
    labels=decision_reasons(original,result,allowed,meta,e)
    counts={str(k):dict(reason=name,cells=int((labels==k).sum())) for k,(name,_) in REASONS.items()}
    targets=observation_targets(labels,e,allowed,meta)
    for name,data in [('verification.json',audit),('decision_reasons.json',counts),('observation_targets.json',targets)]:
        (dest/name).write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    Image.fromarray(labels).save(dest/'decision_labels.png')
    colored=np.stack([original]*3,axis=-1)
    for k,(_,color) in REASONS.items():colored[labels==k]=color
    Image.fromarray(colored).save(dest/'decision_reasons.png')
    delta=np.stack([original]*3,axis=-1);delta[~old&occ]=[225,55,55];delta[cleared]=[35,155,245];delta[old&unknown]=[240,175,40]
    Image.fromarray(delta).save(dest/'changes.png')
    y,x=np.where(allowed)
    box=(max(0,int(x.min())-8),max(0,int(y.min())-8),min(result.shape[1],int(x.max())+9),min(result.shape[0],int(y.max())+9)) if len(x) else (0,0,result.shape[1],result.shape[0])
    fonts=[Path('C:/Windows/Fonts/arial.ttf'),Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')]
    font=next((ImageFont.truetype(str(p),21) for p in fonts if p.exists()),ImageFont.load_default())
    height=round((box[3]-box[1])*570/(box[2]-box[0]));canvas=Image.new('RGB',(1800,height+70),'white');draw=ImageDraw.Draw(canvas)
    for j,(im,title) in enumerate([(Image.open(reference/'map.png'),'Previous candidate'),(Image.fromarray(result),'New: unknown is NOT free'),(Image.fromarray(delta),'Blue: free | Amber: unknown')]):
        draw.text((j*600+8,10),title,font=font,fill='black');canvas.paste(im.convert('RGB').crop(box).resize((570,height),Image.Resampling.NEAREST),(j*600+8,55))
    canvas.save(dest/'grid_comparison.png')
    reasons=Image.fromarray(colored).crop(box).resize((1140,2*height),Image.Resampling.NEAREST)
    reasons.save(dest/'residual_zoom.png')
    capture=Image.fromarray(np.stack([result]*3,axis=-1)).crop(box).resize((1140,2*height),Image.Resampling.NEAREST);d=ImageDraw.Draw(capture)
    sx=1140/(box[2]-box[0]);sy=2*height/(box[3]-box[1])
    for t in targets['targets'][:10]:
        x0,y0,x1,y1=t['bounds_pixels'];rect=[(x0-box[0])*sx,(y0-box[1])*sy,(x1-box[0])*sx,(y1-box[1])*sy]
        d.rectangle(rect,outline=(235,0,155),width=3)
        pos=(max(0,rect[0]),max(0,rect[1]-27));label=t['id'].split('_')[-1]
        d.rectangle([pos[0],pos[1],pos[0]+32,pos[1]+25],fill='white');d.text(pos,label,font=font,fill=(145,0,105))
    capture.save(dest/'observation_targets.png')
    rows=''.join(f'<tr><td>{name}</td><td>{before[key]}</td><td>{report[key]}</td></tr>' for key,name in [('added_obstacles','新增占用'),('cleared_obstacles','旧占用转空闲'),('occupied_to_unknown','旧占用转未知')])
    legend=''.join(f'<li><span style="background:rgb{color};display:inline-block;width:16px;height:16px"></span> {name}：{counts[str(k)]["cells"]} 格</li>' for k,(name,color) in REASONS.items())
    sem=read(dest/'semantic_review.json');states=dict(collections.Counter(i['old_state'] for i in sem['items']))
    cache_hits=sum(bool(c.get('cache_hit')) for c in sem['calls']);calls=len(sem['calls'])
    history=dest/'experiment_comparison.json'
    trial_note=html.escape(read(history)['note']) if history.exists() else '旧表面大部分消失只表示变化，不足以直接判定可通行；因此本版将未解释残余表面保留为未知。'
    page=f'''<!doctype html><meta charset="utf-8"><title>栅格更新与残留诊断</title>
    <style>body{{font:17px system-ui;max-width:1600px;margin:30px auto;padding:0 20px;color:#203040}}img{{max-width:100%}}td,th{{padding:10px 25px;border-bottom:1px solid #ddd}}p{{line-height:1.7}}</style>
    <h1>栅格更新：分开判断旧障碍变化与可通行性</h1>
    <p>已有数据回放。未修改机器人服务或地图；未切换默认自动流程。未知格不能当作已验证的可通行格。</p>
    <table><tr><th>变化（格）</th><th>上一版候选</th><th>本版</th></tr>{rows}</table>
    <img src="grid_comparison.png">
    <p>本版使用 {len(report.get('geometric_evidence_views',[]))} 个独立训练位置复核几何，语义/历史表面采样使用 {len(report.get('surface_source_views',[]))} 个位置。85% 旧表面清除仅提名退役；所有已观测旧表面都得到清除证据、至少三次地面观测且无当前障碍，才标空闲。尚有未解释表面的格子转为未知。正障碍证据不作形态学删除。</p>
    <p>{trial_note}</p>
    <h2>每个残留格为什么还在</h2><img src="residual_zoom.png"><ul>{legend}</ul>
    <h2>针对性补采区域</h2><p>共 {len(targets['targets'])} 个连续证据不足区域，下图显示优先级最高的 10 个。它们是观察目标，不是机器人可执行导航目标；仍需从已验证可行走位置选取视角，不能朝框中心直接行驶。</p><img src="observation_targets.png">
    <h2>DeepSeek 与验证</h2><p>DeepSeek 输出 {html.escape(str(states))}，{calls} 批中 {cache_hits} 批命中缓存；模型仅辅助可移动对象范围，不决定空闲。留出障碍冲突 {conflicts} 格，审计通过：{audit['passed']}。范围外保持不变：{outside_ok}。</p>
    <p>本轮仍是同一采集数据的开发评估，留出结果已用于分析规则问题，不是独立验收。已有米制尺度冲突尚未解决；所有已观测旧表面清除，也不等于未观测体积无障碍。未通过实机导航和尺寸间隙验证。</p>
    <p><a href="navigation_candidate.zip">候选地图及证据包</a> · <a href="decision_reasons.json">残留原因统计</a> · <a href="observation_targets.json">补采区域数据</a> · <a href="verification.json">核验结果</a></p>'''
    (dest/'index.html').write_text(page,encoding='utf-8')
    with zipfile.ZipFile(dest/'navigation_candidate.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name in ['map.png','map.yaml','navigation_update.json','semantic_review.json','verification.json','decision_reasons.json','observation_targets.json']:z.write(dest/name,name)
    return audit


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--destination',type=Path,required=True);p.add_argument('--reference',type=Path,required=True);a=p.parse_args();print(json.dumps(build(a.destination,a.reference)))
