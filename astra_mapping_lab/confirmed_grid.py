"""Export confirmed map edits, retaining the prior map wherever evidence is uncertain."""
import argparse,hashlib,json,zipfile
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from navigation_candidate import classes,constrained_result


def confirmed_result(original,candidate,allowed,meta):
    if original.shape!=candidate.shape or original.shape!=allowed.shape:
        raise ValueError('grid shape mismatch')
    _,_,uncertain=classes(candidate,meta)
    proposed=np.where(uncertain,original,candidate).astype(np.uint8)
    result,counts=constrained_result(original,proposed,allowed,meta)
    counts['uncertain_edits_not_applied']=int((allowed&uncertain&(candidate!=original)).sum())
    return result,counts


def build(source,dest):
    import yaml
    source=Path(source);dest=Path(dest);dest.mkdir(parents=True,exist_ok=False)
    meta=yaml.safe_load((source/'map.yaml').read_text(encoding='utf-8'))
    original=np.array(Image.open(source/'original_map.png').convert('L'))
    candidate=np.array(Image.open(source/'map.png').convert('L'))
    allowed=np.array(Image.open(source/'allowed_scope.png'))>0
    result,counts=confirmed_result(original,candidate,allowed,meta)
    report=dict(**counts,policy='apply confirmed edits; uncertain candidate cells retain original; no smoothing',
        published=False,new_geometric_reconstruction=False,source=str(source.resolve()),
        source_candidate_sha256=hashlib.sha256((source/'map.png').read_bytes()).hexdigest(),
        navigation_deployment_accepted=False)
    Image.fromarray(result).save(dest/'map.png');Image.fromarray(original).save(dest/'original_map.png')
    meta['image']='map.png';(dest/'map.yaml').write_text(yaml.safe_dump(meta,sort_keys=False),encoding='utf-8')
    report['map_sha256']=hashlib.sha256((dest/'map.png').read_bytes()).hexdigest()
    (dest/'navigation_update.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    y,x=np.where(allowed);box=(max(0,int(x.min())-8),max(0,int(y.min())-8),min(result.shape[1],int(x.max())+9),min(result.shape[0],int(y.max())+9)) if len(x) else (0,0,result.shape[1],result.shape[0])
    h=round((box[3]-box[1])*680/(box[2]-box[0]));canvas=Image.new('RGB',(1440,h+90),'white');d=ImageDraw.Draw(canvas)
    fontpath=next((p for p in [Path('C:/Windows/Fonts/arial.ttf'),Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')] if p.exists()),None)
    f=ImageFont.truetype(str(fontpath),26) if fontpath else ImageFont.load_default()
    for j,(a,label) in enumerate([(original,'Original map'),(result,'Updated: confirmed edits only')]):
        d.text((20+j*720,15),label,font=f,fill='black');canvas.paste(Image.fromarray(a).convert('RGB').crop(box).resize((680,h),Image.Resampling.NEAREST),(20+j*720,65))
    canvas.save(dest/'comparison.png')
    (dest/'index.html').write_text(f'''<!doctype html><meta charset="utf-8"><title>黑白栅格更新结果</title>
    <style>body{{font:18px system-ui;max-width:1450px;margin:30px auto;padding:0 20px}}p{{line-height:1.7}}img{{width:100%}}</style>
    <h1>黑白栅格：只应用有证据的删除和更新</h1>
    <p>左：原栅格；右：更新结果。黑色是障碍，白色是空闲。不显示黄色，也不把待确认区域写成新的未知格。</p>
    <img src="comparison.png"><p>确认清除 {counts['cleared_obstacles']} 格，新增障碍 {counts['added_obstacles']} 格。
    {counts['uncertain_edits_not_applied']} 格证据不足，保持原地图状态。本次没有新增删除证据，也没有用平滑操作抹掉细小障碍。</p>
    <p>这是已有 V3 几何结果的确定性更新导出，不是新 RGB-D 子图重建的效果。未改机器人地图或默认服务。</p>
    <p><a href="navigation_candidate.zip">下载地图与配置</a></p>''',encoding='utf-8')
    with zipfile.ZipFile(dest/'navigation_candidate.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name in ['map.png','map.yaml','navigation_update.json']:z.write(dest/name,name)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--destination',type=Path,required=True);a=p.parse_args();print(json.dumps(build(a.source,a.destination)))
