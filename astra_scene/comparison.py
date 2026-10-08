"""Reproducible same-camera comparisons, including unsuccessful proposals."""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from .io import write_json, sha256, output_path


def compare(source, before, after, output, after_label='Candidate'):
    from .postprocess import load, rows_from_data, render_map
    from .quality import structural_metrics
    from vps.refinement.gaussian_renderer import GsplatGaussianRenderer
    import torch, json
    output=output_path(output); output.mkdir(parents=True,exist_ok=False)
    cfg,data,split,source_report=load(source);rows=rows_from_data(data)
    ids=sorted(set(split['validation_views']+split['test_views']))
    if not ids:raise ValueError('comparison_requires_heldout_views')
    scope=json.loads((source/'manual_scope.json').read_text())
    from vps.dynamic.image_mask import content_mask
    capture=json.loads(Path(source_report['capture']).read_text())
    crops={}
    for i in ids:
        h,w=rows[i]['depth'].shape
        with Image.open(capture['frames'][i]['image']) as im:
            valid=content_mask(*im.size,h,w)
        y,x=np.nonzero(valid)
        if not len(x):raise ValueError('comparison_empty_content')
        crops[i]=(int(x.min()),int(y.min()),int(x.max()+1),int(y.max()+1))
    rendered={};metrics=[]
    for name,path in [('before',before),('after',after)]:
        renderer=GsplatGaussianRenderer('cuda'); rendered[name]={}
        for i in ids:
            r=render_map(renderer,path,rows[i]);rendered[name][i]=r['rgb']
            m=structural_metrics(r['rgb'].astype(np.float32)/255,r['depth'],r['alpha'],rows[i],scope,dict(min_pixels=100))
            metrics.append(dict(candidate=name,view=i,split='test' if i in split['test_views'] else 'validation',**m))
        del renderer;torch.cuda.empty_cache()
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18)
    panels=[]
    for i in ids:
        x0,y0,x1,y1=crops[i]
        ims=[rendered['before'][i],rendered['after'][i],np.uint8(np.clip(data['rgb'][i]*255,0,255))]
        w,h=x1-x0,y1-y0
        canvas=Image.new('RGB',(w*3,h+62),'#f0f3f6');draw=ImageDraw.Draw(canvas)
        for j,(im,label) in enumerate(zip(ims,['Before: current candidate',after_label,'Phone observation'])):
            canvas.paste(Image.fromarray(im[y0:y1,x0:x1]),(j*w,62))
            draw.text((j*w+8,8),label,font=font,fill='#182b40')
            draw.text((j*w+8,32),f'View {i} / '+('test' if i in split['test_views'] else 'validation'),font=font,fill='#465a70')
        canvas.save(output/f'view_{i:03d}.png')
        if i in split['test_views']:panels.append(canvas)
    if not panels:panels=[canvas]
    atlas=Image.new('RGB',(max(p.width for p in panels),sum(p.height for p in panels)+50),'white')
    draw=ImageDraw.Draw(atlas);draw.text((10,12),'Same cameras and crops; development regression views, not an independent revisit.',font=font,fill='#182b40')
    y=50
    for panel in panels:atlas.paste(panel,(0,y));y+=panel.height
    atlas.save(output/'comparison.png')
    summary={}
    for name in ['before','after']:
        valid=[m for m in metrics if m['candidate']==name and m['split']=='test' and m['pixels']>=100]
        summary[name]={k:float(np.mean([m[k] for m in valid])) for k in ['target_rgb_l1','missing_surface_fraction','foreground_disagreement_fraction']} if valid else {}
    report=dict(before=str(before),after=str(after),before_sha256=sha256(before),after_sha256=sha256(after),
                after_label=after_label,views=metrics,summary=summary,independent_revisit=False,
                depth_source='estimated',training_history='See candidate provenance; comparison does not train models',deployable=False)
    write_json(output/'report.json',report)
    import html
    (output/'index.html').write_text('<meta charset="utf-8"><title>框架优化回归对照</title><style>body{font:16px system-ui;margin:30px}img{max-width:100%}</style><h1>框架优化回归对照</h1><p>左：原候选；中：'+html.escape(after_label)+'；右：手机实拍。测试视角是已复用的开发回归，不是独立复访。对照图本身不代表通过采用或发布检查。</p><img src="comparison.png"><p><a href="report.json">逐视角与逐区域指标</a></p>',encoding='utf-8')
    return report


if __name__=='__main__':
    import fcntl
    from .io import ROOT
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--before',type=Path,required=True)
    p.add_argument('--after',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--after-label',default='Candidate')
    a=p.parse_args()
    with (ROOT/'runs/gpu.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        compare(a.source,a.before,a.after,a.output,a.after_label)
