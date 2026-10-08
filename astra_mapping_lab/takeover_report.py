"""Robot-height report for regional replacement, with all fixed held-out views."""
import argparse,json,shutil
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont


def build(out, prior):
    dest=out/'artifacts';review=dest/'review'
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',21)
    read=lambda file:json.loads((dest/file).read_text())
    fusion=read('fusion.json');takeover=read('takeover.json');structure=read('structural_review.json');summary=read('summary.json')
    frames=sorted(p.name.removesuffix('_gt.png') for p in review.glob('*_gt.png'))
    for frame in frames:shutil.copyfile(prior/'artifacts/review'/f'{frame}_candidate.png',review/f'{frame}_previous.png')
    def collage(keys,cols,file):
        canvas=Image.new('RGB',(640*len(cols),520*len(keys)),(242,246,248));draw=ImageDraw.Draw(canvas)
        for y,key in enumerate(keys):
            for x,(label,title) in enumerate(cols):
                draw.text((640*x+10,520*y+8),key+' | '+title,font=font,fill=(25,40,55))
                canvas.paste(Image.open(review/f'{key}_{label}.png'),(640*x,520*y+40))
        canvas.save(dest/file,quality=94)
    protected=takeover.get('architecture_protection',False)
    columns=[('old','Original map'),('previous','Before: mixed fusion'),('candidate','Device-only takeover'),('gt','Current observation')] if protected else [('previous','Before: mixed fusion'),('candidate','After: single region owner'),('patch','Spirula reconstruction'),('gt','Current observation')]
    collage(frames[1::2][::3],[('previous','Before: mixed fusion'),('candidate','After: device takeover'),('gt','Current observation')],'robot_height_comparison.jpg')
    pages=[]
    for j in range(0,len(frames),3):
        file=f'robot_views_{j//3+1}.jpg';collage(frames[j:j+3],columns,file);pages.append(file)
    if not protected:
        field=np.load(dest/'takeover_field.npz');f=field['field'];proposed=field['proposed']>0
        im=np.full((*f.shape,3),245,np.uint8);im[proposed]=(245,178,89);im[f]=(42,165,104)
    else:
        old=np.load(dest/'removed_old_centers.npy')[:,:2];fresh=np.load(dest/'replacement_support.npy')[:,:2]
        xy=np.concatenate([old,fresh]);origin=xy.min(0)-.2;shape=np.ceil((xy.max(0)-origin+.2)/.05).astype(int)[::-1]
        im=np.full((*shape,3),245,np.uint8)
        for pts,color in [(old,(230,110,85)),(fresh,(42,165,104))]:
            ij=np.floor((pts-origin)/.05).astype(int);im[ij[:,1],ij[:,0]]=color
    Image.fromarray(np.flipud(im)).resize((800,max(1,int(800*im.shape[0]/im.shape[1]))),Image.Resampling.NEAREST).save(dest/'ownership_scope.png')
    h='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>区域接管 · 机器人机位对比</title><style>body{font:17px/1.75 system-ui;max-width:1550px;margin:auto;padding:24px;background:#eff4f7;color:#193246}section{background:white;border-radius:14px;padding:24px;margin:22px 0}img{width:100%}a{color:#136ab3}td,th{border:1px solid #d1dce4;padding:10px}table{border-collapse:collapse;width:100%}summary{cursor:pointer}.scope{max-width:650px}</style><h1>区域内新重建接管，交界处单独衔接</h1>'''
    if protected:
        h=h.replace('区域内新重建接管，交界处单独衔接','设备区域接管，建筑背景受保护')
        h+='<section><b>独立候选；机器人服务、导航栅格、正式定位图库未修改。</b><p>本版取消二维足迹向天花板贯穿，也取消背景扩张和建筑颜色调整。只在 Boxer＋DeepSeek 提议并经连通补全的三维设备范围内进行新旧替换；天花板、窗、墙和窗外背景由旧/新视图分别做建筑分割保护。设备底座和旧位置范围内的局部地面允许接管；旧设备上部范围已补全，不再只清退约 1.49 m 以下部分。所有保留的旧高斯逐字节不变，插入的新高斯也原样复制。</p><p>保护来自多视角分类和几何约束，不代表分割绝对正确；请结合下面原地图与机器人机位逐张检查。设备区域退役仍不等于可通行证明。</p></section>'
    else:
        h+='<section><b>已弃用：本版更新范围过大，不作为交付。</b><p>二维足迹贯穿和背景扩张会影响窗、天花板和窗外；请使用带建筑保护的三维设备区域版本。</p></section>'
    h+=f'<section><h2>本轮结果</h2><p>删除旧高斯 {fusion["removed_old_points"]:,}，插入原样新高斯 {fusion["inserted_points"]:,}；保留 {fusion["frozen_old_points"]:,} 个逐字节不变的旧高斯。本版保留旧高斯颜色改动 {fusion["old_color_seam_points"]:,} 个。</p><p><a href="scene.ply">下载完整 SH0 设备更新候选</a> · <a href="fusion.json">原样复制及增删审计</a> · <a href="takeover.json">建筑保护与区域规则</a> · <a href="structural_review.json">机器人视角几何检查</a></p></section>'
    h+='<section><h2>同 6 个机器人测试机位、同一评估区域</h2><table><tr><th>版本</th><th>区域 RGB L1 ↓</th><th>缺面 ↓</th><th>前景深度冲突 ↓</th></tr>'
    for name,label in [('previous','之前 Boxer 混合融合'),('candidate','本次区域接管'),('single_spirula','单独 Spirula 参考')]:
        m=structure['summary'][name];h+=f'<tr><td>{label}</td><td>{m["target_rgb_l1"]:.4f}</td><td>{m["missing_surface_fraction"]:.1%}</td><td>{m["foreground_disagreement_fraction"]:.1%}</td></tr>'
    h+='</table><p>结构检查：'+('通过当前离线阈值，尚未实机验证。' if structure['passed'] else '未通过当前全部阈值，仍为待复核候选。')+'同次采集留出图已用于开发诊断，不是独立回访。全局尺度和位姿仍需独立验证；高处俯视不作为验收目标。</p></section>'
    h+='<section><h2>固定机器人测试视角</h2><p>左：调整前；中：区域接管；右：当前实拍。点击看原尺寸。</p><a href="robot_height_comparison.jpg"><img src="robot_height_comparison.jpg"></a></section>'
    h+='<section><h2>全部 12 个机器人留出机位</h2><p>各行依次为：原地图、调整前、建筑保护后的设备接管、实拍。包含正面、转向和设备侧面；不按好看程度筛图。相机相对当前地图推定地面约 1.02–1.08 m。</p>'
    for file in pages:h+=f'<a href="{file}"><img loading="lazy" src="{file}"></a>'
    h+='</section><section><h2>实际接管足迹</h2><p>绿色为新增设备高斯的俯视投影，红色为删除旧高斯的投影。它只是实际三维修改的投影，不表示整列替换；建筑背景另受保护。</p><img class="scope" src="ownership_scope.png"><p>按区域接管的是视图地图，不是逐体素可通行证明。旧地图仍独立保留，可回滚。</p></section></html>'
    (dest/'index.html').write_text(h,encoding='utf-8')
    print('TAKEOVER_REPORT',dest/'index.html',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--prior',type=Path,required=True);a=p.parse_args();build(a.output,a.prior)
