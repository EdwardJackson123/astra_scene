"""Per-job report without hard-coded reference experiments or camera heights."""
import argparse,html,json,shutil
from pathlib import Path

def build(out):
    from PIL import Image,ImageDraw,ImageFont
    dest=out/'artifacts';review=dest/'review'
    structural=json.loads((dest/'structural_review.json').read_text())
    fusion=json.loads((dest/'fusion.json').read_text())
    frames=sorted(p.name.removesuffix('_gt.png') for p in review.glob('*_gt.png'))
    if not frames:raise ValueError('报告缺少留出视角')
    comparison=json.loads((out/'comparison.json').read_text()) if (out/'comparison.json').exists() else {}
    previous=comparison.get('previous_review')
    if previous:
        for frame in frames:shutil.copyfile(Path(previous)/(frame+'_candidate.png'),review/(frame+'_previous.png'))
    columns=[('old','Original map'),('patch','Standalone Spirula'),('candidate','Protected device + floor update'),('gt','Current observation')]
    if previous:columns=[('old','Original map'),('previous','Previous automatic update'),('candidate','Current automatic update'),('gt','Current observation')]
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',21)
    pages=[]
    for j in range(0,len(frames),3):
        keys=frames[j:j+3];canvas=Image.new('RGB',(2560,520*len(keys)),(242,246,248));draw=ImageDraw.Draw(canvas)
        for y,key in enumerate(keys):
            for x,(label,title) in enumerate(columns):
                draw.text((640*x+8,520*y+8),key+' | '+title,font=font,fill=(25,40,55))
                canvas.paste(Image.open(review/f'{key}_{label}.png'),(640*x,520*y+40))
        name=f'robot_views_{j//3+1}.jpg';canvas.save(dest/name,quality=94);pages.append(name)
    h='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>自动地图更新候选</title><style>body{font:17px/1.7 system-ui;max-width:1500px;margin:auto;padding:24px}img{width:100%}td,th{padding:10px;border:1px solid #ccc}table{border-collapse:collapse}</style><h1>设备与局部地面更新候选</h1>'
    h+='<p>自动流程：当前任务对齐 → RGB-D 变化证据 → 几何分流 → 必要的 DeepSeek 判断 → 分割与连通补全 → 冲突检查 → 建筑保护融合 → 留出视角检查。ROI 仅限定授权范围。天花板、窗户、墙和室外背景受保护；允许设备区域内的局部地面更新。分类和配准仍可能有误。</p>'
    h+='<p><strong>自动处理结果；正式原地图、导航栅格和定位图库未修改。</strong>几何检查：'+('通过离线阈值，仍需独立验证。' if structural['passed'] else '未通过，本次结果未应用。需要改善覆盖或配准后重试。')+'</p>'
    nav=dest/'navigation_candidate/navigation_update.json'
    if nav.exists():
        n=json.loads(nav.read_text())
        repair=n.get('hole_repair',{})
        h+='<p>实际栅格后端：'+html.escape(n.get('grid_backend','legacy_rgbd'))+'；导航验收：'+html.escape(n.get('navigation_assessment',{}).get('state','NOT_VERIFIED'))+'。</p>'
        h+=f'<p>栅格后处理：有观测证据的封闭孔洞修补 {repair.get("added_cells",0)} 格；参考连通性检查 {"通过" if repair.get("topology_accepted") else "未通过或未执行"}。未做全图膨胀。<a href="navigation_candidate/hole_repair.json">补洞审计</a></p>'
        h+=f'<h2>导航栅格候选（未发布）</h2><p>新增障碍 {n["added_obstacles"]} 格；确认清除 {n["cleared_obstacles"]} 格；占用转未知 {n["occupied_to_unknown"]} 格。未知不等于可通行，尺度和通行高度待验证。</p><p><a href="navigation_candidate/index.html">查看栅格更新对比</a> · <a href="navigation_candidate/navigation_candidate.zip">下载地图与配置</a></p><img src="navigation_candidate/grid_comparison.png">'
    h+=f'<p>移除旧高斯 {fusion["removed_old_points"]:,}；加入新高斯 {fusion["inserted_points"]:,}。保留旧高斯未改变；设备主体未进行混合外观优化。新地面交界处可做局部颜色衔接，几何与不透明度不变。</p>'
    h+='<p><a href="scene.ply">下载融合候选 PLY</a> · <a href="fusion.json">增删审计</a> · <a href="takeover.json">区域与保护规则</a> · <a href="structural_review.json">几何检查</a></p><table><tr><th>版本</th><th>区域 RGB L1 ↓</th><th>缺失表面 ↓</th><th>前景深度不一致 ↓</th></tr>'
    def value(v,percent=False):return '无有效数据' if v is None else (f'{v:.1%}' if percent else f'{v:.4f}')
    for name,label in ([('previous','优化前融合')] if previous else [('old','原地图')])+[('single_spirula','独立 Spirula'),('candidate','本次融合候选')]:
        m=structural['summary'][name];h+=f'<tr><td>{label}</td><td>{value(m["target_rgb_l1"])}</td><td>{value(m["missing_surface_fraction"],True)}</td><td>{value(m["foreground_disagreement_fraction"],True)}</td></tr>'
    labels='原地图、优化前融合、优化后融合、实拍' if previous else '原地图、独立重建、融合候选、实拍'
    h+=f'</table><h2>全部 {len(frames)} 个实际采集留出机位</h2><p>每行从左到右：{labels}。未添加高处俯视机位；同次采集不等于独立回访验证。</p>'
    for name in pages:h+=f'<a href="{name}"><img loading="lazy" src="{name}"></a>'
    (dest/'index.html').write_text(h+'</html>',encoding='utf-8')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);build(p.parse_args().output)
