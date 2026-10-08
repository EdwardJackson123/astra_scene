"""Full, fixed-view comparison for the bounded Boxer fusion experiment."""
import argparse,json,html,shutil
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont


def build(out, previous):
    dest=out/'artifacts';review=dest/'review'
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',20)
    fusion=json.loads((dest/'fusion.json').read_text())
    geometry=json.loads((dest/'structural_review.json').read_text())
    ownership=json.loads((dest/'boxer_ownership.json').read_text())
    frames=sorted(p.name.removesuffix('_gt.png') for p in review.glob('*_gt.png'))
    for frame in frames:
        shutil.copyfile(previous/'artifacts/review'/f'{frame}_candidate.png',review/f'{frame}_last_method.png')
    def collage(keys, columns, filename, size=(640,480)):
        w,h=size;canvas=Image.new('RGB',(w*len(columns),(h+40)*len(keys)),(241,245,248));draw=ImageDraw.Draw(canvas)
        for y,key in enumerate(keys):
            for x,(suffix,label) in enumerate(columns):
                draw.text((x*w+8,y*(h+40)+8),key+' | '+label,fill=(20,40,60),font=font)
                canvas.paste(Image.open(review/f'{key}_{suffix}.png').resize((w,h)),(x*w,y*(h+40)+40))
        canvas.save(dest/filename,quality=94)
    columns=[('last_method','Previous method'),('candidate','Boxer + assembly completion'),('patch','Masked Spirula alone'),('gt','Current observation')]
    collage(frames[1::2][::3],columns,'boxer_final_comparison.jpg')
    collage(frames[1::2][::3],[columns[0],columns[1],columns[3]],'robot_height_comparison.jpg')
    pages=[]
    for j in range(0,len(frames),3):
        name=f'boxer_views_{j//3+1}.jpg';collage(frames[j:j+3],columns,name);pages.append(name)
    labels={'old':'原地图','last_method':'上一版过度筛选候选','previous':'Boxer 首版','candidate':'Boxer 连通补全版'}
    h='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Boxer 双向区域融合 · 多视角对照</title><style>body{font:16px/1.75 system-ui;max-width:1550px;padding:24px;margin:auto;color:#193246;background:#eef3f7}section{background:white;padding:24px;margin:22px 0;border-radius:14px}img{width:100%}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ccd7df;padding:10px;text-align:left}a{color:#1267b3}.warn{border-left:5px solid #d48b16}summary{cursor:pointer}</style><h1>Boxer 双向检测＋设备连通补全＋Context 融合</h1>'''
    h+='<section class="warn"><b>离线实验候选，未替换原地图，未通过结构验收。</b><p>复用同一组人物遮罩 Spirula SH0 和同一配准；没有重新训练 3DGS。栅格框仅限定权限。机器人导航代码、服务和正式定位图库均未修改。</p><p>32 个训练视角做旧图/当前画面双向 Boxer 检测 → DeepSeek 多视角区域复核 → SAM 设备实例分割 → 有界三维连通补全 → 分别计算旧实例删除与新实例插入 → 同样 320 步 Context RGB-D 优化。所有推理失败或证据不完整项均记录，没有强改 AI 意见为通过。</p><p>旧实例推断退役与实测腾空分开记录：对已经被新结构遮挡的旧背面，部分删除来自实例级推断，不能据此认定可通行。导航清空仍独立检查深度证据。本候选不能当作正式导航地图。</p></section>'
    h+=f'<section><h2>本版实际改动</h2><p>原图 {fusion["base_points"]:,} 个高斯；删除 {fusion["removed_old_points"]:,} 个，新增 {fusion["inserted_points"]:,} 个。{fusion["frozen_old_points"]:,} 个旧高斯逐字节不变；候选共 {fusion["candidate_points"]:,} 个。</p><p><a href="scene.ply">下载本版完整 SH0 候选 PLY</a> · <a href="fusion.json">增删审计</a> · <a href="boxer_ownership.json">逐实例归属/推断记录</a> · <a href="structural_review.json">机器人采集视角几何指标</a></p><p>显示外观改善不等于几何已正确：机器人高度下的侧视残影、细杆、窗边、接缝和缺面需同时检查；全局尺度校正约 0.908，尚无独立米制真值验证。</p></section>'
    h+='<section><h2>相同 6 张测试图、相同本版评估区域</h2><table><tr><th>版本</th><th>区域 RGB L1 ↓</th><th>缺面比例 ↓</th><th>前景深度冲突 ↓</th></tr>'
    for key,label in labels.items():
        if key not in geometry['summary']:continue
        m=geometry['summary'][key];h+=f'<tr><td>{label}</td><td>{m["target_rgb_l1"]:.4f}</td><td>{m["missing_surface_fraction"]:.1%}</td><td>{m["foreground_disagreement_fraction"]:.1%}</td></tr>'
    h+='</table><p>均来自同次采集留出图，已用于开发诊断，不是独立回访验证。人物区域不计入静态指标。结构验收状态：'+('通过本轮阈值，仍需现场验证' if geometry['passed'] else '未通过')+'。</p></section>'
    h+='<section><h2>固定测试视角对照</h2><p>从左到右：上一版 → 本次 Boxer → 单独 Spirula 重建 → 当前实拍。第三列帮助区分“原重建就缺失”与“融合时丢失”；它不是完整地图。</p><a href="boxer_final_comparison.jpg"><img src="boxer_final_comparison.jpg"></a></section>'
    h+='<section><h2>机器人实际高度：全部 12 个留出视角</h2><p>全部采用本次 D455F 实际采集机位，包含 6 张验证图与 6 张测试图。相对本地图推定地面，相机高度约 1.02–1.08 m（尺度尚未独立验证）。只验收机器人正常可达机位；高处俯视不作为目标。点击图片查看原尺寸。</p>'
    for file in pages:h+=f'<a href="{file}"><img loading="lazy" src="{file}"></a>'
    h+='</section><section><h2>逐区域复核</h2>'
    proposals=json.loads((out/'analysis/evidence.json').read_text())['proposals'];ai={a['id']:a for a in json.loads((out/'analysis/ai.json').read_text())['items']}
    for p in proposals:
        a=ai[p['id']];h+=f'<details><summary>{html.escape(p["id"])} · {html.escape(p["label"])} · {html.escape(a["decision"])}</summary><p>{html.escape(a["reason"])}</p>'
        for im in p['images']:h+=f'<img loading="lazy" src="evidence/{im["file"]}">'
        h+='</details>'
    h+='</section></html>';(dest/'index.html').write_text(h,encoding='utf-8')
    print('BOXER_REPORT',dest/'index.html',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--previous',type=Path,required=True);a=p.parse_args();build(a.output,a.previous)
