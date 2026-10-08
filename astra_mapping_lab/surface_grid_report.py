"""Audit and present an existing surface-grid candidate without changing its map."""
import argparse, collections, hashlib, html, json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def build(dest, previous):
    read=lambda p: json.loads(p.read_text(encoding='utf-8'))
    report=read(dest/'navigation_update.json'); old_report=read(previous/'navigation_update.json')
    original=np.array(Image.open(dest/'original_map.png')); current=np.array(Image.open(dest/'map.png'))
    earlier=np.array(Image.open(previous/'map.png')); allowed=np.array(Image.open(dest/'allowed_scope.png'))>0
    e=np.load(dest/'cell_evidence.npz')
    assert np.array_equal(original[~allowed],current[~allowed])
    # This audit fixture uses the captured map's negate=0 trinary convention.
    assert np.all(current[earlier==0]==0) or np.all(e['free'][(earlier==0)&(current!=0)])
    assert not np.any(e['free']&e['current_obstacle'])
    assert not np.any(e['free']&e['held_obstacle'])
    assert np.all(e['ground_views'][e['free']]>=3)
    assert np.count_nonzero((original==0)&(current>=250))==report['cleared_obstacles']
    previous_add=(original!=0)&(earlier==0)
    assert np.all(current[previous_add]==0)
    audit=dict(outside_scope_unchanged=True,previous_additions_preserved=True,
               cleared_cells=int(e['free'].sum()),minimum_ground_views=int(e['ground_views'][e['free']].min()),
               map_sha256=hashlib.sha256((dest/'map.png').read_bytes()).hexdigest())
    (dest/'verification.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    yy,xx=np.where(e['free']); box=(max(0,int(xx.min())-15),max(0,int(yy.min())-15),min(current.shape[1],int(xx.max())+16),min(current.shape[0],int(yy.max())+16))
    font_path=next(p for p in [Path('C:/Windows/Fonts/arial.ttf'),Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')] if p.exists())
    font=ImageFont.truetype(str(font_path),22)
    h=round((box[3]-box[1])*560/(box[2]-box[0])); canvas=Image.new('RGB',(1740,h+70),'white'); draw=ImageDraw.Draw(canvas)
    delta=np.stack([current]*3,axis=-1);delta[e['free']]=[30,155,245]
    for j,(arr,label) in enumerate([(earlier,'Previous candidate'),(current,'New candidate'),(delta,'Blue: 134 cleared old cells')]):
        draw.text((j*580+8,10),label,font=font,fill='black')
        canvas.paste(Image.fromarray(arr).convert('RGB').crop(box).resize((560,h),Image.Resampling.NEAREST),(j*580+8,55))
    canvas.save(dest/'deletion_zoom.png')
    states=collections.Counter(x['old_state'] for x in read(dest/'semantic_review.json')['items'])
    rows=''.join(f'<tr><td>{label}</td><td>{old_report[k]}</td><td>{report[k]}</td></tr>' for k,label in [('added_obstacles','新增占用'),('cleared_obstacles','旧占用→空闲'),('occupied_to_unknown','旧占用→未知')])
    page=f'''<!doctype html><meta charset="utf-8"><title>栅格旧障碍更新对比</title>
    <style>body{{font:17px system-ui;max-width:1500px;margin:30px auto;padding:0 20px;color:#203040}}img{{max-width:100%}}td,th{{padding:10px 25px;border-bottom:1px solid #ddd}}p{{line-height:1.7}}</style>
    <h1>栅格旧障碍更新：DeepSeek 辅助分类 + 多视角旧表面验证</h1>
    <p>原地图 xichuang0915，使用已有 D455F 数据及冻结的重建结果。候选实验未替换机器人地图，尚未接入默认自动构建。</p>
    <table><tr><th>变化（格）</th><th>上一版</th><th>本版</th></tr>{rows}</table>
    <p>删除放大图：蓝色仅表示本版清除的 134 个旧占用栅格。其余黑色是保留或新增的占用。</p><img src="deletion_zoom.png">
    <h2>如何决定删除</h2><p>框选区域与连通设备范围限定允许修改的位置。DeepSeek 辅助区分可移动设备和固定结构；模型判定仍在、固定结构或调用失败时不进入本轮删除。可移动但状态不确定的对象仍可由几何证据复核。每格至少两个历史表面采样，至少 85% 的采样被四个及以上独立视角证实穿透，且没有稳定旧表面支持，才进入退役候选。当前障碍直接否决清空；至少三个地面观测才转为空闲，否则转未知。</p>
    <p>DeepSeek 实际输出：{html.escape(str(dict(states)))}。本轮没有模型明确判断“移走”；有效删除依据来自深度，不能把模型“不确定”解释为可通行。</p>
    <h2>验证及边界</h2><p>134 个清空格均有留出视角地面观测，留出障碍冲突为 0；允许范围外逐像素不变，上一版新增占用全部保留。留出视角来自同一次采集，且已用于开发评估，不等于独立复采验收。新增格增多来自更密集的深度采样，不代表所有噪点都已解决。</p>
    <p>已观测旧表面的消失不等于完整机器人通行体积验证。当前仍存在米制尺度冲突，遮挡处障碍可能遗漏；尚未验证机器人尺寸、定位误差与实际通行。134 格仅为部分删除改善，不能声称旧物已全部清干净。</p>
    <h2>全图与框选范围</h2><p>红色新增，蓝色清除，黄色转未知。</p><img src="grid_comparison.png">
    <p><a href="navigation_candidate.zip">下载候选栅格</a> · <a href="semantic_review.json">DeepSeek 明细</a> · <a href="navigation_update.json">更新证据统计</a> · <a href="verification.json">本地核验</a></p>'''
    (dest/'index.html').write_text(page,encoding='utf-8')
    return audit


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--destination',type=Path,required=True);p.add_argument('--previous',type=Path,required=True);a=p.parse_args();print(json.dumps(build(a.destination,a.previous)))
