"""Read-only progress from stage status and bounded log tails."""
import json,re,time
from pathlib import Path
LABELS={"context_fusion":"大区域融合与接缝联合优化","fusion_guard":"逐区域检查融合回退","object_policy":"比较背景扩张并检查旧物体位置","object_completion":"补全部件并核验旧位置","extract":"视频抽帧","localize":"定位到原地图","reconstruct":"几何重建","anchor_render":"准备原地图对照","bundle":"校正位姿","geometry_check":"几何检查","spirula_sfm":"求解相机位姿（SfM）","spirula_align":"对齐原地图坐标","spirula_train":"训练 3DGS","spirula_validate":"验证训练检查点","auto_evidence":"检测新旧变化","ai_review":"AI审核变化与完整性","auto_fusion":"分割及多视角核验","candidate":"生成更新候选","references":"生成定位图库","artifacts":"检查模型与导航候选","finalize":"验收候选包","capture":"准备采集"}
def read(p):
 try:return json.loads(p.read_text())
 except (OSError,ValueError):return {}
def tail(p):
 try:
  with p.open("rb") as f:f.seek(max(0,p.stat().st_size-16000));return f.read().decode(errors="replace")
 except OSError:return ""
def details(root,name,state,kind):
 root=Path(root);run=root/"runs"/name;source=run;stage=state.get("stage","");updated=state.get("updated")
 busy=state.get("state") in ["running","starting","queued"]
 if kind=="automatic" and busy and stage=="capture":
  child=root/"runs"/(name+"_capture");cs=read(child/"status.json")
  if cs:source=child;stage=cs.get("stage",stage);updated=cs.get("updated",updated)
 if kind=="automatic" and busy and stage=="candidate":
  child=root/"runs"/(name+"_candidate");cs=read(child/"status.json")
  if cs:source=child;stage=cs.get("stage",stage);updated=cs.get("updated",updated)
 if stage=="spirula_sfm" and "DONE spirula_sfm" in tail(source/"spirula.log"):stage="spirula_align"
 result=dict(stage=stage,label=LABELS.get(stage,stage or "等待启动"),updated=updated,training=None,stale=False)
 if busy and stage=="spirula_train":
  p=source/"spirula_train.log";s=tail(p)
  matches=list(re.finditer(r"step\s+(\d+)/(\d+)\s+\(\s*(\d+)%\).*?elapsed\s+([^|\]]+)\| ETA\s+([^\]]+)",s))
  if matches:
   m=matches[-1];result["training"]=dict(step=int(m[1]),total=int(m[2]),percent=int(m[3]),elapsed=m[4].strip(),eta=m[5].strip())
   result["updated"]=p.stat().st_mtime
 if busy and stage=="object_policy":
  p=run/"artifacts/object_policy/progress.json";progress=read(p)
  if progress:
   result["label"]="逐区域比较背景与旧位置 · 已处理 "+str(progress.get("trials",0))+" 项"
   result["updated"]=p.stat().st_mtime
 if busy and stage=="fusion_guard":
  p=run/"artifacts/fusion_guard/progress.json";progress=read(p)
  if progress:
   result["label"]="融合回退检查 · "+str(progress.get("completed",0))+" / "+str(progress.get("total",0))
   result["updated"]=p.stat().st_mtime
 if busy and stage=="context_fusion":
  p=source/"context_fusion/progress.json";progress=read(p)
  if progress:
   result["label"]="大区域联合优化 · "+str(progress.get("step",0))+" / "+str(progress.get("total",320))+" 步"
   result["updated"]=p.stat().st_mtime
 if busy and result["updated"]:result["stale"]=time.time()-result["updated"]>180
 return result
