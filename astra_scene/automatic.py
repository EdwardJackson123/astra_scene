"""One-click orchestration: prepare -> detect -> AI -> geometry -> isolated candidate."""
from pathlib import Path
import argparse,json,base64,html,time,fcntl,shutil
import numpy as np,cv2,requests
from .io import ROOT,write_json,sha256
from .pipeline import environment,run_stage,build
def status(run,stage,**extra):write_json(run/"status.json",dict(state="running",stage=stage,updated=time.time(),**extra))
def evidence_stage(source,run):
 from .proposals import boxes_path,proposals,detect
 from .postprocess import load,rows_from_data,render_map
 from .geometry import discover
 from .roi import frame_manifest
 cfg,d,split,_=load(source)
 if boxes_path(source) is None:
  out=source/"auto_detection";write_json(source/"region_detection_pointer.json",dict(output=str(out)))
  detect(source,out,1)
 allp=proposals(source)["proposals"]+proposals(source,side="old")["proposals"];ps=allp
 rows=rows_from_data(d);frames={f["view"]:f for f in frame_manifest(source)["frames"]}
 from vps.refinement.gaussian_renderer import GsplatGaussianRenderer
 with (ROOT/"runs/gpu.lock").open("a") as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);renderer=GsplatGaussianRenderer("cuda");base={}
  for i in split["train_views"]:base[i]=render_map(renderer,cfg["map_ply"],rows[i])
  points,_,audit=discover(rows,split["train_views"],base,cfg)
 del renderer
 out=run/"artifacts/evidence";out.mkdir(parents=True,exist_ok=True)
 for p in ps:
  bounds=np.array(p["bounds"]);p["novelty_points"]=int(((points>=bounds[0])&(points<=bounds[1])).all(1).sum())
  p["images"]=[]
  for sel in p["selections"]:
   i=sel["view"];x0,y0,x1,y1=frames[i]["crop"]
   old=base[i]["rgb"][y0:y1,x0:x1].copy();real=np.uint8(np.clip(d["rgb"][i,y0:y1,x0:x1]*255,0,255))
   h,w=real.shape[:2];rect=np.rint(np.array(sel["rect"])*[w,h,w,h]).astype(int)
   for im in [old,real]:cv2.rectangle(im,tuple(rect[:2]),tuple(rect[2:]),(255,165,0),2)
   file=f'{p["id"]}_{i:03d}.jpg'
   cv2.imwrite(str(out/file),cv2.cvtColor(np.concatenate([old,real],1),cv2.COLOR_RGB2BGR))
   p["images"].append(dict(view=i,file=file))
 write_json(run/"analysis/evidence.json",dict(proposals=ps,total_detected=len(allp),unreviewed=max(0,len(allp)-len(ps)),discovery=audit,depth="estimated"))
def ai_stage(run):
 from .task_review import analyze
 analyze(run)

def render_report(run,source,child=None,blocked=None,review_only=False):
 data=json.loads((run/"analysis/evidence.json").read_text());ai=json.loads((run/"analysis/ai.json").read_text())
 a=run/"artifacts";a.mkdir(exist_ok=True)
 page='<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>自动更新结果</title><style>body{font:16px/1.65 system-ui;max-width:1150px;margin:30px auto;padding:20px;background:#f3f6f7;color:#203a38}section{background:white;padding:24px;margin:18px 0;border-radius:14px}img{max-width:100%}a{color:#166b63}summary{cursor:pointer}</style><h1>自动分析与更新结果</h1>'
 if review_only:
  page+='<section><h2>AI筛选对照 · 尚未生成地图候选</h2><p>此页只验证任务影响筛选。推荐项目仍须通过分割、几何覆盖和更新后回归，原地图未修改。</p></section>'
 elif child:
  page+='<section><h2>局部候选已生成</h2><p>请分别检查模型、导航和定位结果。自动生成不等于验收通过；未替换原数据。</p><a href="/results/'+child.name+'/index.html">查看实拍与更新前后效果、下载候选</a></section>'
 else:
  uncertain=any(x["decision"]=="defer" for x in ai["items"])
  page+='<section><h2>'+('有待复核的变化，尚未生成候选' if uncertain or blocked else '本轮未选出值得更新的区域')+'</h2><p>没有选区同时通过任务影响筛选和几何覆盖检查；这不代表场景没有变化。原场景保持不变。</p></section>'

 page+='<section><h2>更新范围说明</h2><p>本轮同时检查当前物体和原地图旧位置。旧位置变化、搬移关联或部件不完整会进入待复核，系统不会将其当成纯新增，也不会自动清空未确认区域。AI失效不代表没有变化。</p></section>'
 page+='<p>共发现 '+str(data["total_detected"])+' 个物体建议；本轮检查 '+str(len(data["proposals"]))+' 个；未检查 '+str(data["unreviewed"])+' 个。检测覆盖不代表全部变化已覆盖。</p>'
 for x in ai["items"]:
  p=next(p for p in data["proposals"] if p["id"]==x["id"])
  label={"review_update":"优先复核","deprioritize":"本批不推荐","defer":"待补证据"}[x["decision"]]
  page+='<section><details'+(' open' if x["decision"]=="review_update" else '')+'><summary>'+html.escape(p["label"]+" · "+label)+'</summary><p>'+html.escape(x["reason"])+'</p><p>AI意见，仅作筛选依据；定位影响未实测。</p>'
  side="旧位置检查" if p.get("evidence_side")=="old" else "当前位置检查"
  page+='<p>'+side+' · 变化类型：'+html.escape({"add":"新增","remove":"移除","move":"搬移","appearance":"外观","uncertain":"待确认"}.get(x.get("change_type"),"待确认"))+'</p>'
  if x.get("missing_parts"):page+='<p>待补全部件：'+html.escape("、".join(map(str,x["missing_parts"])))+'</p>'
  if x.get("ai_validation_failed") or (x.get("audit") or {}).get("ai_validation_failed"):page+='<p>AI输出校验未通过，本项保留待复核，不据此增删地图。</p>'
  if "localization_relevance" in x:
   labels={"potential":"可能相关，待实测","low":"低相关","unknown":"证据不足"}
   page+='<p>导航：'+labels[x["navigation_relevance"]]+'；VPS定位：'+labels[x["localization_relevance"]]+'</p>'
   for obs in x.get("observations",[]):page+='<p>'+html.escape(str(obs))+'</p>'
   if x.get("audit"):page+='<p>第二轮复核：'+html.escape(x["audit"]["reason"])+'</p>'
  for im in p["images"]:page+='<a target="_blank" href="evidence/'+im["file"]+'"><img loading="lazy" src="evidence/'+im["file"]+'"></a><p>左原地图／右实拍。</p>'
  if blocked and x["id"] in blocked:page+='<p>本项暂缓：'+html.escape(blocked[x["id"]])+'</p>'
  page+='</details></section>'
 completion=read_completion=run/"artifacts/completion/report.json"
 if completion.exists():
  c=json.loads(completion.read_text())
  page+='<section><h2>整套设备补全与旧位置关联</h2><p>已继续处理被暂缓的设备候选，共 '+str(len(c.get("groups",[])))+' 组。补全预览尚未通过整物体验收，不代表整个场景更新完成。</p><a href="completion/index.html">查看多区域补全、旧位置检查和实拍对比</a></section>'
 if (run/"artifacts/object_policy/report.json").exists():page+='<section><h2>自动整物体范围检查</h2><a href="object_policy/index.html">查看背景扩张与旧椅子位置的自动比较</a><p>整物体范围已接入待验收候选，覆盖不足会明确标记；候选内同时生成3DGS、栅格和定位图库，尚未发布。</p></section>'
 guard_path=run/"artifacts/fusion_guard/report.json"
 if guard_path.exists():
  guard=json.loads(guard_path.read_text())
  page+='<section><h2>融合回退检查</h2><p>逐区域检查已选更新是否造成外观回退；暂缓 '+str(len(guard["deferred"]))+' 项。暂缓不代表没有变化，仍需补证据。此检查不替代几何验收。</p><a href="fusion_guard/report.json">查看逐区域对照记录</a></section>'
 backend_path=run/"fusion_backend.json"
 if backend_path.exists():
  selected_backend=json.loads(backend_path.read_text())
  if selected_backend["backend"]=="context":
   page+='<section><h2>连续区域联合更新</h2><p>物体检测用于定位更新范围；最终使用包含周边上下文的连续过渡带，联合优化区域与接缝，不逐个物体撤回更新。导航和定位图库仍独立检查。</p></section>'
  elif selected_backend.get("note"):
   page+='<p>'+html.escape(selected_backend["note"])+'</p>'
 page+='<p><a href="/select3d/'+source.name+'">高级：手动调整选区</a> · <a href="fusion/index.html">查看分割与几何修正</a></p>'
 (a/"index.html").write_text(page)
 write_json(a/"automatic.json",dict(source=source.name,candidate=child.name if child else None,blocked=blocked or {},published=False))
def execute(run,config=None,input_path=None,source=None,gpu=1,sh_degree="auto"):
 run=Path(run);run.mkdir(parents=True,exist_ok=False);(run/"analysis").mkdir()
 try:
  if source is None:
   source=ROOT/"runs"/(run.name+"_capture");cfg=json.loads(Path(config).read_text())
   run_stage(run,"capture",[cfg["vps_python"],"-m","astra_scene.pipeline","--video",input_path,"--output",source,"--config",config,"--geometry-only","--spirula","--sh-degree",sh_degree,"--gpu",gpu],environment(gpu))
  source=Path(source)
  from .spirula import asset
  reconstruction=asset(source,True)
  write_json(run/"automatic_source.json",dict(source=source.name))
  cfg=json.loads((source/"config.json").read_text());vp=cfg["vps_python"]
  from .object_candidate import resolve_backend
  backend,backend_note=resolve_backend(cfg,reconstruction)
  write_json(run/"fusion_backend.json",dict(backend=backend,note=backend_note))
  run_stage(run,"auto_evidence",[vp,"-m","astra_scene.automatic","--phase","evidence","--source",source,"--output",run],environment(gpu))
  status(run,"ai_review");ai_stage(run)
  run_stage(run,"auto_fusion",[cfg["game_python"],"-m","astra_scene.auto_fusion","--source",source,"--output",run],environment(gpu))
  run_stage(run,"object_completion",[vp,"-m","astra_scene.completion","--source",source,"--output",run],environment(gpu))
  run_stage(run,"object_policy",[vp,"-m","astra_scene.object_policy","--source",source,"--output",run],environment(gpu))
  ai=json.loads((run/"analysis/ai.json").read_text());fusion=json.loads((run/"artifacts/fusion/report.json").read_text())
  from .roi import preview,update
  from .deletion import render_request,render_worker
  from .postprocess import load
  selected=[];blocked={}
  for item in ai["items"]:
   if item["decision"]!="review_update":continue
   if item.get("evidence_side")=="old":blocked[item["id"]]="旧位置移除需要整物体范围与腾空证据核验";continue
   g=next((x for x in fusion["regions"] if x["id"]==item["id"]),None)
   if g is None:blocked[item["id"]]="尚未获得当前物体分割结果";continue
   if not g["geometry_supported"]:blocked[item["id"]]="多视角分割和几何支持不足";continue
   payload=dict(id=item["id"],label=g["label"],bounds=g["new_bounds"],mode="replace")
   try:
    s=preview(source,payload)
    if not s["coverage"]["ready"]:blocked[item["id"]]=s["coverage"]["message"];continue
   except ValueError as e:blocked[item["id"]]=str(e);continue
   if len(selected)>=8:blocked[item["id"]]="本批超过8个区域，保留待后续处理";continue
   selected.append(payload)
  policy=json.loads((run/"artifacts/object_policy/report.json").read_text())
  completed=policy.get("regions",[])
  blocked.update(policy.get("blocked",{}))
  # The same selected union drives the PLY, occupancy grid, and reference candidate.
  selected=completed+[x for x in selected if x["id"] not in {r["id"] for r in completed}]
  if len(selected)>32:
   for item in selected[32:]:blocked[item["id"]]="本批超过32个区域，保留待复核"
   selected=selected[:32]
  child=None
  if selected and backend=="scoped":
   status(run,"fusion_guard")
   write_json(run/"fusion_guard_regions.json",selected)
   run_stage(run,"fusion_guard",[vp,"-m","astra_scene.fusion_guard","--source",source,"--regions",run/"fusion_guard_regions.json","--output",run/"artifacts/fusion_guard"],environment(gpu))
   guard=json.loads((run/"artifacts/fusion_guard/report.json").read_text())
   selected=guard["regions"]
   for ident in guard["deferred"]:blocked[ident]="该区域替换导致训练视角外观回退，已暂缓，仍需补证据"
   policy={**policy,"fusion_guard":guard,"regions":selected}
  if selected:
   status(run,"candidate")
   from .object_candidate import build_candidate
   child=ROOT/"runs"/(run.name+"_candidate")
   policy={**policy,"fusion_backend":backend,"backend_note":backend_note}
   build_candidate(source,child,selected,gpu,policy,backend=backend)
  render_report(run,source,child,blocked)
  write_json(run/"status.json",dict(state="needs_review" if child or blocked or any(x["decision"]=="defer" for x in ai["items"]) else "no_update",stage="complete",updated=time.time()))
 except Exception as e:
  write_json(run/"status.json",dict(state="failed",stage="failed",error=str(e),updated=time.time()));raise
def main():
 p=argparse.ArgumentParser();p.add_argument("--output",required=True,type=Path);p.add_argument("--source",type=Path);p.add_argument("--input",type=Path);p.add_argument("--config",type=Path);p.add_argument("--gpu",type=int,default=1);p.add_argument("--sh-degree",default="auto");p.add_argument("--phase",choices=["run","evidence"],default="run")
 a=p.parse_args()
 if a.phase=="evidence":evidence_stage(a.source,a.output)
 else:execute(a.output,a.config,a.input,a.source,a.gpu,a.sh_degree)
if __name__=="__main__":main()
