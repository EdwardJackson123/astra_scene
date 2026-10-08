from pathlib import Path
import json,uuid,time,shutil
from flask import Blueprint,request,jsonify
from .io import ROOT,write_json,sha256
from .maps import maps,config_path,register,add_references
from .workflow import enqueue,safe_name
def blueprint(root):
 root=Path(root);bp=Blueprint("intake",__name__)
 @bp.get("/api/maps")
 def listing():
  try:return jsonify(maps=maps(root),ai_configured=(root/"configs/secrets/deepseek.key").is_file())
  except (ValueError,OSError,KeyError,TypeError) as e:return jsonify(error="读取VPS地图配置失败："+str(e)),400
 @bp.post("/api/maps")
 def importing():
  try:return jsonify(register(root,request.files,request.form.get("name",""),request.form.get("coordinates")=="yes")),201
  except (ValueError,OSError,KeyError,TypeError) as e:return jsonify(error=str(e)),400
 @bp.post("/api/maps/<map_id>/references")
 def references(map_id):
  try:
   file=request.files.get("references")
   if file is None or not file.filename:raise ValueError("请选择参考资料ZIP")
   return jsonify(add_references(root,map_id,file))
  except (ValueError,OSError,KeyError,TypeError) as e:return jsonify(error=str(e)),400
 @bp.get("/api/captures")
 def captures():
  rows=[]
  for p in (root/"runs").glob("*/reconstruction.json"):
   cfgpath=p.parent/"config.json"
   if not cfgpath.exists():continue
   cfg=json.loads(cfgpath.read_text())
   rows.append(dict(id=p.parent.name,map_id=cfg["map_id"],title="已有采集 · "+p.parent.name))
  return jsonify(rows)
 @bp.post("/api/generate")
 def generate():
  folder=None
  try:
   map_id=request.form.get("map_id","xichuang0915");cfgpath=config_path(root,map_id);cfg=json.loads(cfgpath.read_text())
   if not cfg.get("registration",{}).get("ready",True):raise ValueError("地图资料尚未补齐，不能自动对齐")
   if not (root/"configs/secrets/deepseek.key").is_file():raise ValueError("AI接口尚未配置")
   choice=request.form.get("sh_degree","auto")
   if choice not in ["auto","0","1","2"]:raise ValueError("SH参数无效")
   from .spirula import ply_degree
   degree=ply_degree(cfg["map_ply"])
   if degree>2 or (choice!="auto" and int(choice)<degree):raise ValueError("不能降低原图SH阶数；当前支持0–2阶")
   name="auto_"+time.strftime("%Y%m%d_%H%M%S_")+uuid.uuid4().hex[:6]
   command=[cfg["vps_python"],"-u","-m","astra_scene.automatic","--output",str(root/"runs"/name),"--gpu","1","--sh-degree",choice]
   source=request.form.get("source_run","")
   if source:
    safe_name(source);p=root/"runs"/source
    if request.files:raise ValueError("已有采集与新上传不能同时选择")
    if not (p/"reconstruction.json").exists() or json.loads((p/"config.json").read_text())["map_id"]!=map_id:raise ValueError("已有采集与所选地图不一致")
    oldcfg=json.loads((p/"config.json").read_text())
    for key in ["map_ply","nav_map","nav_yaml","ref_root","descriptors"]:
     if Path(oldcfg[key]).resolve()!=Path(cfg[key]).resolve():raise ValueError("原地图配置已改变，请上传新采集")
    from .spirula import asset
    spec=asset(p,True)
    if choice!="auto" and int(choice)!=spec["output_sh_degree"]:raise ValueError("复用采集不能改变已训练SH，请上传新采集")
    command+=["--source",str(p)]
    identity="reuse:"+source+":"+spec["key"];title="自动更新 · 已有采集"
   else:
    files=request.files.getlist("inputs");kind=request.form.get("input_kind","video")
    if kind not in ["video","images"]:raise ValueError("请选择视频或照片")
    if (kind=="video" and len(files)!=1) or (kind=="images" and not 14<=len(files)<=200):raise ValueError("请上传1段视频，或14–200张照片")
    folder=root/"uploads"/name;folder.mkdir(parents=True)
    relative=[]
    if kind=="images":
     try:relative=json.loads(request.form.get("relative_paths","[]"))
     except ValueError:raise ValueError("照片文件夹清单无效")
     if not isinstance(relative,list) or relative and (len(relative)!=len(files) or any(not isinstance(x,str) or len(x)>1024 or "\x00" in x for x in relative)):raise ValueError("照片文件夹清单无效")
    saved=[]
    for i,f in enumerate(files):
     suffix=Path(f.filename or "").suffix.lower()
     allowed=[".mp4",".mov",".m4v"] if kind=="video" else [".jpg",".jpeg",".png"]
     if suffix not in allowed:raise ValueError("文件格式不支持")
     dest=folder/(f"{i:04d}"+suffix);f.save(dest);size=dest.stat().st_size
     if not size or size>(1024**3 if kind=="video" else 30*1024**2):raise ValueError("视频须小于1GB，单张照片须小于30MB")
     saved.append(dest)
    if sum(x.stat().st_size for x in saved)>1024**3:raise ValueError("采集文件总量须小于1GB")
    if kind=="images":
     sourcefile=folder/"photos.json";write_json(sourcefile,dict(images=[str(x) for x in saved],original_relative_paths=relative))
    else:sourcefile=saved[0]
    identity="input:"+":".join(sha256(x) for x in saved)
    command+=["--input",str(sourcefile),"--config",str(cfgpath)]
    title="自动更新 · "+(str(len(saved))+"张照片" if kind=="images" else Path(files[0].filename).name[:80])
   dedupe="automatic-v2-context:"+identity+":"+sha256(cfgpath)+":"+choice
   result=enqueue(root,dict(id=name,kind="automatic",title=title,map_id=map_id,backend="场景范围识别 + 连续区域联合更新 / SH跟随原图"),command,dedupe,cfg["vps_python"])
   if result["reused"] and folder:shutil.rmtree(folder)
   return jsonify(result),202
  except (ValueError,OSError,KeyError,TypeError) as e:
   if folder and folder.exists():shutil.rmtree(folder)
   return jsonify(error=str(e)),400
 return bp
