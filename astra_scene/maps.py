"""Map registry and bounded, validated asset imports."""
from pathlib import Path,PurePosixPath
import json,uuid,zipfile,stat,math,shutil
import numpy as np,cv2,yaml,h5py
from .io import ROOT,write_json,rigid
from .workflow import safe_name
from .spirula import ply_degree
def config_path(root,map_id):
 safe_name(map_id)
 from .vps_catalog import discover,materialize
 catalog=discover(root)
 if catalog is not None:
  for row in catalog:
   if row["id"]==map_id:return materialize(root,row)
 p=Path(root)/"maps"/map_id/"config.json"
 if p.exists():return p
 if catalog is None and map_id=="xichuang0915":return Path(root)/"configs/xichuang0915.json"
 raise ValueError("地图不存在")
def maps(root):
 root=Path(root)
 from .vps_catalog import discover
 catalog=discover(root)
 out=[{k:v for k,v in row.items() if not k.startswith("_")} for row in (catalog or [])]
 paths=([root/"configs/xichuang0915.json"] if catalog is None else [])+list((root/"maps").glob("*/config.json"))
 for p in paths:
  if not p.exists():continue
  cfg=json.loads(p.read_text());meta=cfg.get("registration",{})
  if any(row["id"]==cfg["map_id"] for row in out):continue
  out.append(dict(id=cfg["map_id"],name=meta.get("name",cfg["map_id"]),ready=meta.get("ready",True),
   reason=meta.get("reason",""),sh_degree=meta.get("sh_degree",0),built_in=not bool(meta)))
 return out
def extract_references(path,target):
 with zipfile.ZipFile(path) as z:
  infos=z.infolist()
  if len(infos)>30000 or sum(i.file_size for i in infos)>12*1024**3:raise ValueError("参考资料解压后过大")
  for i in infos:
   name=PurePosixPath(i.filename)
   if name.is_absolute() or ".." in name.parts or "\\" in i.filename or stat.S_ISLNK(i.external_attr>>16):
    raise ValueError("参考压缩包包含不安全路径")
   if i.is_dir():continue
   if not (name.parts[0] in ["rgb","poses"] or i.filename=="ref.h5"):raise ValueError("参考ZIP仅应包含rgb/、poses/和ref.h5")
   dest=target.joinpath(*name.parts);dest.parent.mkdir(parents=True,exist_ok=True)
   with z.open(i) as src,dest.open("wb") as out:shutil.copyfileobj(src,out)
 rgb=list((target/"rgb").glob("*"));valid=[]
 if len(rgb)<6 or not (target/"ref.h5").is_file():raise ValueError("需要至少6张参考图片、对应C2W位姿和ref.h5")
 with h5py.File(target/"ref.h5","r") as index:
  for im in rgb:
   if im.suffix.lower() not in [".jpg",".jpeg",".png"] or cv2.imread(str(im)) is None:raise ValueError("参考图片无效")
   pose=target/"poses"/(im.stem+".txt")
   if not pose.exists():raise ValueError("参考图片缺少同名4×4 C2W位姿")
   rigid(np.loadtxt(pose).reshape(4,4))
   if im.name not in index or "global_descriptor" not in index[im.name]:raise ValueError("参考索引与图片名称不匹配")
   desc=index[im.name]["global_descriptor"][()]
   if desc.shape!=(8448,) or not np.isfinite(desc).all():raise ValueError("需要现有MegaLoc 8448维参考索引")
   valid.append(im.name)
 return len(valid)
def register(root,files,name,coordinate_confirmed=False):
 root=Path(root)
 if not coordinate_confirmed:raise ValueError("请确认PLY、栅格和参考位姿使用同一米制坐标系")
 for k in ["ply","grid","yaml"]:
  if not files.get(k) or not files[k].filename:raise ValueError("请提供PLY、栅格图片及YAML")
 if len(name.strip())<1:raise ValueError("请输入地图名称")
 ident="map_"+uuid.uuid4().hex[:12];folder=root/"maps"/ident;folder.mkdir(parents=True)
 try:
  files["ply"].save(folder/"scene.ply");degree=ply_degree(folder/"scene.ply")
  if degree not in [0,1,2]:raise ValueError("当前自动流程支持SH 0–2阶，不会隐式降低原图阶数")
  from .postprocess import read_gaussians
  vertices,_,_=read_gaussians(folder/"scene.ply")
  if len(vertices)<100:raise ValueError("PLY不是有效的场景高斯模型")
  files["grid"].save(folder/"grid.upload");grid=cv2.imread(str(folder/"grid.upload"),0)
  if grid is None or grid.size>100000000:raise ValueError("栅格图片无效或过大")
  cv2.imwrite(str(folder/"map.png"),grid)
  raw=files["yaml"].read(65537)
  if len(raw)>65536:raise ValueError("YAML过大")
  meta=yaml.safe_load(raw)
  if not isinstance(meta,dict):raise ValueError("地图YAML无效")
  res=float(meta.get("resolution",0));origin=np.asarray(meta.get("origin"),float)
  if not math.isfinite(res) or not 0<res<=5 or origin.shape!=(3,) or not np.isfinite(origin).all():raise ValueError("YAML分辨率或原点无效")
  low=float(meta.get("free_thresh",.196));high=float(meta.get("occupied_thresh",.65))
  if not 0<=low<high<=1 or meta.get("negate",0) not in [0,1]:raise ValueError("YAML占用阈值无效")
  meta=dict(image="map.png",resolution=res,origin=origin.tolist(),negate=meta.get("negate",0),free_thresh=low,occupied_thresh=high)
  (folder/"map.yaml").write_text(yaml.safe_dump(meta))
  refs=files.get("references");ready=bool(refs and refs.filename);count=0
  if ready:
   refs.save(folder/"references.zip");count=extract_references(folder/"references.zip",folder/"reference")
  cfg=json.loads((root/"configs/xichuang0915.json").read_text())
  cfg.update(map_id=ident,map_ply=str(folder/"scene.ply"),nav_map=str(folder/"map.png"),nav_yaml=str(folder/"map.yaml"),
   ref_root=str(folder/"reference"),descriptors=str(folder/"reference/ref.h5"),ref_cache=None,
   registration=dict(name=name[:80],ready=ready,sh_degree=degree,reference_count=count,coordinate_user_confirmed=True,
    reason="" if ready else "缺少参考图片、位姿和索引，暂不能自动对齐；可在地图卡片补齐。"))
  write_json(folder/"config.json",cfg)
  return dict(id=ident,name=name[:80],ready=ready,reason=cfg["registration"]["reason"])
 except Exception:
  shutil.rmtree(folder);raise
def add_references(root,map_id,file):
 root=Path(root);p=config_path(root,map_id);cfg=json.loads(p.read_text())
 if not cfg.get("registration") or cfg["registration"].get("ready"):raise ValueError("该地图无需补齐参考资料")
 folder=p.parent;staging=folder/("references_"+uuid.uuid4().hex[:8]);staging.mkdir()
 try:
  file.save(staging/"input.zip");count=extract_references(staging/"input.zip",staging/"data")
  (staging/"data").rename(folder/"reference")
  cfg["registration"].update(ready=True,reason="",reference_count=count);write_json(p,cfg)
 finally:shutil.rmtree(staging)
 return dict(id=map_id,ready=True)
