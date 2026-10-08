"""Read-only VPS scene discovery; immutable task configs live only in Astra."""
from pathlib import Path
import copy,hashlib,json
import yaml
from .io import write_json
from .workflow import safe_name

FIELDS={"ref_root":"ref_data_path","descriptors":"ref_descriptors_path","ref_cache":"vggt_omega_ref_cache_path","map_ply":"gaussian_ply_path","nav_map":"nav_map_path","nav_yaml":"nav_yaml_path"}

def discover(root):
 root=Path(root);setting=root/"configs/vps_catalog.json"
 if not setting.exists():return None
 source=Path(json.loads(setting.read_text())["source"])
 data=yaml.safe_load(source.read_text())
 entries=data.get("service",{}).get("maps",[])
 if not isinstance(entries,list):raise ValueError("VPS service.maps 必须为列表")
 rows=[];seen=set()
 for entry in entries:
  logical=str(entry["id"]);safe_name(logical)
  variants=entry.get("light_variants") or {"default":{}}
  if not isinstance(variants,dict):raise ValueError("VPS 光照配置无效")
  for variant,extra in variants.items():
   variant=str(variant);safe_name(variant)
   ident=logical if len(variants)==1 else logical+"__"+variant
   safe_name(ident)
   if ident in seen:raise ValueError("VPS 地图标识重复："+ident)
   seen.add(ident);assets={**entry,**(extra or {})};assets.pop("light_variants",None)
   paths={}
   for dest,key in FIELDS.items():
    raw=assets.get(key)
    paths[dest]=str((source.parent.parent/str(raw)).resolve()) if raw else None
   # Navigation metadata follows the grid's config.yaml, never legacy map.yaml.
   paths["nav_yaml"]=str(Path(paths["nav_map"]).parent/"config.yaml") if paths["nav_map"] else None
   missing=[label for key,label in [("map_ply","PLY"),("nav_map","栅格图"),("nav_yaml","栅格config.yaml"),("descriptors","参考索引")] if not paths[key] or not Path(paths[key]).is_file()]
   ref=Path(paths["ref_root"]) if paths["ref_root"] else None
   if not ref or not (ref/"rgb").is_dir() or not (ref/"poses").is_dir():missing.append("参考图像/位姿")
   degree=None
   if not missing:
    from .spirula import ply_degree
    try:
     degree=ply_degree(paths["map_ply"])
     if degree not in [0,1,2]:missing.append("当前不支持的SH阶数")
    except Exception:missing.append("无效PLY")
   if paths["ref_cache"] and not Path(paths["ref_cache"]).is_file():paths["ref_cache"]=None
   rows.append(dict(id=ident,name=logical+(" · "+variant if len(variants)>1 else ""),ready=not missing,
    reason="缺少或无效："+ "、".join(missing) if missing else "",sh_degree=degree,built_in=True,source="VPS default",
    source_config=str(source),logical_map_id=logical,variant=variant,can_complete=False,_paths=paths,_assets=assets))
 return rows

def materialize(root,row):
 root=Path(root)
 cfg=json.loads((root/"configs/xichuang0915.json").read_text())
 cfg.update(row["_paths"]);cfg.update(map_id=row["id"],floor_z=None,floor_verified=False)
 cfg["registration"]={k:v for k,v in row.items() if not k.startswith("_")}
 base=yaml.safe_load(Path(cfg["vps_config"]).read_text())
 asset=copy.deepcopy(row["_assets"]);asset["id"]=row["id"]
 for dest,key in FIELDS.items():asset[key]=row["_paths"][dest]
 base.setdefault("service",{})["maps"]=[asset]
 engine=yaml.safe_dump(base,allow_unicode=True,sort_keys=False)
 digest=hashlib.sha256((json.dumps(cfg,sort_keys=True)+engine).encode()).hexdigest()[:20]
 folder=root/"configs/vps_scenes"/digest;folder.mkdir(parents=True,exist_ok=True)
 cfg["vps_config"]=str(folder/"engine.yaml")
 if not (folder/"engine.yaml").exists():(folder/"engine.yaml").write_text(engine)
 if not (folder/"config.json").exists():write_json(folder/"config.json",cfg)
 return folder/"config.json"
