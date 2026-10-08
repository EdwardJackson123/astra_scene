"""AI advisory + SAM image masks + multi-view geometric verification."""
from pathlib import Path
import json,re,html,fcntl
import numpy as np,cv2,torch
from scipy.spatial import cKDTree
from segment_anything import sam_model_registry,SamPredictor
from astra_scene.io import ROOT,independent,sha256
from astra_scene.postprocess import load,rows_from_data
from astra_scene.geometry import backproject,voxel_fuse,project,evidence
from astra_scene.roi import box_corners,frame_manifest
import argparse
parser=argparse.ArgumentParser();parser.add_argument("--source",type=Path,required=True);parser.add_argument("--output",type=Path,required=True);args=parser.parse_args()
R=args.source;O=args.output/"artifacts/fusion";O.mkdir(parents=True,exist_ok=True)
cfg,d,split,_=load(R);rows=rows_from_data(d)
protected={k:sha256(cfg[k]) for k in ["map_ply","nav_map","nav_yaml","descriptors"]}
ps=json.loads((args.output/"analysis/evidence.json").read_text())["proposals"]
ps=[p for p in ps if p.get("evidence_side","current")=="current"]
ai=json.loads((args.output/"analysis/ai_groups.json").read_text())
advice={i:g for g in ai["groups"] for i in g["member_ids"]}
ids=independent(rows,split["train_views"],.15)
crops={f["view"]:f["crop"] for f in frame_manifest(R)["frames"]}
masks={p["id"]:{} for p in ps};meta={p["id"]:{} for p in ps}
def rectangle(bounds,row):
 corners=box_corners(dict(bounds=bounds));cam=(corners-row["pose"][:3,3])@row["pose"][:3,:3]
 if (cam[:,2]<.1).any():return None
 uv=cam@row["K"].T;uv=uv[:,:2]/uv[:,2:]
 h,w=row["depth"].shape
 rect=np.r_[uv.min(0),uv.max(0)]
 if rect[2]<0 or rect[3]<0 or rect[0]>=w or rect[1]>=h:return None
 return np.clip(rect,[0,0,0,0],[w-1,h-1,w-1,h-1])
lock=(ROOT/"runs/gpu.lock").open("w");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
sam=sam_model_registry["vit_b"](checkpoint="/home/xujie/VPS_dynamic/runtime/game_weights/sam_vit_b_01ec64.pth").cuda().eval()
predictor=SamPredictor(sam)
for i in ids:
 row=rows[i];im=np.uint8(np.clip(row["rgb"]*255,0,255));x0,y0,x1,y1=crops[i];predictor.set_image(im[y0:y1,x0:x1])
 for p in ps:
  bounds=np.array(p["bounds"]);rect=rectangle(bounds,row)
  if rect is None or np.prod(rect[2:]-rect[:2])<400:continue
  # Bounded expansion can recover parts outside the original prompt.
  delta=(rect[2:]-rect[:2])*.15
  prompt=np.clip(rect+np.r_[-delta,delta],[x0,y0,x0,y0],[x1-1,y1-1,x1-1,y1-1])-np.array([x0,y0,x0,y0])
  if np.prod(prompt[2:]-prompt[:2])<100:continue
  with torch.no_grad():mm,score,_=predictor.predict(box=prompt,multimask_output=True)
  mask=np.zeros(im.shape[:2],bool);mask[y0:y1,x0:x1]=mm[int(np.argmax(score))];confidence=float(max(score))
  # Require high SAM score; reject near-full-image masks and tiny masks.
  if confidence<.8 or mask.mean()>.65 or mask.sum()<80:continue
  masks[p["id"]][i]=mask
  meta[p["id"]][i]=confidence
 print("SAM_VIEW",i,flush=True)
del predictor,sam;torch.cuda.empty_cache();lock.close()
np.savez_compressed(O/"sam_masks.npz",**{name+"_"+str(i):m for name,ms in masks.items() for i,m in ms.items()})
results=[];pointsets={}
for p in ps:
 name=p["id"];bounds=np.array(p["bounds"]);parts=[]
 for i,m in masks[name].items():
  xyz,_=backproject(rows[i],m,stride=4)
  xyz=xyz[((xyz>=bounds[0]-.25)&(xyz<=bounds[1]+.25)).all(1)]
  parts.append(xyz)
 xyz=np.concatenate(parts) if parts else np.empty((0,3))
 if len(xyz):xyz,_=voxel_fuse(xyz,np.zeros_like(xyz),.03)
 votes=np.zeros(len(xyz),int)
 for i,m in masks[name].items():
  ix,u,v,z=project(xyz,rows[i]);r=rows[i]
  good=m[v,u]&r["valid"][v,u]&(np.abs(r["depth"][v,u]-z)<=.08)
  votes[ix]+=good
 xyz=xyz[votes>=3]
 # Geometric contradiction check is independent of SAM identity.
 if len(xyz):
  support,free,_=evidence(xyz,rows,ids,.08)
  xyz=xyz[(support>=3)&(free==0)]
 pointsets[name]=xyz;np.save(O/(name+"_surface.npy"),xyz)
 new=bounds.copy();passed=len(xyz)>=80
 if passed:
  new=np.array([np.quantile(xyz,.005,axis=0)-.03,np.quantile(xyz,.995,axis=0)+.03])
  # Do not allow a segmentation fragment to shrink away the original object.
  new=np.array([np.minimum(bounds[0],new[0]),np.maximum(bounds[1],new[1])])
  passed=bool(np.max(new[1]-new[0])<=4 and np.prod(new[1]-new[0])<=12)
 if not passed:new=bounds.copy()
 images=[]
 for sel in p["selections"]:
  i=sel["view"];rgb=cv2.cvtColor(np.uint8(np.clip(rows[i]["rgb"]*255,0,255)),cv2.COLOR_RGB2BGR)
  original=rgb.copy();masked=rgb.copy();refined=rgb.copy()
  for im,b,color in [(original,bounds,(0,165,255)),(refined,new,(60,210,60))]:
   rect=rectangle(b,rows[i])
   if rect is not None:
    rect=np.rint(rect).astype(int);cv2.rectangle(im,tuple(rect[:2]),tuple(rect[2:]),color,2)
  m=masks[name].get(i)
  if m is not None:masked[m]=(.5*masked[m]+.5*np.array([220,200,30])).astype(np.uint8)
  file=f"{name}_{i:03d}.jpg";cv2.imwrite(str(O/file),np.concatenate([im[crops[i][1]:crops[i][3],crops[i][0]:crops[i][2]] for im in [original,masked,refined]],1))
  images.append(dict(view=i,file=file,mask_available=m is not None))
 g=advice[name]
 status="范围可供复核" if passed else "多视角支持不足，保留原框"
 if g["action"] in ["defer","low_priority"]:status="AI建议暂缓；修正范围仅供查看"
 results.append(dict(id=name,label=p["label"],old_bounds=bounds.tolist(),new_bounds=new.tolist(),
  surface_points=len(xyz),sam_views=len(masks[name]),geometry_supported=passed,
  status=status,ai_action=g["action"],ai_reason=g["reason"],images=images))
 print("REGION",name,len(xyz),passed,flush=True)
# Merge only explicitly suggested pairs with both geometric and segmentation agreement.
pairs=[];parent={p["id"]:p["id"] for p in ps}
def root(x):
 while parent[x]!=x:x=parent[x]
 return x
for a in ps:
 ga=advice[a["id"]];text=ga.get("reason","")+" "+ga.get("box_advice","")
 for b in ps:
  if a["id"]==b["id"] or b["id"] not in {"boxer_"+n for n in re.findall(r"(?<![0-9])([0-9]{3})(?![0-9])",text)}:continue
  key=tuple(sorted([a["id"],b["id"]]))
  if any(tuple(x["members"])==key for x in pairs):continue
  ma,mb=masks[key[0]],masks[key[1]];common=sorted(set(ma)&set(mb))
  overlaps=[float((ma[i]&mb[i]).sum()/max(1,(ma[i]|mb[i]).sum())) for i in common]
  pa,pb=pointsets[key[0]],pointsets[key[1]];coverage=0.
  if len(pa)>=80 and len(pb)>=80:
   coverage=float(min((cKDTree(pb).query(pa)[0]<.06).mean(),(cKDTree(pa).query(pb)[0]<.06).mean()))
  agree=sum(v>=.65 for v in overlaps)
  passed=agree>=3 and coverage>=.5
  pairs.append(dict(members=list(key),common_views=len(common),agree_views=agree,
   median_mask_iou=float(np.median(overlaps)) if overlaps else 0,mutual_surface_overlap=coverage,merge=passed))
  if passed:parent[root(key[1])]=root(key[0])
groups={}
for p in ps:groups.setdefault(root(p["id"]),[]).append(p["id"])
report=dict(method="DeepSeek existing advice + SAM ViT-B + 3-view estimated-depth consistency",
 regions=results,merge_checks=pairs,groups=list(groups.values()),sam_prompt_expansion_fraction=.15,
 max_3d_growth_m=.25,geometry_views=ids,minimum_support=3,minimum_sam_score=.8,
 new_capture_required=False,published=False,editor_modified=False,
 segmentation_ground_truth=False,navigation_verified=False,localization_verified=False,
 protected_hashes=protected,original_unchanged=all(sha256(cfg[k])==v for k,v in protected.items()))
(O/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2))
page='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>西创 · AI 与多视角融合选框</title><style>body{font:16px/1.65 system-ui;max-width:1300px;margin:24px auto;padding:16px;background:#eef2f6;color:#192637}article{padding:22px;background:white;border-radius:14px;margin:20px 0}img{width:100%}a{color:#1268ba}</style><h1>AI＋SAM＋三维几何：选框修正对比</h1>'
page+=f'<p>{len(ps)} 个原始建议 → {len(groups)} 组；{sum(x["geometry_supported"] for x in results)} 个框具有至少三个视角的表面支持。修正框是实验建议，尚未写入编辑器或地图。</p><article>每行左：原框（橙）；中：SAM 分割（青）；右：几何支持的修正框（绿）。均为对应实拍。没有分割颜色的视角未得到有效掩码。仅使用现有训练视角，不能作为独立质量验收。</article>'
page+='<article><h2>合并检查</h2>'
for x in pairs:page+=f'<p>{html.escape(" + ".join(x["members"]))}：'+("通过" if x["merge"] else "未通过，保留分开")+f'；一致视角 {x["agree_views"]}，掩码 IoU 中位数 {x["median_mask_iou"]:.2f}，双向表面重叠 {x["mutual_surface_overlap"]:.2f}。</p>'
page+='</article>'
for p in results:
 page+=f'<article><h2>{html.escape(p["label"])} · {p["id"]}</h2><p>{p["status"]}。有效分割视角 {p["sam_views"]}，三视角几何点 {p["surface_points"]}。</p><details><summary>AI建议</summary>{html.escape(p["ai_reason"])}</details>'
 for im in p["images"]:page+=f'<a href="{im["file"]}" target="_blank"><img loading="lazy" src="{im["file"]}"></a><p>视角 {im["view"]}</p>'
 page+='</article>'
page+='<p>限制：SAM 不保证实例身份，估计深度不是真值；扩框可能包含邻物。当前策略不自动缩框、不清除旧障碍、不把可移动物体认定为正在移动。导航与定位任务仍待实际配置验证。</p><a href="report.json">完整几何与合并报告</a></html>'
(O/"index.html").write_text(page)
print("DONE",len(groups),report["original_unchanged"],flush=True)
