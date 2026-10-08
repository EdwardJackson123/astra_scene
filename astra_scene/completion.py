"""Object completion pass. Produces isolated previews; uncertain associations never authorize publication."""
from pathlib import Path
import argparse,json,fcntl,html
import numpy as np,cv2
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from .io import ROOT,write_json
from .postprocess import load,rows_from_data,read_gaussians,render_map
from .geometry import voxel_fuse,evidence
from .roi import preview,frame_manifest
from .spirula import fuse,asset

STRUCTURES={"easel","robot","metal frame","equipment cart","cart","trolley"}
def overlap(a,b):
 a=np.asarray(a);b=np.asarray(b)
 return float(np.prod(np.maximum(0,np.minimum(a[1],b[1])-np.maximum(a[0],b[0])))/max(1e-9,min(np.prod(a[1]-a[0]),np.prod(b[1]-b[0]))))
def clusters(proposals):
 # Seed grouping requires substantial shared volume, not merely proximity.
 groups=[]
 for p in proposals:
  matches=[g for g in groups if any(overlap(p["bounds"],q["bounds"])>=.45 for q in g)]
  if not matches:groups.append([p])
  else:
   merged=[p]
   for g in matches:merged.extend(g);groups.remove(g)
   groups.append(merged)
 return groups
def run(source,output):
 source=Path(source);output=Path(output);out=output/"artifacts/completion";out.mkdir(parents=True,exist_ok=True)
 cfg,d,split,_=load(source);rows=rows_from_data(d);asset(source,True)
 proposals=json.loads((output/"analysis/evidence.json").read_text())["proposals"]
 ai={x["id"]:x for x in json.loads((output/"analysis/ai.json").read_text())["items"]}
 current=[p for p in proposals if p.get("evidence_side","current")=="current" and p["source_label"] in STRUCTURES and ai[p["id"]]["decision"]!="deprioritize"]
 old=[p for p in proposals if p.get("evidence_side")=="old" and p["source_label"] in STRUCTURES]
 groups=clusters(current)
 groups.sort(key=lambda g:sum(p.get("novelty_points",0) for p in g),reverse=True)
 v,p,_=read_gaussians(source/"spirula/aligned_scene.ply")
 opacity=1/(1+np.exp(-np.clip(v["opacity"],-50,50)));scale=np.exp(np.column_stack([v[k] for k in ["scale_0","scale_1","scale_2"]])).max(1)
 p=p[(opacity>.10)&(scale<.12)]
 ov,op,_=read_gaussians(cfg["map_ply"])
 records=[];regions=[];used_old=set()
 for index,g in enumerate(groups):
  bounds=np.array([np.min([q["bounds"][0] for q in g],0),np.max([q["bounds"][1] for q in g],0)])
  search=bounds+np.array([[-.35,-.35,0],[.35,.35,.8]])
  m=((p>=search[0])&(p<=search[1])).all(1)&(p[:,2]>bounds[0,2]+.12)
  q=p[m];q,_=voxel_fuse(q,np.zeros_like(q),.025) if len(q) else (q,None)
  grown=bounds.copy();clipped=False;count=0
  if len(q):
   pairs=cKDTree(q).query_pairs(.075,output_type="ndarray")
   graph=coo_matrix((np.ones(len(pairs)),(pairs[:,0],pairs[:,1])),shape=(len(q),len(q)))
   _,labels=connected_components(graph,directed=False)
   seed=((q>=bounds[0])&(q<=bounds[1])).all(1)
   counts=np.bincount(labels[seed],minlength=labels.max()+1)
   keep=counts[labels]>=max(10,int(seed.sum()*.015));q=q[keep];count=len(q)
   if count:
    grown=np.array([np.minimum(bounds[0],q.min(0)-.025),np.maximum(bounds[1],q.max(0)+.025)])
    clipped=bool((q.max(0)>=search[1]-.05).any() or (q.min(0)[:2]<=search[0,:2]+.05).any())
  if np.max(grown[1]-grown[0])>4 or np.prod(grown[1]-grown[0])>12:
   records.append(dict(id=f"assembly_{index:03d}",members=[q["id"] for q in g],state="needs_review",reason="组合范围过大，不能作为单个物体替换",included=False));continue
  id=f"assembly_{index:03d}";reg=dict(id=id,label="设备组件组合",bounds=grown.tolist(),mode="replace")
  try:coverage=preview(source,reg)["coverage"]
  except ValueError as exc:coverage=dict(ready=False,message=str(exc))
  links=[]
  for o in old:
   ob=np.array(o["bounds"])
   distance=float(np.linalg.norm(ob.mean(0)[:2]-grown.mean(0)[:2]))
   if overlap(ob,grown)<.05 and distance>1.2:continue
   # Association is spatial only. Keep separate volumes rather than filling the gap.
   oq=op[((op>=ob[0])&(op<=ob[1])).all(1)]
   if len(oq):oq,_=voxel_fuse(oq,np.zeros_like(oq),.04)
   if len(oq)>8000:oq=oq[::int(np.ceil(len(oq)/8000))]
   if len(oq):
    sup,free,_=evidence(oq,rows,split["train_views"],.10)
    clear=(free>=4)&(sup==0);stable=sup>=3
    fraction=float(clear.mean());stable_fraction=float(stable.mean())
   else:fraction=0.;stable_fraction=0.
   links.append(dict(id=o["id"],bounds=o["bounds"],clear_fraction=fraction,stable_fraction=stable_fraction,
     correspondence="spatial_hypothesis_not_verified_identity",deletion_verified=False))
  included=True
  if included:
   regions.append(reg)
   for link in links:
    # Research preview requires predominant free-space evidence; never reuse this as a publication gate.
    if link["clear_fraction"]>=.6 and link["stable_fraction"]<=.1 and link["id"] not in used_old:
     used_old.add(link["id"]);regions.append(dict(id=link["id"],label="旧位置实验替换",bounds=link["bounds"],mode="replace"))
  records.append(dict(id=id,members=[q["id"] for q in g],original_bounds=bounds.tolist(),bounds=grown.tolist(),
   grown_points=count,search_boundary_reached=clipped,coverage=coverage,old_links=links,included=included,
   state="needs_review",identity_verified=False,complete_object_verified=False,
   reason="已完成三维组件扩展与旧位置证据检查；仍须核验物体身份、边界和更新回归"))
  write_json(out/"report.json",dict(state="running",preview_only=True,groups=records))
  print("COMPLETION_GROUP",index+1,len(groups),count,len(links),flush=True)
 if regions:
  audit=fuse(source,dict(mode="replace",regions=regions),out/"scene_preview.ply",out/"lineage.npz")
  audit.update(preview_only=True,deployable=False,complete_object_verified=False)
  write_json(out/"merge_audit.json",audit)
  ids=list(dict.fromkeys(split["test_views"]+[p["images"][0]["view"] for p in current if p.get("images")]))[:6]
  crops={x["view"]:x["crop"] for x in frame_manifest(source)["frames"]};images={i:[] for i in ids}
  with (ROOT/"runs/gpu.lock").open("a") as lock:
   fcntl.flock(lock,fcntl.LOCK_EX)
   from vps.refinement.gaussian_renderer import GsplatGaussianRenderer
   renderer=GsplatGaussianRenderer("cuda")
   for label,path in [("original",cfg["map_ply"]),("completion preview",out/"scene_preview.ply"),("full trained",source/"spirula/aligned_scene.ply")]:
    for i in ids:
     x0,y0,x1,y1=crops.get(i,next(iter(crops.values())));im=render_map(renderer,path,rows[i])["rgb"][y0:y1,x0:x1].copy()
     cv2.putText(im,label,(6,20),0,.5,(255,80,50),1);images[i].append(im)
   for i in ids:
    x0,y0,x1,y1=crops.get(i,next(iter(crops.values())));im=np.uint8(d["rgb"][i,y0:y1,x0:x1]*255);images[i].append(im)
    cv2.imwrite(str(out/f"view_{i:03d}.jpg"),cv2.cvtColor(np.vstack([np.hstack(images[i][:2]),np.hstack(images[i][2:])]),cv2.COLOR_RGB2BGR))
 else:ids=[]
 report=dict(state="needs_review",preview_only=True,deployable=False,groups=records,regions=regions,views=ids,
  automatic_adoption=False,reused_training=True,old_regions_in_preview=sorted(used_old),
  limitation="Spatial association is not identity; inferred depth and GS share video evidence. Search-boundary and unknown surfaces require verification.")
 write_json(out/"report.json",report)
 page='<meta charset="utf-8"><title>设备补全与旧位置关联</title><style>body{font:18px sans-serif;max-width:1100px;margin:30px auto;line-height:1.7}img{max-width:100%}section{padding:15px;background:#f0f4f5;margin:15px 0}</style><h1>设备补全与旧位置关联</h1><p>自动接续此前被暂缓的设备候选：重叠组件组合、已训练3DGS连通扩展、旧位置多视角腾空检查、生成整范围对比。没有重新训练。</p><p><b>实验预览，未验收、未发布。</b>空间邻近不证明同一物体，连通扩展可能混入邻物。原地图、导航和定位图库不变。该页不是最终候选验收通过。</p><p>左上原图；右上补全实验；左下完整训练结果；右下实拍。</p>'
 for i in ids:page+=f'<h2>视角{i}</h2><img src="view_{i:03d}.jpg">'
 for g in records:page+='<section><b>'+g["id"]+'</b><p>'+html.escape("、".join(g["members"]))+'</p><p>'+html.escape(g["reason"])+'</p><p>旧位置关联提案：'+str(len(g.get("old_links",[])))+'；范围触及搜索边界：'+str(g.get("search_boundary_reached",False))+'</p></section>'
 (out/"index.html").write_text(page)
 return report
if __name__=="__main__":
 parser=argparse.ArgumentParser();parser.add_argument("--source",type=Path,required=True);parser.add_argument("--output",type=Path,required=True);args=parser.parse_args();run(args.source,args.output)
