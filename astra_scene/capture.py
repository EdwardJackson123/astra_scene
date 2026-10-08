"""Validated photo capture input; never invent a video or duplicate frames."""
from pathlib import Path
import json,hashlib
import cv2,numpy as np
from .io import ROOT,write_json,sha256,output_path
def import_photos(manifest,output,max_frames=32):
 manifest=output_path(manifest);output=output_path(output)
 data=json.loads(manifest.read_text());files=[output_path(p) for p in data["images"]]
 if not 14<=len(files)<=200:raise ValueError("请提供14–200张有重叠的照片")
 output.mkdir(parents=True,exist_ok=False);(output/"frames").mkdir()
 accepted=[];rejected=[];seen=set()
 for index,path in enumerate(files):
  digest=sha256(path)
  if digest in seen:rejected.append(dict(frame_index=index,reason="duplicate"));continue
  seen.add(digest);im=cv2.imread(str(path))
  if im is None:raise ValueError("照片无法解码")
  h,w=im.shape[:2]
  if min(h,w)<160:rejected.append(dict(frame_index=index,reason="resolution"));continue
  thumb=cv2.resize(im,(320,max(1,round(320*h/w))));gray=cv2.cvtColor(thumb,cv2.COLOR_BGR2GRAY)
  sharp=float(cv2.Laplacian(gray,cv2.CV_64F).var())
  if sharp<35 or max((gray<5).mean(),(gray>250).mean())>.55:rejected.append(dict(frame_index=index,reason="quality"));continue
  if max(h,w)>1920:im=cv2.resize(im,(round(w*1920/max(h,w)),round(h*1920/max(h,w))))
  target=output/"frames"/f"{index:08d}.jpg"
  if not cv2.imwrite(str(target),im,[cv2.IMWRITE_JPEG_QUALITY,95]):raise ValueError("保存照片失败")
  accepted.append(dict(frame_index=index,time_seconds=float(index),image=str(target),sharpness=sharp,source_sha256=digest))
 if len(accepted)<14:raise ValueError("去重和清晰度筛选后不足14张照片，请补充其他视角")
 ids=np.unique(np.linspace(0,len(accepted)-1,min(max_frames,len(accepted)),dtype=int))
 report=dict(source="phone_rgb_images",source_video=str(manifest),source_sha256=sha256(manifest),
  nominal_fps=1,frames=[accepted[i] for i in ids],all_images=accepted,rejected_quality=rejected,
  sampled_usable_frames=len(accepted),active_map_modified=False,time_is_synthetic_order=True)
 write_json(output/"capture.json",report);return report
