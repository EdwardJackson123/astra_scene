from __future__ import annotations
import hashlib, json, os
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

def output_path(path):
    p = Path(path).expanduser().absolute()
    if not p.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError("all_outputs_must_be_inside:" + str(ROOT))
    return p

def write_json(path, value):
    path = output_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temp, path)

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(4*1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()

def rigid(p):
    p = np.asarray(p, np.float64)
    if p.shape != (4,4) or not np.isfinite(p).all():
        raise ValueError("pose_requires_finite_4x4_optical_c2w")
    if not np.allclose(p[3], [0,0,0,1], atol=1e-5) or not np.allclose(p[:3,:3].T@p[:3,:3], np.eye(3), atol=1e-3) or abs(np.linalg.det(p[:3,:3])-1)>1e-3:
        raise ValueError("pose_must_be_SE3_not_scale_or_reflection")
    return p

def intrinsic(k):
    k = np.asarray(k, np.float64)
    if k.shape!=(3,3) or not np.isfinite(k).all() or k[0,0]<=0 or k[1,1]<=0 or not np.allclose(k[2],[0,0,1]) or abs(k[0,1])+abs(k[1,0])>1e-8:
        raise ValueError("requires_pinhole_3x3_K")
    return k


def independent(rows, indices, baseline):
    result=[]
    for i in indices:
        if all(np.linalg.norm(rows[i]["pose"][:3,3]-rows[j]["pose"][:3,3])>=baseline for j in result):
            result.append(i)
    return result
