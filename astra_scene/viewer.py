"""Lightweight, read-only scene visualization; derived assets live in this run."""
import fcntl,json,os
import numpy as np
from plyfile import PlyData
from .postprocess import load,rows_from_data
from .geometry import backproject,voxel_fuse
from .io import write_json,sha256

def cloud_manifest(run):
    out=run/"viewer_assets";out.mkdir(exist_ok=True)
    with (out/"build.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        manifest=out/"manifest.json"
        fingerprint=sha256(run/"job/views.npz")
        reconstruction_path=run/"reconstruction.json"
        reconstruction_key=sha256(reconstruction_path) if reconstruction_path.exists() else None
        if manifest.exists():
            data=json.loads(manifest.read_text())
            if data.get("views_sha256")==fingerprint and data.get("reconstruction_key")==reconstruction_key:return data
        cfg,d,split,_=load(run);rows=rows_from_data(d)
        if reconstruction_key:
            from .spirula import asset
            spec=asset(run)
            v=PlyData.read(spec["aligned_ply"],mmap=True)["vertex"].data
            v=v[::max(1,len(v)//250000)]
            good=v["opacity"]>-2;v=v[good]
            p=np.column_stack([v[n] for n in ["x","y","z"]])
            c=np.column_stack([v["f_dc_"+str(i)] for i in range(3)])*.28209479177387814+.5
        else:
            xyz=[];rgb=[]
            for i in split["train_views"]:
                q,color=backproject(rows[i],stride=4);xyz.append(q);rgb.append(color)
            p,c=voxel_fuse(np.concatenate(xyz),np.concatenate(rgb),.025)
        finite=np.isfinite(p).all(axis=1)&np.isfinite(c).all(axis=1);p=p[finite];c=c[finite]
        if not len(p):raise ValueError("没有可显示的有效点云")
        take=np.linspace(0,len(p)-1,min(len(p),150000),dtype=int);p=p[take];c=c[take]
        low=np.percentile(p,1,axis=0);high=np.percentile(p,99,axis=0)
        def save(name,points,colors):
            data=np.column_stack([points,np.clip(colors,0,1)]).astype("<f4")
            temp=out/(name+".tmp");data.tofile(temp);os.replace(temp,out/name)
            return dict(file=name,count=len(data),bytes=int(data.nbytes))
        current=save("current.bin",p,c)
        vertex=PlyData.read(cfg["map_ply"],mmap=True)["vertex"].data
        # Strided sample bounds memory; local context avoids an enormous distant map.
        sample=vertex[::max(1,len(vertex)//400000)]
        q=np.column_stack([sample[n] for n in ("x","y","z")])
        keep=np.isfinite(q).all(axis=1)&((q>=low-2)&(q<=high+2)).all(axis=1)&(sample["opacity"]>-2)
        q=q[keep];colors=np.column_stack([sample["f_dc_"+str(i)][keep] for i in range(3)])*.28209479177387814+.5
        take=np.linspace(0,len(q)-1,min(len(q),100000),dtype=int) if len(q) else np.array([],int)
        base=save("base.bin",q[take],colors[take])
        cameras=[dict(view=int(i),position=d["poses"][i,:3,3].tolist()) for i in split["train_views"]]
        data=dict(schema="astra.viewer.v1",format="little_endian_float32_xyz_rgb",views_sha256=fingerprint,
                  reconstruction_key=reconstruction_key,current_source="spirula_gaussian_centers" if reconstruction_key else "estimated_depth",current=current,base=base,bounds=[low.tolist(),high.tolist()],cameras=cameras,
                  note="彩色：手机估计表面；灰色：原场景抽样。显示点云不代表未经拍摄的表面可靠。")
        write_json(manifest,data);return data
