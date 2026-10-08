from __future__ import annotations
import argparse,json,os,subprocess,time,uuid
from pathlib import Path
from flask import Flask,request,jsonify,send_file,make_response,redirect
from .io import ROOT,output_path
from .pipeline import environment

def create_app(root=ROOT):
    root=Path(root);app=Flask(__name__);app.config["MAX_CONTENT_LENGTH"]=4*1024*1024*1024
    from .intake import blueprint
    app.register_blueprint(blueprint(root))
    @app.before_request
    def request_marker():
        if request.method=="POST" and request.path.startswith("/api/") and request.headers.get("X-Astra-Request")!="1":
            return jsonify(error="缺少请求标记"),403

    @app.route("/login",methods=["GET","POST"])
    def login():
        return redirect("/")

    @app.get("/")
    def home():return send_file(root/"astra_scene/dashboard.html")

    @app.get("/api/settings")
    def settings():
        from .spirula import ply_degree
        cfg=json.loads((root/"configs/xichuang0915.json").read_text())
        degree=ply_degree(cfg["map_ply"])
        return jsonify(base_sh_degree=degree,default_sh_degree="auto",sh_choices=[i for i in [0,1,2] if i>=degree])

    @app.get("/api/runs")
    def runs():
        from .workflow import all_tasks
        return jsonify(all_tasks(root))

    @app.get("/tasks/<name>")
    def task_page(name):
        from .workflow import snapshot
        try:snapshot(root,name)
        except (ValueError,FileNotFoundError):return "任务不存在",404
        return send_file(root/"astra_scene/task.html")

    @app.get("/api/tasks/<name>")
    def task_status(name):
        from .workflow import snapshot,all_tasks
        try:
            data=snapshot(root,name)
            data["children"]=[x for x in all_tasks(root) if x.get("parent")==name]
            return jsonify(data)
        except (ValueError,FileNotFoundError) as e:return jsonify(error=str(e)),404

    @app.post("/api/tasks/<name>/retry")
    def task_retry(name):
        from .workflow import snapshot,read,task_path,enqueue
        try:
            state=snapshot(root,name)
            if not state["can_retry"]:raise ValueError("本任务不能重试")
            meta=read(task_path(root,name));new=meta["kind"]+"_"+time.strftime("%Y%m%d_%H%M%S_")+uuid.uuid4().hex[:6]
            command=[str(root/"runs"/new) if x==str(root/"runs"/name) else x for x in meta["command"]]
            fresh={k:v for k,v in meta.items() if k not in ("pid","state","created","error","command","dedupe")}
            fresh.update(id=new,retry_of=name)
            return jsonify(enqueue(root,fresh,command,meta.get("dedupe","retry:"+name),command[0])),202
        except (ValueError,FileNotFoundError) as e:return jsonify(error=str(e)),400

    @app.post("/api/upload")
    def upload():
        video=request.files.get("video")
        if not video or Path(video.filename or "").suffix.lower() not in (".mp4",".mov",".m4v"):
            return jsonify(error="请上传原始 MP4 或 MOV 视频"),400
        from .workflow import enqueue
        from .io import sha256
        sh_degree=request.form.get("sh_degree","auto")
        if sh_degree not in ("auto","0","1","2"):return jsonify(error="SH 阶数无效"),400
        name=time.strftime("phone_%Y%m%d_%H%M%S_")+uuid.uuid4().hex[:6]
        (root/"uploads").mkdir(exist_ok=True)
        target=root/"uploads"/(name+Path(video.filename).suffix.lower());video.save(target)
        if target.stat().st_size==0:
            target.unlink();return jsonify(error="视频为空"),400
        cfg=root/"configs/xichuang0915.json";python=json.loads(cfg.read_text())["vps_python"]
        command=[python,"-u","-m","astra_scene.pipeline","--video",str(target),"--output",str(root/"runs"/name),
            "--config",str(cfg),"--geometry-only","--spirula","--sh-degree",sh_degree,"--gpu",os.environ.get("ASTRA_GPU","1")]
        meta=dict(id=name,kind="capture",title=Path(video.filename.replace("\\\\","/")).name[:120],
            backend="Spirula SfM + 3DGS / SH "+("跟随原地图" if sh_degree=="auto" else sh_degree),requested_sh_degree=sh_degree)
        result=enqueue(root,meta,command,"video:spirula:sh"+sh_degree+":"+sha256(target)+":"+sha256(cfg),python)
        if result["reused"]:target.unlink()
        return jsonify(result),202



    def scene_run(name):
        import re
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,120}",name):raise ValueError("无效任务")
        p=(root/"runs"/name).resolve()
        if not p.is_relative_to((root/"runs").resolve()) or not (p/"job/ready.json").is_file():raise ValueError("几何尚未准备完成")
        from .workflow import read,task_path
        meta=read(task_path(root,name))
        if meta.get("kind")=="capture" and "--spirula" in meta.get("command",[]) and not (p/"reconstruction.json").is_file():
            raise ValueError("高质量重建尚未完成，请在任务页查看进度")
        return p


    @app.route("/api/regions/<name>/draft",methods=["GET","POST"])
    def editor_draft(name):
        from .workflow import read,save_draft,draft_key
        try:
            source=scene_run(name)
            if request.method=="GET":
                draft=read(source/"editor_draft.json")
                return jsonify({**draft,"geometry_key":draft_key(source),"stale":bool(draft and draft.get("geometry_key")!=draft_key(source))})
            return jsonify(save_draft(source,request.get_json(force=True)))
        except FileExistsError as e:return jsonify(error=str(e)),409
        except (ValueError,TypeError,KeyError) as e:return jsonify(error=str(e)),400

    @app.get("/select3d/<name>")
    def selection_3d_page(name):
        try:scene_run(name)
        except ValueError as e:return str(e),400
        return send_file(root/"astra_scene/selection3d.html")

    @app.get("/web-assets/<name>")
    def web_asset(name):
        allowed={"three.module.js","three.core.js","OrbitControls.js","TransformControls.js"}
        if name=="selection3d.js":return send_file(root/"astra_scene/selection3d.js",mimetype="text/javascript")
        if name not in allowed:return "Not found",404
        return send_file(root/"vendor/web"/name,mimetype="text/javascript")

    @app.get("/api/regions/<name>/cloud")
    def region_cloud(name):
        from .viewer import cloud_manifest
        try:return jsonify(cloud_manifest(scene_run(name)))
        except ValueError as e:return jsonify(error=str(e)),400

    @app.get("/cloud-data/<name>/<filename>")
    def cloud_data(name,filename):
        if filename not in ("base.bin","current.bin"):return "Not found",404
        try:source=scene_run(name)
        except ValueError:return "Not found",404
        path=source/"viewer_assets"/filename
        if not path.is_file():return "Not found",404
        return send_file(path,mimetype="application/octet-stream",conditional=True)

    @app.get("/select/<name>")
    def selection_page(name):
        try:scene_run(name)
        except ValueError as e:return str(e),400
        return send_file(root/"astra_scene/selection.html")


    @app.get("/api/regions/<name>/proposals")
    def region_proposals(name):
        from .proposals import proposals,boxes_path
        try:
            source=scene_run(name)
            if boxes_path(source) is not None:return jsonify(proposals(source))
            pointer=source/"region_detection_pointer.json"
            if pointer.exists():
                output=Path(json.loads(pointer.read_text())["output"])
                status=output/"status.json"
                if status.exists():return jsonify(json.loads(status.read_text()))
                return jsonify(state="running")
            return jsonify(state="not_prepared")
        except ValueError as e:return jsonify(error=str(e)),400

    @app.post("/api/regions/<name>/detect")
    def region_detect(name):
        from .proposals import boxes_path
        try:
            source=scene_run(name)
            if boxes_path(source) is not None:return jsonify(state="ready")
            import fcntl
            with (root/"runs/gpu.lock").open("a") as lock:
                try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:return jsonify(error="另一个候选正在处理，请稍后预框；仍可先手动画框"),409
            output=source/("region_detection_"+uuid.uuid4().hex[:8])
            (source/"region_detection_pointer.json").write_text(json.dumps(dict(output=str(output))))
            python=json.loads((source/"config.json").read_text())["vps_python"]
            with (root/"logs"/(output.name+".log")).open("w") as log:
                subprocess.Popen([python,"-u","-m","astra_scene.proposals","--run",str(source),"--output",str(output),
                    "--gpu",os.environ.get("ASTRA_GPU","1")],cwd=root,env=environment(os.environ.get("ASTRA_GPU","1")),
                    stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            return jsonify(state="running"),202
        except ValueError as e:return jsonify(error=str(e)),400

    @app.get("/api/regions/<name>/frames")
    def region_frames(name):
        from .roi import frame_manifest
        try:return jsonify(frame_manifest(scene_run(name)))
        except ValueError as e:return jsonify(error=str(e)),400

    @app.get("/region-images/<name>/<path:relative>")
    def region_image(name,relative):
        try:base=scene_run(name)
        except ValueError:return "Not found",404
        p=(base/relative).resolve()
        if not p.is_relative_to(base) or p.suffix.lower()!=".jpg" or not p.is_file():return "Not found",404
        return send_file(p)

    @app.post("/api/regions/<name>/preview")
    def region_preview(name):
        from .roi import preview
        try:return jsonify(preview(scene_run(name),request.get_json(force=True)))
        except (ValueError,TypeError,KeyError) as e:return jsonify(error=str(e)),400


    @app.post("/api/regions/<name>/render")
    def start_render(name):
        from .deletion import render_request
        from .io import write_json
        import fcntl
        try:
            source=scene_run(name);key,data=render_request(source,request.get_json(force=True))
            out=source/"render_previews"/key
            with (source/"render_start.lock").open("a") as launch_lock:
                fcntl.flock(launch_lock,fcntl.LOCK_EX)
                if (out/"status.json").exists():
                    state=json.loads((out/"status.json").read_text())
                    if state.get("state") in ("ready","running"):return jsonify(state)
                with (root/"runs/gpu.lock").open("a") as lock:
                    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    except BlockingIOError:return jsonify(error="正在处理其他任务，请稍后预览"),409
                out.mkdir(parents=True,exist_ok=True)
                write_json(out/"request.json",data);write_json(out/"status.json",dict(state="running",id=key))
                python=json.loads((source/"config.json").read_text())["vps_python"]
                with (out/"render.log").open("w") as log:
                    subprocess.Popen([python,"-u","-m","astra_scene.deletion","--run",str(source),"--id",key,
                        "--gpu",os.environ.get("ASTRA_GPU","1")],cwd=root,env=environment(os.environ.get("ASTRA_GPU","1")),
                        stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            return jsonify(state="running",id=key),202
        except (ValueError,TypeError,KeyError,FileNotFoundError) as e:return jsonify(error=str(e)),400

    @app.get("/api/regions/<name>/render/<key>")
    def render_state(name,key):
        from .deletion import safe_id
        try:
            path=scene_run(name)/"render_previews"/safe_id(key)/"status.json"
            return jsonify(json.loads(path.read_text()))
        except (ValueError,FileNotFoundError) as e:return jsonify(error=str(e)),400

    @app.post("/api/regions/<name>/update")
    def region_update(name):
        try:
            source=scene_run(name);preview_id=str((request.get_json() or {}).get("preview_id",""))
            if len(preview_id)!=64 or any(c not in "0123456789abcdef" for c in preview_id):raise ValueError("请先预览选区")
            scope=json.loads((source/"region_previews"/preview_id/"scope.json").read_text())
            if not scope["coverage"]["ready"]:raise ValueError("视角覆盖不足，请补拍")
            render_id=(request.get_json() or {}).get("render_id")
            if scope["mode"]=="delete":
                from .deletion import verify_render_proof
                verify_render_proof(source,preview_id,render_id)
            reconstruction=None
            if scope["mode"]!="delete":
                from .spirula import asset
                from .deletion import verify_candidate_proof
                reconstruction=asset(source)
                if reconstruction:verify_candidate_proof(source,preview_id,render_id,reconstruction["key"])
            from .workflow import enqueue
            child="region_"+time.strftime("%Y%m%d_%H%M%S_")+uuid.uuid4().hex[:6]
            python=json.loads((source/"config.json").read_text())["vps_python"]
            command=[python,"-u","-m","astra_scene.roi","update","--source",str(source),"--preview",preview_id,
                "--output",str(root/"runs"/child),"--gpu",os.environ.get("ASTRA_GPU","1")]+(["--render-proof",render_id] if render_id else [])
            if reconstruction:command+=["--reconstruction-key",reconstruction["key"]]
            labels={"add":"新增","replace":"替换","delete":"删除"};count=len(scope.get("regions",[scope]))
            meta=dict(id=child,kind="candidate",source=name,mode=scope["mode"],region_count=count,
                title=labels[scope["mode"]]+" "+str(count)+" 个区域",
                backend="原模型裁剪" if scope["mode"]=="delete" else "Spirula 重建复用 / 选区合并" if reconstruction else "VGGT + 位姿校正 / GaME 局部训练")
            return jsonify(enqueue(root,meta,command,"region:"+name+":"+preview_id+":"+(reconstruction["key"] if reconstruction else "legacy"),python)),202
        except (ValueError,FileNotFoundError,KeyError) as e:return jsonify(error=str(e)),400

    @app.get("/results/<name>/<path:relative>")
    def result(name,relative):
        base=(root/"runs"/name/"artifacts").resolve()
        if not base.is_relative_to((root/"runs").resolve()):return "Not found",404
        p=(base/relative).resolve()
        if not p.is_relative_to(base) or not p.is_file():return "Not found",404
        response=send_file(p,conditional=True)
        response.headers["X-Content-Type-Options"]="nosniff"
        return response

    @app.errorhandler(413)
    def too_large(_):return jsonify(error="上传请求过大：采集总量最多1GB，地图资料请求最多4GB"),413
    return app

def main():
    p=argparse.ArgumentParser();p.add_argument("--host",default="127.0.0.1");p.add_argument("--port",type=int,default=8065);a=p.parse_args()
    create_app().run(host=a.host,port=a.port,debug=False,threaded=True)
if __name__=="__main__":main()
