"""Standalone preview server. nav adapter permits GET metadata/image only."""
import argparse,base64,hashlib,io,json,os,threading,time,uuid
from pathlib import Path
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from urllib.request import urlopen
from urllib.parse import urlsplit,parse_qs
import yaml
from PIL import Image
from core import normalize_metadata,make_plan
from replay import read_dataset,load_dataset,unpack_dataset,replay,fingerprint,ReplayCancelled
from demo_dataset import create_demo
from workflow import Workflow

BASE=Path(__file__).resolve().parent
Image.MAX_IMAGE_PIXELS=4_000_000
MAX_BODY=12*1024*1024


def atomic_json(path,data):
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    os.replace(temp,path)


def decode_image(raw):
    if len(raw)>8*1024*1024:raise ValueError('栅格图像上限为 8 MB')
    with Image.open(io.BytesIO(raw)) as im:
        if im.width*im.height>4_000_000:raise ValueError('栅格图像最多 400 万像素')
        im.load();return im.convert('RGBA')


class Lab:
    def __init__(self,data,nav_url,nav_root=None):
        self.data=Path(data).resolve();self.nav_url=nav_url.rstrip('/');self.lock=threading.RLock()
        self.nav_root=Path(nav_root).resolve() if nav_root else None
        for sub in ['maps','sessions','datasets','replays']:(self.data/sub).mkdir(parents=True,exist_ok=True)
        self.mode='OFF';self.session=None;self.token=uuid.uuid4().hex
        self.replay_running=False
        self.workflow=Workflow(self)
        self.cloud=None
        if os.environ.get("ASTRA_CLOUD_URL"):
            from cloud_client import CloudClient
            self.cloud=CloudClient(self)

    def status(self):
        return dict(mode=self.mode,session_id=self.session,token=self.token,
                    cloud_processing=bool(self.cloud),stage='CAPTURE_AND_RECONSTRUCT',motion_enabled=False,detector_enabled=False,offline_replay_running=self.replay_running,
                    deepseek_enabled=False,spirula_enabled=self.workflow.backend_ready(),nav_access='GET_ONLY',capture=self.workflow.capture_state(),processing=self.workflow.processing)

    def event(self,kind,details):
        with (self.data/'events.jsonl').open('a',encoding='utf-8') as f:
            f.write(json.dumps(dict(time=time.time(),event=kind,details=details),ensure_ascii=False)+'\n')

    def mode_change(self,enabled):
        if type(enabled) is not bool:raise ValueError('enabled 必须是布尔值')
        with self.lock:
            if enabled and self.mode=='OFF':self.session=uuid.uuid4().hex;self.mode='PREVIEW'
            elif not enabled:
                self.mode='OFF';self.session=None;self.workflow.stop()
            self.event('mode',dict(mode=self.mode,session=self.session));return self.status()

    def get_nav(self,kind):
        if kind not in ('metadata','image'):raise ValueError('只允许读取地图')
        with urlopen(self.nav_url+'/viz/api/map/'+kind,timeout=4) as response:
            raw=response.read(8*1024*1024+1)
        if len(raw)>8*1024*1024:raise ValueError('导航地图响应过大')
        return json.loads(raw) if kind=='metadata' else raw

    @staticmethod
    def signature(meta):
        return {k:meta.get(k) for k in ['map_id','map_version','revision','frame_id','image_path','width','height','resolution','origin']}

    def import_current(self):
        a=self.get_nav('metadata');raw=self.get_nav('image');b=self.get_nav('metadata')
        if self.signature(a)!=self.signature(b):raise ValueError('读取过程中地图发生切换，请重试')
        merged,semantics=self.map_semantics(a)
        return self.store_map(raw,merged,str(a.get('map_id','当前地图')),dict(kind='nav_snapshot',
            signature=self.signature(a),semantics=semantics),a)

    def map_semantics(self,metadata):
        if not self.nav_root:
            return metadata,'默认 trinary 阈值，机器人栅格阈值尚待核实'
        image=Path(metadata.get('image_path','')).resolve()
        if self.nav_root not in image.parents:raise ValueError('导航地图路径不在只读 nav 根目录中')
        config=image.parent/'config.yaml'
        if not config.is_file():raise ValueError('导航地图缺少 config.yaml，不能确定栅格语义')
        if self.nav_root not in config.resolve().parents:raise ValueError('地图配置链接超出 nav 目录')
        config_data=yaml.safe_load(config.read_text(encoding='utf-8'))
        if config_data.get('resolution')!=metadata['resolution'] or config_data.get('origin')!=metadata['origin']:
            raise ValueError('导航接口与地图配置不一致，请稍后重新加载')
        if (image.parent/config_data.get('image','')).resolve()!=image:raise ValueError('地图配置引用的图像不一致')
        merged=dict(metadata)
        for key in ['free_thresh','occupied_thresh','negate','mode']:
            if key in config_data:merged[key]=config_data[key]
        return merged,'只读快照；栅格阈值来自该地图的 config.yaml'

    def store_map(self,raw,metadata,name,source,expected=None):
        image=decode_image(raw);meta=normalize_metadata(metadata,image.width,image.height)
        if expected and (expected.get('width')!=image.width or expected.get('height')!=image.height):
            raise ValueError('地图元数据与图像尺寸不一致')
        buf=io.BytesIO();image.save(buf,format='PNG');png=buf.getvalue()
        identity=json.dumps(dict(metadata=meta,source=source),sort_keys=True).encode()
        key=hashlib.sha256(png+identity).hexdigest()
        record=dict(id=key,name=str(name)[:100],metadata=meta,source=source,
                    image_sha256=hashlib.sha256(png).hexdigest(),created_at=time.time())
        with self.lock:
            target=self.data/'maps'/key;target.mkdir(exist_ok=True)
            if not (target/'metadata.json').exists():
                (target/'map.png').write_bytes(png);atomic_json(target/'metadata.json',record)
            return self.map(key)

    def map(self,key):
        if not isinstance(key,str) or len(key)!=64 or any(c not in '0123456789abcdef' for c in key):
            raise ValueError('地图标识无效')
        return json.loads((self.data/'maps'/key/'metadata.json').read_text(encoding='utf-8'))

    def maps(self):
        return sorted([json.loads(p.read_text(encoding='utf-8')) for p in (self.data/'maps').glob('*/metadata.json')],key=lambda m:m['created_at'],reverse=True)

    def task(self,key):
        if not isinstance(key,str) or len(key)!=32 or any(c not in '0123456789abcdef' for c in key):
            raise ValueError('任务标识无效')
        return json.loads((self.data/'sessions'/(key+'.json')).read_text(encoding='utf-8'))

    def tasks(self):
        rows=[]
        for p in (self.data/'sessions').glob('*.json'):
            task=json.loads(p.read_text(encoding='utf-8'))
            rows.append({k:task[k] for k in ['id','created_at','map_id','task_type','status']})
        return sorted(rows,key=lambda r:r['created_at'],reverse=True)

    def dataset_path(self,key):
        if not isinstance(key,str) or len(key)!=32 or any(c not in '0123456789abcdef' for c in key):raise ValueError('数据集标识无效')
        path=self.data/'datasets'/key
        if not (path/'registered.json').is_file():raise ValueError('数据集未注册')
        return path

    def register_dataset(self,raw=None):
        key=uuid.uuid4().hex;path=self.data/'datasets'/key
        if raw is None:create_demo(path)
        else:unpack_dataset(raw,path)
        manifest,image,meta=read_dataset(path);buf=io.BytesIO();image.save(buf,format='PNG')
        record=self.store_map(buf.getvalue(),meta,manifest.get('name','离线数据'),dict(kind='offline_dataset',semantics='离线回放地图；未与当前机器人地图绑定'))
        info=dict(id=key,pose_status='RAW' if manifest.get('kind')=='raw_rgbd' else manifest.get('pose_status','MAP_ALIGNED'),name=manifest.get('name','离线数据'),synthetic=bool(manifest.get('synthetic',False)),map_id=record['id'],map_fingerprint=fingerprint(image,meta),frames=len(manifest['frames']),suggested_roi=manifest.get('suggested_roi'))
        atomic_json(path/'registered.json',info);return info

    def datasets(self):
        return [json.loads(p.read_text(encoding='utf-8')) for p in (self.data/'datasets').glob('*/registered.json')]

    def run_replay(self,payload):
        with self.lock:
            if self.mode!='PREVIEW':raise ValueError('请先进入实验建图模式')
            if self.replay_running:raise ValueError('已有回放正在运行')
            session=self.session;path=self.dataset_path(payload.get('dataset_id'));record=self.map(payload.get('map_id'))
            manifest,image,meta=load_dataset(path)
            with Image.open(self.data/'maps'/record['id']/'map.png') as selected:
                if fingerprint(selected,record['metadata'])!=fingerprint(image,meta):raise ValueError('回放数据与当前选区地图不一致，禁止混用坐标')
            self.replay_running=True
        key=uuid.uuid4().hex;out=self.data/'replays'/key
        try:
            result=replay(path,payload.get('roi'),out,payload.get('parameters'),cancelled=lambda:self.mode!='PREVIEW' or self.session!=session)
            with self.lock:
                if self.mode!='PREVIEW' or self.session!=session:raise ReplayCancelled('建图模式已退出，结果未提交')
                result.update(id=key,dataset_id=payload['dataset_id'],session_id=session,map_id=record['id'])
                atomic_json(out/'report.json',result);atomic_json(out/'published.json',dict(id=key))
                self.event('offline_replay',dict(id=key,synthetic=result['synthetic'],regions=len(result['regions'])))
            return result
        finally:
            with self.lock:self.replay_running=False

    def replay_path(self,key):
        if not isinstance(key,str) or len(key)!=32 or any(c not in '0123456789abcdef' for c in key):raise ValueError('回放标识无效')
        path=self.data/'replays'/key
        if not (path/'published.json').is_file():raise ValueError('回放结果尚未提交')
        return path

    def plan(self,payload):
        with self.lock:
            if self.mode!='PREVIEW':raise ValueError('请先进入实验建图模式')
            session=self.session
            record=self.map(payload.get('map_id'))
            if record['source']['kind']=='nav_snapshot':
                now=self.get_nav('metadata')
                if self.signature(now)!=record['source']['signature']:
                    raise ValueError('导航地图版本已改变，请重新加载地图并框选')
                merged,_=self.map_semantics(now)
                if normalize_metadata(merged,now['width'],now['height'])!=record['metadata']:
                    raise ValueError('栅格配置已改变，请重新加载地图并框选')
                current_image=decode_image(self.get_nav('image'));buf=io.BytesIO();current_image.save(buf,format='PNG')
                after=self.get_nav('metadata')
                if self.signature(now)!=self.signature(after) or hashlib.sha256(buf.getvalue()).hexdigest()!=record['image_sha256']:
                    raise ValueError('栅格内容已改变，请重新加载地图并框选')
            task_type=payload.get('task_type')
            if task_type not in ('inspect','rebuild'):raise ValueError('任务类型无效')
            with Image.open(self.data/'maps'/record['id']/'map.png') as im:
                plan=make_plan(im,record['metadata'],payload.get('roi'),payload.get('margin_m',.8),payload.get('clearance_m',.35))
            task=dict(id=uuid.uuid4().hex,session_id=session,created_at=time.time(),map_id=record['id'],
                      source_map=record['source'],task_type=task_type,status='PREVIEW_ONLY',plan=plan)
            atomic_json(self.data/'sessions'/(task['id']+'.json'),task)
            self.event('plan_saved',dict(task=task['id'],map=record['id']));return task


def handler(lab):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,fmt,*args):
            print(time.strftime('%H:%M:%S'),fmt%args,flush=True)

        def respond(self,code,body,content='application/json; charset=utf-8',download=None):
            if not isinstance(body,bytes):body=json.dumps(body,ensure_ascii=False,allow_nan=False).encode()
            self.send_response(code);self.send_header('Content-Type',content)
            self.send_header('Content-Length',str(len(body)));self.send_header('Cache-Control','no-store')
            if download:self.send_header('Content-Disposition','attachment; filename="'+download+'"')
            self.send_header('X-Content-Type-Options','nosniff')
            self.send_header('Content-Security-Policy',"default-src 'self'; img-src 'self' blob:; script-src 'self'; style-src 'self'; connect-src 'self'; frame-ancestors 'none'")
            self.end_headers();self.wfile.write(body)

        def send_file(self,path,filename,content='application/octet-stream'):
            self.send_response(200);self.send_header('Content-Type',content);self.send_header('Content-Length',str(path.stat().st_size));self.send_header('Content-Disposition','attachment; filename="'+filename+'"');self.send_header('Cache-Control','no-store');self.end_headers()
            with path.open('rb') as f:
                while True:
                    chunk=f.read(1024*1024)
                    if not chunk:break
                    self.wfile.write(chunk)

        def do_GET(self):
            path=urlsplit(self.path).path
            try:
                if path=='/api/cloud/jobs':return self.respond(200,lab.cloud.jobs() if lab.cloud else [])
                if lab.cloud and (path.startswith('/api/packages/') or path.endswith('/download') and path.startswith('/api/datasets/')):return self.respond(400,dict(error='请从服务器结果页下载，不在机器人打包或回传候选'))
                if path=='/api/status':return self.respond(200,lab.status())
                if path=='/api/maps':return self.respond(200,lab.maps())
                if path=='/api/tasks':return self.respond(200,lab.tasks())
                if path=='/api/datasets':return self.respond(200,lab.datasets())
                if path.startswith('/api/datasets/'):
                    parts=path.split('/');directory=lab.dataset_path(parts[3])
                    if len(parts)==5 and parts[4]=='manifest':
                        p=directory/('manifest.json' if (directory/'manifest.json').exists() else 'capture_manifest.json')
                        return self.respond(200,p.read_bytes(),download='dataset-'+parts[3]+'.json')
                    if len(parts)==5 and parts[4]=='download':
                        import zipfile
                        archives=lab.data/'archives';archives.mkdir(exist_ok=True);archive=archives/(parts[3]+'.zip')
                        with lab.lock:
                            if not archive.exists():
                                temp=archive.with_suffix('.tmp')
                                with zipfile.ZipFile(temp,'w',zipfile.ZIP_DEFLATED) as z:
                                    for p in directory.iterdir():
                                        if p.is_file() and p.suffix in ('.json','.png','.jpg','.npy'):z.write(p,p.name)
                                os.replace(temp,archive)
                        return self.send_file(archive,'dataset-'+parts[3]+'.zip','application/zip')
                    return self.respond(404,dict(error='未找到'))
                if path=='/api/capture/config':return self.respond(200,lab.workflow.config())
                if path=='/api/capture/status':return self.respond(200,lab.workflow.capture_state())
                if path=='/api/capture/preview.jpg':
                    key=parse_qs(urlsplit(self.path).query).get('capture_id',[''])[0]
                    return self.respond(200,lab.workflow.preview(key),'image/jpeg')
                if path=='/api/packages':return self.respond(200,[dict(json.loads(p.read_text(encoding='utf-8')),job=lab.workflow.job(p.parent.name)) for p in (lab.data/'packages').glob('*/package.json')])
                if path.startswith('/api/packages/'):
                    parts=path.split('/');directory=lab.workflow.package_path(parts[3])
                    if len(parts)==4:return self.respond(200,lab.workflow.job(parts[3]))
                    if len(parts)==5 and parts[4]=='download':return self.send_file(directory/'reconstruction.zip','reconstruction-'+parts[3]+'.zip','application/zip')
                    if len(parts)==5 and parts[4]=='review':
                        import html
                        job=lab.workflow.job(parts[3]);pictures=job.get('evaluation_images',[])
                        body='<html><head><meta charset="utf-8"><title>重建留出视角</title></head><body><h1>留出视角比较</h1><p>训练结果仅为候选；以下外观比较不代替几何和地图对齐验收。</p><pre>'+html.escape(json.dumps(job.get('metrics',{}),ensure_ascii=False,indent=2))+'</pre>'
                        for name in pictures:body+='<p>'+html.escape(name)+'</p><img width="640" src="/api/packages/'+parts[3]+'/evaluation/'+name+'">'
                        return self.respond(200,(body+'</body></html>').encode(),'text/html; charset=utf-8')
                    if len(parts)==6 and parts[4]=='evaluation':
                        import re
                        if not re.fullmatch(r'eval-(render|gt)-[0-9]{5}\.png',parts[5]):raise ValueError('图像名称无效')
                        return self.respond(200,(directory/parts[5]).read_bytes(),'image/png')
                    if len(parts)==5 and parts[4]=='candidate':return self.send_file(directory/'candidate.ply','candidate-'+parts[3]+'.ply')
                    if len(parts)==5 and parts[4]=='log':
                        p=directory/'training.log'
                        if not p.exists():p=directory/'remote.log'
                        with p.open('rb') as f:f.seek(max(0,p.stat().st_size-16000));data=f.read()
                        return self.respond(200,data,'text/plain; charset=utf-8')
                    return self.respond(404,dict(error='未找到'))
                if path.startswith('/api/replays/'):
                    parts=path.split('/');directory=lab.replay_path(parts[3])
                    if len(parts)==5 and parts[4]=='image':return self.respond(200,(directory/'evidence.png').read_bytes(),'image/png')
                    if len(parts)==5 and parts[4]=='download':return self.respond(200,(directory/'report.json').read_bytes(),download='replay-'+parts[3]+'.json')
                    return self.respond(404,dict(error='未找到'))
                if path.startswith('/api/tasks/'):
                    parts=path.split('/');task=lab.task(parts[3])
                    if len(parts)==5 and parts[4]=='download':
                        return self.respond(200,task,download='astra-task-'+task['id']+'.json')
                    if len(parts)==4:return self.respond(200,task)
                    return self.respond(404,dict(error='未找到'))
                if path.startswith('/api/maps/') and path.endswith('/image'):
                    key=path.split('/')[3];lab.map(key)
                    return self.respond(200,(lab.data/'maps'/key/'map.png').read_bytes(),'image/png')
                if path in ('/','/app.js','/style.css'):
                    name='index.html' if path=='/' else path[1:]
                    mime={'index.html':'text/html; charset=utf-8','app.js':'text/javascript; charset=utf-8','style.css':'text/css; charset=utf-8'}[name]
                    return self.respond(200,(BASE/'static'/name).read_bytes(),mime)
                self.respond(404,dict(error='未找到'))
            except (ValueError,FileNotFoundError) as e:self.respond(404,dict(error=str(e)))

        def do_POST(self):
            try:
                if self.headers.get('X-Lab-Token')!=lab.token:return self.respond(403,dict(error='请刷新页面后重试'))
                if self.headers.get('Content-Type','').split(';')[0]!='application/json':return self.respond(415,dict(error='需要 JSON 请求'))
                size=int(self.headers.get('Content-Length','0'))
                if not 0<size<=MAX_BODY:return self.respond(413,dict(error='请求大小超出限制'))
                payload=json.loads(self.rfile.read(size))
                if not isinstance(payload,dict):raise ValueError('请求必须是对象')
                path=urlsplit(self.path).path
                if lab.cloud and path in ('/api/trajectory','/api/alignment/anchors','/api/alignment/reference','/api/packages','/api/reconstruction/start','/api/replays','/api/datasets/demo','/api/datasets/tracking-demo','/api/datasets/import'):
                    return self.respond(400,dict(error='轻量机器人模式：此计算已迁移到服务器，请使用服务器任务入口'))
                if path=='/api/cloud/submit':
                    if not lab.cloud:raise ValueError('未配置服务器')
                    result=lab.cloud.submit(payload)
                elif path=='/api/mode':result=lab.mode_change(payload.get('enabled'))
                elif path=='/api/maps/current':result=lab.import_current()
                elif path=='/api/maps/import':
                    raw=base64.b64decode(payload.get('image_base64',''),validate=True)
                    metadata=yaml.safe_load(payload.get('metadata_yaml',''))
                    result=lab.store_map(raw,metadata,payload.get('name','导入地图'),dict(kind='uploaded',semantics='来自用户元数据；未与机器人坐标核对'))
                elif path=='/api/plan':result=lab.plan(payload)
                elif path=='/api/datasets/tracking-demo':result=lab.workflow.tracking_demo()
                elif path=='/api/datasets/demo':result=lab.register_dataset()
                elif path=='/api/datasets/import':result=lab.register_dataset(base64.b64decode(payload.get('zip_base64',''),validate=True))
                elif path=='/api/replays':result=lab.run_replay(payload)
                elif path=='/api/capture/config':result=lab.workflow.save_config(payload)
                elif path=='/api/capture/start':result=lab.workflow.start(payload)
                elif path=='/api/capture/stop':result=lab.workflow.stop()
                elif path=='/api/capture/finish':result=lab.workflow.finish(payload)
                elif path=='/api/trajectory':result=lab.workflow.process_dataset(payload,'track')
                elif path=='/api/alignment/anchors':result=lab.workflow.process_dataset(payload,'anchors')
                elif path=='/api/alignment/reference':result=lab.workflow.process_dataset(payload,'reference')
                elif path=='/api/collection/route':result=lab.workflow.route(payload)
                elif path=='/api/packages':result=lab.workflow.package(payload)
                elif path=='/api/reconstruction/start':result=lab.workflow.reconstruct(payload)
                else:return self.respond(404,dict(error='未找到；本服务没有运动控制接口'))
                self.respond(200,result)
            except (ValueError,TypeError,KeyError,yaml.YAMLError) as e:self.respond(400,dict(error=str(e)))
            except Exception as e:
                self.respond(503,dict(error='操作失败：'+str(e)))
    return Handler


def main():
    p=argparse.ArgumentParser();p.add_argument('--host',default='127.0.0.1');p.add_argument('--port',type=int,default=8018)
    p.add_argument('--nav-url',default='http://127.0.0.1:8008');p.add_argument('--nav-root',type=Path)
    p.add_argument('--data',type=Path,default=BASE/'data');args=p.parse_args()
    lab=Lab(args.data,args.nav_url,args.nav_root);server=ThreadingHTTPServer((args.host,args.port),handler(lab))
    lab.event('start',dict(mode='OFF',motion_enabled=False));print(f'Mapping lab listening on {args.host}:{args.port}; preview only',flush=True)
    server.serve_forever()


if __name__=='__main__':main()
