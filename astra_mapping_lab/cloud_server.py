"""Server-owned durable queue. RGB-D and candidates never return to the robot."""
from gpu_config import gpu_index
import argparse, hashlib, hmac, html, json, os, queue, re, secrets, threading, time
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit
from recording import atomic

class Cloud:
    def __init__(self, root, token, runner=None):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)
        self.token=token;self.lock=threading.RLock();self.queue=queue.Queue();self.runner=runner or self.pipeline
        for p in self.root.glob('*/state.json'):
            s=json.loads(p.read_text())
            if s['state']=='QUEUED':self.queue.put(p.parent.name)
            elif s['state'] not in ('UPLOADING','READY','FAILED','INTERRUPTED'):
                self.update(p.parent.name,state='INTERRUPTED',error='服务重启，请创建新任务；原始数据已保留')
        threading.Thread(target=self.work,daemon=True).start()
    def directory(self,key):
        if not re.fullmatch('[a-f0-9]{32}',key):raise ValueError('任务标识无效')
        p=self.root/key
        if not (p/'state.json').exists():raise ValueError('任务不存在')
        return p
    def state(self,key):
        from result_contract import status_summary
        result=json.loads((self.directory(key)/'state.json').read_text())
        result['result_status']=status_summary(result)
        return result
    def update(self,key,**changes):
        with self.lock:
            s=self.state(key);now=time.time()
            if 'state' in changes and changes['state']!=s['state']:
                started=s.get('stage_started_at',s.get('created_at',now))
                s.setdefault('stage_timings',[]).append(dict(stage=s['state'],started_at=started,finished_at=now,elapsed_s=max(0,now-started)))
                s['stage_started_at']=now
            s.update(changes,updated_at=now);atomic(self.directory(key)/'state.json',s);return s
    def create(self,spec):
        from grid_pipeline import BACKENDS,DEFAULT_BACKEND
        from instance_policy import DEFAULT_FUSION_POLICY,FUSION_POLICIES
        spec=dict(spec)
        spec.setdefault('grid_backend',DEFAULT_BACKEND)
        spec.setdefault('fusion_policy',DEFAULT_FUSION_POLICY)
        if spec['fusion_policy'] not in FUSION_POLICIES:raise ValueError('unknown fusion policy')
        if spec['grid_backend'] not in BACKENDS:raise ValueError('unknown grid backend')
        if type(spec.get('gpu_index',1)) is not int or spec.get('gpu_index',1)<0:raise ValueError('invalid GPU index')
        iterations=spec.get('iterations',6000)
        if type(iterations)!=int or not 100<=iterations<=30000:raise ValueError('训练步数为 100–30000')
        key=secrets.token_hex(16);p=self.root/key;p.mkdir()
        atomic(p/'request.json',spec);atomic(p/'state.json',dict(id=key,state='UPLOADING',read_key=secrets.token_urlsafe(24),created_at=time.time(),quality_accepted=False))
        return self.state(key)
    def upload(self,key,group,name,offset,data):
        if group not in ('source','reference') or not re.fullmatch(r'[A-Za-z0-9_.-]+',name) or name.startswith('.') or Path(name).suffix not in ('.json','.png','.jpg','.npy'):raise ValueError('文件名无效')
        if len(data)>262144 or offset<0 or offset+len(data)>64*1024*1024:raise ValueError('文件块超限')
        with self.lock:
            p=self.directory(key)
            if self.state(key)['state']!='UPLOADING':raise ValueError('任务已封存')
            folder=p/group;folder.mkdir(exist_ok=True);f=folder/name
            size=f.stat().st_size if f.exists() else 0
            # A response can time out after the chunk has already been written.
            # A byte-identical retry must acknowledge, never append twice.
            if size==offset+len(data) and data:
                with f.open('rb') as old:
                    old.seek(offset)
                    if old.read(len(data))==data:return dict(offset=size)
            if offset!=size:raise ValueError('上传偏移不一致')
            if sum(x.stat().st_size for x in p.glob('*/*') if x.is_file())+len(data)>4*1024**3:raise ValueError('任务超过 4 GB')
            with f.open('ab') as out:out.write(data)
        return dict(offset=offset+len(data))
    def commit(self,key):
        with self.lock:
            if self.state(key)['state']!='UPLOADING':raise ValueError('任务已经提交')
            spec=json.loads((self.directory(key)/'request.json').read_text())
            if not spec.get('demo'):
                from replay import read_dataset
                read_dataset(self.directory(key)/'source')
            result=self.update(key,state='QUEUED');self.queue.put(key);return result
    def work(self):
        while True:
            key=self.queue.get()
            try:self.runner(key)
            except Exception as e:self.update(key,state='FAILED',error=str(e))
            finally:self.queue.task_done()
    def pipeline(self,key):
        from trajectory import track,reference_anchors,align_anchors,require_trajectory_quality
        from replay import read_dataset,replay,fingerprint
        from collection import export_spirula
        from core import make_plan
        from reconstruct import run
        import fcntl,subprocess
        p=self.directory(key);spec=json.loads((p/'request.json').read_text());source=p/'source'
        os.environ['ASTRA_GPU_INDEX']=str(spec.get('gpu_index',1))
        os.environ['ASTRA_CONNECTED_GPU']=os.environ['ASTRA_GPU_INDEX']
        previous=self.state(key)
        if previous.get('reconstruction_ready'):
            result={k:previous[k] for k in ('map_aligned','coordinate_frame','metrics','ply')}
            self.finish_reconstruction(key,Path(previous['effective_source']),result,not bool(spec.get('demo')))
            return
        self.update(key,state='PREPARING')
        if spec.get('demo'):
            from tracking_demo import create_tracking_demo
            create_tracking_demo(source,p/'reference')
        m,image,meta=read_dataset(source)
        if spec.get("map_fingerprint") and spec["map_fingerprint"]!=fingerprint(image,meta):raise ValueError("任务与源数据地图不一致")
        automatic=not bool(spec.get('demo'))
        if automatic:
            from automatic_update import select_map
            select_map(source,json.loads(Path(__file__).with_name('map_update_registry.json').read_text()))
            from automatic_source import prepare
            source=prepare(source,p,lambda stage:self.update(key,state=stage))
            m,image,meta=read_dataset(source)
            from automatic_update import PROFILE
            self.update(key,person_mask=dict(enabled=True),pipeline_profile=PROFILE)
        elif m.get('kind')=='raw_rgbd':
            self.update(key,state='TRACKING');m=track(source,p/'tracked');source=p/'tracked'
        if 'trajectory_report' in m:
            self.update(key,trajectory_quality=m['trajectory_report'].get('quality_gate'))
            require_trajectory_quality(m)
        if not automatic and (p/'reference').exists() and m.get('world_convention')=='camera_local':
            self.update(key,state='ALIGNING');anchors=reference_anchors(source,p/'reference');m=align_anchors(source,p/'aligned',anchors);source=p/'aligned'
        roi=spec.get('roi') or m.get('suggested_roi')
        plan=make_plan(image,meta,roi,spec.get('margin_m',.8),.35)
        if m.get('world_convention')=='xy_ground_z_up':
            self.update(key,state='DETECTING');replay(source,roi,p/'detection')
        self.update(key,state='PACKAGING');package=p/'package'
        package_spec=export_spirula(source,package,roi,plan['context_pixels'],pipeline_profile=PROFILE if automatic else None)
        self.update(key,sfm_rgbd_report=package_spec.get('sfm_rgbd_report'),reconstruction_scope=package_spec['reconstruction_scope'],keyframe_selection=package_spec['keyframe_selection'])
        if automatic:
            from automatic_update import preflight
            if not preflight(self,key,source,package):return
        self.update(key,state='WAITING_GPU')
        with open(os.environ.get('ASTRA_GPU_LOCK','/home/xujie/astra_scene/runs/gpu.lock'),'a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            self.update(key,state='RECONSTRUCTING')
            gpu=subprocess.check_output(['nvidia-smi','--id',str(gpu_index()),'--query-gpu=uuid','--format=csv,noheader'],text=True).strip().replace('GPU-','').replace('-','')
            result=run(package,os.environ.get('ASTRA_SPIRULA','/home/xujie/astra_scene/vendor/spirula_20260920/spirula'),'uuid:'+gpu,spec.get('iterations',6000))
        self.update(key,synthetic=bool(m.get('synthetic')))
        self.finish_reconstruction(key,source,result,automatic)

    def finish_reconstruction(self,key,source,result,automatic=True):
        self.update(key,map_aligned=result['map_aligned'],coordinate_frame=result['coordinate_frame'],metrics=result['metrics'],ply=result['ply'],reconstruction_ready=True,effective_source=str(source))
        if automatic:
            from automatic_update import run as update_map
            try:
                update=update_map(self,key,source,result)
            except Exception as e:
                # A failed registration/semantic stage must not erase a successful reconstruction.
                update={**self.state(key).get('map_update',{}), 'state':'FAILED','error':str(e),'quality_accepted':False,'published':False}
            self.update(key,map_update=update)
        self.update(key,state='READY',quality_accepted=False)

    def retry(self,key):
        with self.lock:
            s=self.state(key)
            if s['state'] not in ('FAILED','INTERRUPTED','READY') or s.get('map_update',{}).get('state') not in ('FAILED',None):raise ValueError('任务未处于可重试的失败状态')
            if not s.get('reconstruction_ready'):raise ValueError('重建尚未完成，请重新提交；已验证的人物遮罩会从内容缓存复用')
            result=self.update(key,state='QUEUED',error=None);self.queue.put(key);return result

def handler(cloud):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass # Capability URLs must not enter access logs.
        def reply(self,code,value):
            body=json.dumps(value,ensure_ascii=False).encode();self.send_response(code);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
        def authorized(self):return hmac.compare_digest(self.headers.get('Authorization',''),'Bearer '+cloud.token)
        def do_POST(self):
            if not self.authorized():return self.reply(403,dict(error='未授权'))
            try:
                size=int(self.headers.get('Content-Length','0'))
                if not 0<=size<=262144:raise ValueError('请求过大')
                body=self.rfile.read(size);parts=urlsplit(self.path).path.strip('/').split('/')
                if parts==['jobs']:result=cloud.create(json.loads(body))
                elif len(parts)==3 and parts[0]=='jobs' and parts[2]=='commit':result=cloud.commit(parts[1])
                elif len(parts)==3 and parts[0]=='jobs' and parts[2]=='retry':result=cloud.retry(parts[1])
                elif len(parts)==5 and parts[0]=='jobs' and parts[2]=='files':result=cloud.upload(parts[1],parts[3],parts[4],int(self.headers.get('X-Offset','0')),body)
                else:return self.reply(404,dict(error='未找到'))
                self.reply(200,result)
            except Exception as e:self.reply(400,dict(error=str(e)))
        def do_GET(self):
            try:
                parts=urlsplit(self.path).path.strip('/').split('/')
                if parts==['health']:
                    from automatic_update import PROFILE
                    from grid_pipeline import DEFAULT_BACKEND
                    from instance_policy import DEFAULT_FUSION_POLICY
                    return self.reply(200,dict(status='ok',pipeline_profile=PROFILE,default_grid_backend=DEFAULT_BACKEND,default_fusion_policy=DEFAULT_FUSION_POLICY,automatic_map_update=True,automatic_publication=False))
                if len(parts)==2 and parts[0]=='jobs':
                    if not self.authorized():return self.reply(403,dict(error='未授权'))
                    return self.reply(200,cloud.state(parts[1]))
                if len(parts)<3 or parts[0]!='view':return self.reply(404,{})
                key,secret=parts[1:3];s=cloud.state(key)
                if not hmac.compare_digest(secret,s['read_key']):return self.reply(403,{})
                p=cloud.directory(key);base='/view/'+key+'/'+secret
                if len(parts)==3:
                    body='<html><meta charset="utf-8"><meta name="referrer" content="no-referrer"><title>Astra 服务器任务</title><body><h1>服务器处理任务</h1><p>状态：'+s['state']+'</p><p>合成样例或真实采集均需质量验收；不会自动更新正式地图。</p>'
                    stages=s['result_status']
                    body+='<h2>结果状态（READY仅代表任务产物可用）</h2><pre>'+html.escape(json.dumps(stages,ensure_ascii=False,indent=2))+'</pre>'
                    if s.get('reconstruction_scope') in ('connected_local_subset','sfm_local_subset'):body+='<p><strong>仅重建可靠连通片段，未覆盖整段记录；不代表完整选区地图。</strong></p>'
                    if s.get('keyframe_selection'):body+='<details><summary>关键帧筛选</summary><pre>'+html.escape(json.dumps(s['keyframe_selection'],ensure_ascii=False,indent=2))+'</pre></details>'
                    if s.get('trajectory_quality'):body+='<h2>重建前轨迹检查</h2><pre>'+html.escape(json.dumps(s['trajectory_quality'],ensure_ascii=False,indent=2))+'</pre>'
                    if s.get('error'):body+='<p>'+html.escape(s['error'])+'</p>'
                    if s.get('person_mask',{}).get('enabled'):body+='<p><strong>已启用人物遮罩：</strong>人物观测不参与 SfM、深度尺度检查、初始化和训练。指标中的静态区域排除了人物及遮罩边缘；全图指标另存。遮挡背景和人物残影仍需人工检查。</p>'
                    if s.get('map_update',{}).get('result_url'):
                        body+='<p><strong>后续原地图更新：</strong><a href="'+html.escape(s['map_update']['result_url'],quote=True)+'">查看融合候选地图和全部留出视角对照</a>。导航栅格、定位图库和正式地图均未替换。</p>'
                    if s.get('map_update'):
                        update=s['map_update'];body+='<p>融合状态：'+html.escape(update.get('state',''))+'。'+html.escape(update.get('error',''))+'</p>'
                        if update.get('candidate_url'):body+='<p><a href="'+html.escape(update['candidate_url'],quote=True)+'">下载更新后的完整候选地图 PLY</a></p>'
                    if s['state']=='READY' and s.get('ply'):
                        body+='<p><a href="'+base+'/candidate">下载独立 Spirula 局部重建 PLY（非融合地图）</a> · <a href="'+base+'/package">重建包</a> · <a href="'+base+'/log">日志</a> · <a href="'+base+'/manifest">重建输入轨迹报告</a></p>'
                        checkpoints=sorted((p/'package/training').glob('step-*.ckpt/splat.ply'))
                        for checkpoint in checkpoints:
                            step=checkpoint.parent.name[5:14]
                            if step in ('000008000','000016000','000024000'):
                                body+='<p><a href="'+base+'/checkpoint-'+step+'">下载 '+str(int(step))+' 步检查点 PLY</a></p>'
                        body+='<details><summary>指标</summary><pre>'+html.escape(json.dumps(s.get('metrics',{}),indent=2))+'</pre></details>'
                        if (p/'comparison/comparison.json').exists():
                            body+='<h2>同配置人物遮罩对照</h2><p>均为 SH0 / 24,000 步：左侧原图，中间未遮罩，右侧人物遮罩。包含全部 12 个留出视角。</p>'
                            for i in range(1,5):
                                if (p/'comparison'/f'comparison_views_{i}.jpg').exists():body+='<p><a href="'+base+f'/comparison_views_{i}.jpg"><img width="100%" src="'+base+f'/comparison_views_{i}.jpg"></a></p>'
                            body+='<p><a href="'+base+'/comparison.json">完整对照指标 JSON</a></p>'
                        for f in sorted((p/'package/training').glob('eval-gt-*.png')):
                            render=f.name.replace('eval-gt','eval-render');body+='<h2>留出视角 '+f.stem[-5:]+'</h2><p>原图 / 重建</p><img width="45%" src="'+base+'/'+f.name+'"><img width="45%" src="'+base+'/'+render+'">'
                    body+='</body></html>';raw=body.encode();self.send_response(200);self.send_header('Content-Type','text/html; charset=utf-8');self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw);return
                name=parts[3]
                if len(parts)!=4 or s['state']!='READY':return self.reply(404,{})
                if name=='candidate':file=p/'package'/s['ply'];mime='application/octet-stream'
                elif re.fullmatch(r'checkpoint-(000008000|000016000|000024000)',name):file=p/'package/training'/('step-'+name[11:]+'.ckpt')/'splat.ply';mime='application/octet-stream'
                elif name=='package':file=p/'package/reconstruction.zip';mime='application/zip'
                elif name=='manifest':
                    folder=Path(s['effective_source']) if s.get('effective_source') else next((p/x for x in ('aligned','tracked','source') if (p/x/'manifest.json').exists()));file=folder/'manifest.json';mime='application/json'
                elif name=='detection':file=p/'detection/report.json';mime='application/json'
                elif name=='log':file=p/'package/training.log';mime='text/plain'
                elif re.fullmatch(r'comparison_views_[1-4]\.jpg',name):file=p/'comparison'/name;mime='image/jpeg'
                elif name=='comparison.json':file=p/'comparison/comparison.json';mime='application/json'
                elif re.fullmatch(r'eval-(gt|render)-[0-9]{5}\.png',name):file=p/'package/training'/name;mime='image/png'
                else:return self.reply(404,{})
                self.send_response(200);self.send_header('Content-Type',mime);self.send_header('Content-Length',str(file.stat().st_size))
                if name in ('candidate','package') or name.startswith('checkpoint-'):self.send_header('Content-Disposition','attachment; filename="'+(name+'.ply' if name.startswith('checkpoint-') else file.name)+'"')
                self.end_headers()
                with file.open('rb') as f:
                    while chunk:=f.read(262144):self.wfile.write(chunk)
            except Exception as e:self.reply(404,dict(error=str(e)))
    return Handler

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--port',type=int,default=8030);args=ap.parse_args()
    token=os.environ['ASTRA_CLOUD_TOKEN'];assert len(token)>=32
    os.environ.setdefault('VK_ICD_FILENAMES','/usr/share/vulkan/icd.d/nvidia_icd.json');os.environ.setdefault('SS_LANG','en')
    ThreadingHTTPServer(('0.0.0.0',args.port),handler(Cloud(args.root,token))).serve_forever()
