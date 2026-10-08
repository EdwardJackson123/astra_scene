"""Robot-side bounded, uncompressed upload and small status requests only."""
import json,os,threading,time,uuid
from pathlib import Path
from urllib.request import Request,urlopen
from urllib.error import HTTPError,URLError
from recording import atomic

class CloudClient:
    def __init__(self,lab):
        self.lab=lab;self.url=os.environ['ASTRA_CLOUD_URL'].rstrip('/');self.token=os.environ['ASTRA_CLOUD_TOKEN'];self.active=False
        self.root=lab.data/'cloud_jobs';self.root.mkdir(exist_ok=True)
        for p in self.root.glob('*.json'):
            s=json.loads(p.read_text())
            if s.get('state')=='UPLOADING':s.update(state='INTERRUPTED',error='上传中断，原记录保留；请重新提交');atomic(p,s)
    def request(self,path,data=None,offset=None):
        headers={'Authorization':'Bearer '+self.token}
        if data is not None and not isinstance(data,bytes):data=json.dumps(data).encode()
        if offset is not None:headers['X-Offset']=str(offset)
        for attempt in range(3 if offset is not None else 1):
            try:
                with urlopen(Request(self.url+path,data=data,headers=headers),timeout=30) as r:return json.loads(r.read(1024*1024))
            except HTTPError:raise
            except (URLError,TimeoutError,ConnectionError):
                if offset is None or attempt==2:raise
                time.sleep(attempt+1)
    def submit(self,payload):
        with self.lab.lock:
            self.lab.workflow.require_mode()
            if self.active:raise ValueError('已有上传进行中')
            demo=payload.get('demo') is True;spec=dict(demo=demo,iterations=payload.get('iterations',6000));groups={}
            if not demo:
                task=self.lab.task(payload.get('task_id'));source=self.lab.dataset_path(payload.get('dataset_id'))
                registration=json.loads((source/'registered.json').read_text())
                if registration['map_id']!=task['map_id']:raise ValueError('记录与任务地图不一致')
                spec.update(roi=task['plan']['roi_pixels'],margin_m=task['plan']['context_margin_m'],map_fingerprint=registration['map_fingerprint']);groups['source']=source
                if payload.get('reference_id'):groups['reference']=self.lab.dataset_path(payload['reference_id'])
            key=uuid.uuid4().hex;path=self.root/(key+'.json');atomic(path,dict(id=key,state='UPLOADING',uploaded_bytes=0));self.active=True
            session=self.lab.session
            threading.Thread(target=self.upload,args=(path,spec,groups,session),daemon=True).start();return json.loads(path.read_text())
    def upload(self,path,spec,groups,session):
        state=json.loads(path.read_text());sent=0;start=time.monotonic()
        try:
            remote=self.request('/jobs',spec);state.update(remote_id=remote['id'],result_url=self.url+'/view/'+remote['id']+'/'+remote['read_key']);atomic(path,state)
            for group,root in groups.items():
                manifest=root/('capture_manifest.json' if (root/'capture_manifest.json').exists() else 'manifest.json')
                m=json.loads(manifest.read_text());names={manifest.name,m['map_image']}
                for f in m['frames']:names.update(f[k] for k in ('rgb','depth') if k in f)
                for name in sorted(names):
                    if Path(name).name!=name:raise ValueError('请使用录制器生成的平面文件布局')
                    offset=0
                    with (root/name).open('rb') as f:
                        while chunk:=f.read(262144):
                            if self.lab.mode!='PREVIEW' or self.lab.session!=session:raise ValueError('已退出模式，上传取消；原始记录保留')
                            self.request('/jobs/'+remote['id']+'/files/'+group+'/'+name,chunk,offset);offset+=len(chunk);sent+=len(chunk)
                            # No compression/hash pass; at most 256 KiB held per transfer.
                            delay=sent/(1024*1024)-(time.monotonic()-start)
                            if delay>0:time.sleep(min(delay,.25))
                            state.update(uploaded_bytes=sent);atomic(path,state)
            if self.lab.mode!='PREVIEW' or self.lab.session!=session:raise ValueError('已退出模式，未提交计算')
            result=self.request('/jobs/'+remote['id']+'/commit',{});state.update(state=result['state']);atomic(path,state)
        except Exception as e:state.update(state='FAILED',error=str(e));atomic(path,state)
        finally:self.active=False
    def jobs(self):
        rows=[]
        for path in self.root.glob('*.json'):
            s=json.loads(path.read_text())
            # Browser polls explicitly. No background result downloads.
            if s.get('remote_id') and s['state'] not in ('UPLOADING','READY','FAILED','INTERRUPTED'):
                try:
                    r=self.request('/jobs/'+s['remote_id']);s.update({k:v for k,v in r.items() if k in ('state','error','map_aligned','coordinate_frame','synthetic','trajectory_quality')});atomic(path,s)
                except Exception:s['connection']='服务器暂不可达；不影响已提交任务'
            rows.append(s)
        return rows
