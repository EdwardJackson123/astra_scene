"""Independent capture, collection and reconstruction orchestration."""
import json,os,shlex,subprocess,threading,time,uuid,zipfile
from pathlib import Path
from PIL import Image
from collection import collection_route,export_spirula
from recording import atomic,validate_capture_config,require_capture_ready
from replay import load_dataset,fingerprint
from trajectory import raw_manifest,track,align_anchors,reference_anchors


class Workflow:
    def __init__(self,lab):
        self.lab=lab;self.capture=None;self.process=None;self.processing=False
        for sub in ['captures','packages']:(lab.data/sub).mkdir(exist_ok=True)
        # Restart cannot silently revive old capture ownership.
        for p in (lab.data/'captures').glob('*.json'):
            if p.name.endswith('.spec.json'):continue
            s=json.loads(p.read_text(encoding='utf-8'))
            if s.get('state')=='ACTIVE':s.update(state='INTERRUPTED',reason='服务重启；请手动完成已有记录');atomic(p,s)

    def tracking_demo(self):
        import io
        from tracking_demo import create_tracking_demo
        raw_id=uuid.uuid4().hex;ref_id=uuid.uuid4().hex
        raw,ref=create_tracking_demo(self.lab.data/'datasets'/raw_id,self.lab.data/'datasets'/ref_id)
        image=Image.open(self.lab.data/'datasets'/raw_id/'map.png');buf=io.BytesIO();image.save(buf,format='PNG')
        record=self.lab.store_map(buf.getvalue(),raw['map_metadata'],'合成轨迹验证地图',dict(kind='offline_dataset',semantics='合成数据，仅验证完整软件流程'))
        results=[]
        for key,m,status in [(raw_id,raw,'RAW'),(ref_id,ref,'MAP_ALIGNED')]:
            row=dict(id=key,name=m['name'],synthetic=True,map_id=record['id'],map_fingerprint=fingerprint(image,record['metadata']),frames=len(m['frames']),suggested_roi=m['suggested_roi'],pose_status=status)
            atomic(self.lab.data/'datasets'/key/'registered.json',row);results.append(row)
        return dict(raw=results[0],reference=results[1])

    def require_mode(self):
        if self.lab.mode!='PREVIEW':raise ValueError('请先进入实验建图模式')

    def config(self):
        p=self.lab.data/'capture_config.json'
        if p.exists():
            saved=json.loads(p.read_text(encoding='utf-8'))
            if saved.get('pose_source')=='rgbd_odometry':return saved
        return dict(rgb_topic='/camera/camera/color/image_raw',depth_topic='/camera/camera/aligned_depth_to_color/image_raw',
            info_topic='/camera/camera/color/camera_info',pose_source='rgbd_odometry')

    def save_config(self,config):
        if self.process is not None and self.process.poll() is None:raise ValueError('录制期间不能改标定')
        cfg=validate_capture_config(config);atomic(self.lab.data/'capture_config.json',cfg);return cfg

    def capture_state(self):
        if not self.capture:return dict(state='IDLE',frames=0,reason='尚未启动真实采集')
        key=self.capture;meta=json.loads((self.lab.data/'captures'/(key+'.json')).read_text(encoding='utf-8'))
        p=self.lab.data/'datasets'/key/'capture_status.json'
        detail=json.loads(p.read_text(encoding='utf-8')) if p.exists() else dict(state='STARTING',frames=0)
        if self.process is not None and self.process.poll() is not None and detail['state'] not in ('STOPPED','FAILED'):
            detail.update(state='FAILED',reason='采集进程已退出，请检查该任务 capture.log')
        return {**meta,**detail,'id':key,'frame_age_s':max(0,time.time()-detail['last_saved_at']) if detail.get('last_saved_at') else None}

    def preview(self,key):
        if self.lab.mode!='PREVIEW' or not self.capture or key!=self.capture:raise ValueError('当前没有可预览的采集')
        s=self.capture_state()
        if not s.get('frames'):raise ValueError('等待第一帧有效 RGB-D')
        # Only completed, accepted frame files; never opens a camera or starts ROS.
        return (self.lab.data/'datasets'/self.capture/(f"{s['frames']-1:06d}.jpg")).read_bytes()

    def start(self,payload):
        with self.lab.lock:
            self.require_mode()
            if self.process is not None and self.process.poll() is None:raise ValueError('已有采集进行中')
            cfg=require_capture_ready(self.config());task=self.lab.task(payload.get('task_id'))
            record=self.lab.map(task['map_id'])
            if record['source']['kind']!='nav_snapshot':raise ValueError('真实采集必须绑定当前机器人栅格快照')
            # Revalidate same-version image/config as well as metadata before subscription.
            checked=self.lab.plan(dict(map_id=task['map_id'],roi=task['plan']['roi_pixels'],task_type=task['task_type'],margin_m=task['plan']['context_margin_m'],clearance_m=task['plan']['preview_clearance_m']))
            key=uuid.uuid4().hex;output=self.lab.data/'datasets'/key
            spec=dict(output=str(output),map=record,map_image=str(self.lab.data/'maps'/record['id']/'map.png'),roi=checked['plan']['roi_pixels'],calibration=cfg,nav_url=self.lab.nav_url)
            specpath=self.lab.data/'captures'/(key+'.spec.json');atomic(specpath,spec)
            atomic(self.lab.data/'captures'/(key+'.json'),dict(id=key,task_id=task['id'],map_id=record['id'],state='ACTIVE',created_at=time.time()))
            worker=Path(__file__).with_name('ros_capture.py');command='source /opt/ros/humble/setup.bash && exec /usr/bin/python3 '+shlex.quote(str(worker))+' --config '+shlex.quote(str(specpath))
            env=os.environ.copy();env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',RMW_IMPLEMENTATION='rmw_cyclonedds_cpp',ROS_LOCALHOST_ONLY='0')
            with (self.lab.data/'captures'/(key+'.log')).open('ab') as log:
                self.process=subprocess.Popen(['nice','-n','10','bash','-c',command],stdout=log,stderr=log,stdin=subprocess.DEVNULL,env=env)
            self.capture=key;self.lab.event('capture_started',dict(id=key));return self.capture_state()

    def stop(self):
        if self.capture:(self.lab.data/'datasets'/(self.capture+'.stop')).touch()
        return self.capture_state()

    def finish(self,payload):
        key=payload.get('id',self.capture)
        if not isinstance(key,str) or len(key)!=32 or any(c not in '0123456789abcdef' for c in key):raise ValueError('采集标识无效')
        if key==self.capture and self.process is not None and self.process.poll() is None:raise ValueError('请先停止采集并等待进程完成')
        info=json.loads((self.lab.data/'captures'/(key+'.json')).read_text(encoding='utf-8'));root=self.lab.data/'datasets'/key
        manifest=raw_manifest(root)
        with Image.open(root/'map.png') as im:image=im.convert('RGBA')
        meta=manifest['map_metadata']
        registered=dict(id=key,name='原始 RGB-D '+key[:8],synthetic=False,pose_status='RAW',map_id=info['map_id'],frames=len(manifest['frames']),map_fingerprint=fingerprint(image,meta),suggested_roi=manifest['suggested_roi'])
        atomic(root/'registered.json',registered);info.update(state='SAVED');atomic(self.lab.data/'captures'/(key+'.json'),info);return registered

    def process_dataset(self,payload,action):
        with self.lab.lock:
            self.require_mode()
            if self.processing:raise ValueError('已有轨迹或对齐任务进行中')
            source=self.lab.dataset_path(payload.get('dataset_id'));registration=json.loads((source/'registered.json').read_text(encoding='utf-8'))
            session=self.lab.session;key=uuid.uuid4().hex;target=self.lab.data/'datasets'/key;self.processing=True
        cancelled=lambda:self.lab.mode!='PREVIEW' or self.lab.session!=session
        try:
            if action=='track':manifest=track(source,target,cancelled)
            else:
                spec=payload.get('anchors')
                if action=='reference':
                    reference=self.lab.dataset_path(payload.get('reference_id'));spec=reference_anchors(source,reference,cancelled)
                if not isinstance(spec,dict):raise ValueError('需要完整三维锚点 JSON 或参考 RGB-D 数据集')
                manifest=align_anchors(source,target,spec,cancelled)
            with self.lab.lock:
                if cancelled():raise ValueError('模式已退出，处理结果未提交')
                result={**registration,'id':key,'name':manifest['name']+' '+key[:8],'frames':len(manifest['frames']),
                    'pose_status':manifest['pose_status'],'source_dataset':registration['id'],'map_alignment':manifest.get('map_alignment')}
                atomic(target/'registered.json',result);return result
        finally:
            with self.lab.lock:self.processing=False

    def route(self,payload):
        with self.lab.lock:
            self.require_mode();task=self.lab.task(payload.get('task_id'));record=self.lab.map(task['map_id'])
            with Image.open(self.lab.data/'maps'/record['id']/'map.png') as im:route=collection_route(im,record['metadata'],task['plan'],payload.get('start_xy'))
            task['collection_route']=route;atomic(self.lab.data/'sessions'/(task['id']+'.json'),task);return task

    def package(self,payload):
        with self.lab.lock:
            self.require_mode();task=self.lab.task(payload.get('task_id'));root=self.lab.dataset_path(payload.get('dataset_id'));record=self.lab.map(task['map_id'])
            _,image,meta=load_dataset(root,require_map=False)
            with Image.open(self.lab.data/'maps'/record['id']/'map.png') as selected:
                if fingerprint(selected,record['metadata'])!=fingerprint(image,meta):raise ValueError('采集数据与任务地图不一致')
            key=uuid.uuid4().hex;target=self.lab.data/'packages'/key
            spec=export_spirula(root,target,task['plan']['roi_pixels'],task['plan']['context_pixels']);spec.update(id=key,task_id=task['id'],dataset_id=payload['dataset_id'])
            atomic(target/'package.json',spec);return spec

    def package_path(self,key):
        if not isinstance(key,str) or len(key)!=32 or any(c not in '0123456789abcdef' for c in key):raise ValueError('重建任务标识无效')
        path=self.lab.data/'packages'/key
        if not (path/'package.json').exists():raise ValueError('重建任务不存在')
        return path

    def backend_ready(self):return all(os.environ.get('ASTRA_GPU_'+k) for k in ['HOST','USER','PASSWORD','HOSTKEY'])

    def reconstruct(self,payload):
        with self.lab.lock:
            self.require_mode();path=self.package_path(payload.get('id'))
            if not self.backend_ready():raise ValueError('未配置重建服务器连接；可下载数据包在 Spirula 主机运行')
            statepath=path/'reconstruction_status.json'
            if statepath.exists():raise ValueError('此包已有重建任务，请查看状态或重新生成包')
            iterations=payload.get('iterations',6000)
            if type(iterations) is not int or not 100<=iterations<=30000:raise ValueError('训练步数必须为 100–30000')
            atomic(statepath,dict(state='QUEUED',quality_accepted=False));threading.Thread(target=self.remote,args=(path,iterations),daemon=True).start()
            return dict(id=path.name,state='QUEUED')

    def remote(self,path,iterations=6000):
        import paramiko,base64,hashlib
        statepath=path/'reconstruction_status.json';client=paramiko.SSHClient()
        class Pinned(paramiko.MissingHostKeyPolicy):
            def missing_host_key(self,client,hostname,key):
                if base64.b64encode(hashlib.sha256(key.asbytes()).digest()).decode()!=os.environ['ASTRA_GPU_HOSTKEY']:raise ValueError('重建服务器主机密钥不匹配')
        client.set_missing_host_key_policy(Pinned())
        try:
            client.connect(os.environ['ASTRA_GPU_HOST'],port=int(os.environ.get('ASTRA_GPU_PORT','22')),username=os.environ['ASTRA_GPU_USER'],password=os.environ['ASTRA_GPU_PASSWORD'],timeout=10)
            remote='/home/xujie/astra_scene/robot_mapping_jobs/'+path.name;sftp=client.open_sftp()
            for parent in ['/home/xujie/astra_scene/robot_mapping_jobs',remote]:
                try:sftp.mkdir(parent)
                except IOError:
                    if parent==remote:raise ValueError('远端重建目录已存在')
            atomic(statepath,dict(state='UPLOADING',quality_accepted=False))
            for p in path.rglob('*'):
                if not p.is_file() or p.name in ('reconstruction_status.json','reconstruction.zip'):continue
                target=remote+'/'+p.relative_to(path).as_posix();parent=target.rsplit('/',1)[0];parts=parent[len(remote):].strip('/').split('/');current=remote
                for part in parts:
                    if not part:continue
                    current+='/'+part
                    try:sftp.mkdir(current)
                    except IOError:pass
                sftp.put(str(p),target)
            # Share the existing astra GPU mutex and select its established GPU 1.
            script="""import fcntl,subprocess,sys,os
lock=open('/home/xujie/astra_scene/runs/gpu.lock','a')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
env=os.environ.copy();env.update(SS_LANG='en',VK_ICD_FILENAMES='/usr/share/vulkan/icd.d/nvidia_icd.json')
gpu=subprocess.check_output(['nvidia-smi','--id','1','--query-gpu=uuid','--format=csv,noheader'],text=True).strip().replace('GPU-','').replace('-','')
subprocess.run([sys.executable,sys.argv[1]+'/reconstruct.py','--package',sys.argv[1],'--binary','/home/xujie/astra_scene/vendor/spirula_20260920/spirula','--device','uuid:'+gpu,'--iterations',sys.argv[2]],env=env,check=True)
"""
            atomic(statepath,dict(state='RECONSTRUCTING',quality_accepted=False,remote_directory=remote))
            command='/home/xujie/miniconda3/envs/vps/bin/python - '+shlex.quote(remote)+' '+str(iterations)+" <<'PY'\n"+script+'\nPY'
            _,stdout,stderr=client.exec_command(command);stdout.channel.settimeout(7200)
            # Only this job's runner; no global process stops or navigation calls.
            text=stdout.read();errors=stderr.read();code=stdout.channel.recv_exit_status()
            (path/'remote.log').write_bytes(text+b'\n'+errors)
            try:sftp.get(remote+'/training.log',str(path/'training.log'))
            except IOError:pass
            if code:raise ValueError('服务器重建失败或 GPU 正忙，查看任务日志；不会自动重试')
            with sftp.open(remote+'/reconstruction_status.json') as f:result=json.loads(f.read())
            rel=result.get('ply','');parts=Path(rel).parts
            if not rel or rel.startswith('/') or '..' in parts:raise ValueError('远端候选路径无效')
            sftp.get(remote+'/'+rel,str(path/'candidate.ply'))
            evaluations=[]
            for name in sftp.listdir(remote+'/training'):
                import re
                if re.fullmatch(r'eval-(render|gt)-[0-9]{5}\.png',name):
                    sftp.get(remote+'/training/'+name,str(path/name));evaluations.append(name)
            result['evaluation_images']=sorted(evaluations);result['ply']='candidate.ply';result['remote_directory']=remote;atomic(statepath,result)
        except Exception as e:atomic(statepath,dict(state='FAILED',error=str(e),quality_accepted=False))
        finally:client.close()

    def job(self,key):
        path=self.package_path(key);p=path/'reconstruction_status.json'
        return dict(id=key,**(json.loads(p.read_text(encoding='utf-8')) if p.exists() else {'state':'PACKAGE_READY'}))
