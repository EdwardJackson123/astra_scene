"""Content-addressed, verified artifacts and explicit stage checkpoints."""
import hashlib,json,os,time
from pathlib import Path
from recording import atomic

_files={}
def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def model_digest(path):
    p=Path(path);s=p.stat();key=(str(p.resolve()),s.st_size,s.st_mtime_ns)
    if key not in _files:_files[key]=digest(p)
    return _files[key]

def signature(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()

class Checkpoints:
    def __init__(self,root):self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)
    def run(self,name,identity,outputs,action):
        record=self.root/(name+'.json');key=signature(identity)
        if record.exists():
            old=json.loads(record.read_text(encoding='utf-8'))
            if old.get('signature')!=key:raise ValueError('阶段输入或代码已变化，请创建新任务版本：'+name)
            if old.get('state')=='complete' and all(Path(p).is_file() and digest(p)==h for p,h in old['outputs'].items()):
                return dict(cached=True,elapsed_s=0)
            if old.get('state')=='complete':raise ValueError('已完成阶段输出被修改或缺失：'+name)
        started=time.time();atomic(record,dict(signature=key,state='running',started_at=started))
        try:
            action()
            outputs=outputs() if callable(outputs) else outputs
            missing=[str(p) for p in outputs if not Path(p).is_file()]
            if missing:raise ValueError('阶段缺少输出：'+repr(missing))
            result=dict(signature=key,state='complete',started_at=started,elapsed_s=time.time()-started,
                        outputs={str(p):digest(p) for p in outputs})
            atomic(record,result);return dict(cached=False,elapsed_s=result['elapsed_s'])
        except Exception as e:
            atomic(record,dict(signature=key,state='failed',error=str(e),started_at=started));raise


def sam_mask(rgb,root,work,device,text,negative,keep_prompted,dilate_ratio,env):
    """Cache independent-frame SAM, preserving the established mask convention."""
    import fcntl,subprocess,numpy as np
    from PIL import Image
    from masks import read_mask
    root=Path(root);work=Path(work);rgb=np.asarray(rgb,dtype=np.uint8)
    binary=root/'vendor/spirula_20260920/spirula';model=root/'cache/spirula/models/sam3-q4_0.ggml'
    params=dict(version=1,image=hashlib.sha256(rgb.tobytes()).hexdigest(),shape=list(rgb.shape),
                binary=model_digest(binary),model=model_digest(model),text=text,negative=negative,
                keep_prompted=keep_prompted,dilate_ratio=float(dilate_ratio))
    key=signature(params);cache=root/'cache/astra_independent_sam_v1'/key[:2]/key;cache.mkdir(parents=True,exist_ok=True)
    started=time.time();hit=False
    with (cache/'lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        target=cache/'mask.png';receipt=cache/'receipt.json'
        if receipt.exists() and target.exists():
            r=json.loads(receipt.read_text(encoding='utf-8'))
            hit=r.get('parameters')==params and r.get('sha256')==digest(target)
        if not hit:
            images=work/'images';images.mkdir(parents=True,exist_ok=True)
            Image.fromarray(rgb).save(images/'frame.png')
            cmd=[str(binary),'sam','track','--model',str(model),'--frames',str(images),'--text',text,
                 '--neg-text',negative,'--dilate-ratio',str(dilate_ratio),'--out',str(work/'masks'),'--device',device]
            if keep_prompted:cmd+=['--keep-prompted']
            with (work/'sam.log').open('w') as log:subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=180)
            mask=read_mask(work/'masks/frame_00000.png',rgb.shape[:2])
            tmp=cache/'mask.tmp.png';Image.fromarray(mask.astype('uint8')*255).save(tmp);tmp.replace(target)
            atomic(receipt,dict(parameters=params,sha256=digest(target)))
        mask=read_mask(target,rgb.shape[:2])
    return mask,dict(key=key,cache_hit=hit,elapsed_s=time.time()-started)
