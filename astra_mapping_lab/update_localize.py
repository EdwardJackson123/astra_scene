"""Read-only reference localization for a masked local map; no robot writes."""
from gpu_config import gpu_index
import os,sys,json,subprocess,fcntl,hashlib,time
from pathlib import Path
ROOT=Path('/home/xujie/astra_scene');sys.path.insert(0,str(ROOT))
from astra_scene.pipeline import environment
env=environment(gpu_index());os.environ.update(env)
from update_context import settings
CFG=settings();OUT=Path(CFG['alignment']);OUT.mkdir(parents=True,exist_ok=True)
SOURCE=Path(CFG['source'])
m=json.loads((SOURCE/'manifest.json').read_text())
import numpy as np
from PIL import Image
# Verify the selected robot-map snapshot against the source map pixels.
base=Path(CFG['reference_root'])
assert np.array_equal(np.array(Image.open(SOURCE/'map.png').convert('RGBA')),np.array(Image.open(base/'map.png').convert('RGBA'))),'map_snapshot_mismatch'
training=[f for f in m['frames'] if f['reconstruction_split']=='train']
chosen=[training[i] for i in np.linspace(0,len(training)-1,20,dtype=int)]
capture=dict(source_sha256=hashlib.sha256((SOURCE/'manifest.json').read_bytes()).hexdigest(),frames=[dict(frame_index=int(f['id']),time_seconds=f['timestamp_s'],image=str(SOURCE/f['rgb'])) for f in chosen])
(OUT/'capture.json').write_text(json.dumps(capture,indent=2))
with (ROOT/'runs/gpu.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX)
    cmd=['/home/xujie/miniconda3/envs/vps/bin/python',str(ROOT/'vendor/phone/localize_video_vps.py'),str(OUT/'capture.json'),str(OUT/'anchors'),
         '--root',str(ROOT/'vendor/engine'),'--config',str(ROOT/'vendor/engine/configs/default.yaml'),'--ref-root',str(base/'train'),
         '--descriptors',str(base/'outputs/ref.h5'),'--ref-cache',str(base/'outputs/256.pt'),'--map-id',CFG['map_id']]
    with (OUT/'localize.log').open('w') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,env=env,check=True)
print('LOCALIZATION_COMPLETE',flush=True)
