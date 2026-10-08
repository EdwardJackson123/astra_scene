"""Portable Spirula runner. Produces a candidate, never modifies a base map."""
import argparse,json,os,subprocess,time
from pathlib import Path


def run(package,binary,device='auto',iterations=6000):
    root=Path(package).resolve();spec=json.loads((root/'package.json').read_text(encoding='utf-8'))
    if not 1<=iterations<=30000:raise ValueError('iterations 超出范围')
    if (root/'training').exists():raise ValueError('训练输出已存在，请保留结果并使用新的任务目录')
    binary=Path(binary).resolve()
    if not binary.is_file():raise ValueError('Spirula 可执行文件不存在')
    cmd=[str(binary),'train','3dgs','--data',str(root/'dataset'),'--data-format','colmap',
         '--output-dir-prefix',str(root),'--output-dir-name','training','--device',device,
         '--num-iterations',str(iterations),'--sh-degree','0','--cap-max','500000',
         '--scene-center','none','--eval-mode','filename','--save-eval-images','1',
         '--steps-per-save',str(iterations),'--disable-viewer','1','--keep-viewer-alive','0','--cache-images','cpu']
    if spec.get('mask_policy')=='ignore_black_observations':cmd+=['--load-masks','1','--apply-loss-for-mask','0']
    if spec.get('pipeline_profile') in ('protected_device_floor_v1','geometry_device_floor_v2','geometry_device_floor_v2_cached','geometry_device_floor_v2_efficient','geometry_device_floor_v2_grid','automatic_geometry_semantics_v3'):
        cmd[cmd.index('--cap-max')+1]='1500000'
        cmd+=['--quality','high','--train-resolution-divisor','1','--validation-fraction','0.1']
    result=dict(state='RUNNING',started_at=time.time(),sh_degree=0,iterations=iterations,synthetic=spec.get('synthetic',False),quality_accepted=False,automatic_application=False)
    def save():
        p=root/'reconstruction_status.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');os.replace(t,p)
    save()
    try:
        with (root/'training.log').open('w',encoding='utf-8') as log:
            from stage_runtime import run as bounded_run
            bounded_run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True,cwd=root,timeout=14400)
        plys=sorted((root/'training').rglob('splat.ply'),key=lambda p:p.stat().st_mtime)
        if not plys:raise ValueError('训练结束但没有找到 splat.ply')
        transform=json.loads((root/'training/scene_transform.json').read_text())['train_from_world']['matrix_4x4']
        if any(abs(transform[i][j]-(1 if i==j else 0))>1e-6 for i in range(4) for j in range(4)):raise ValueError('Spirula 输出坐标发生变化，禁止标记为地图坐标候选')
        metrics=json.loads((root/'training/metrics.json').read_text())
        # No arbitrary crop or old-map removal: result is local reconstruction only.
        result.update(state='CANDIDATE_READY',ply=str(plys[-1].relative_to(root)),finished_at=time.time(),
                      review_required=['多视角外观','深度一致性','坐标对齐','区域接缝'],map_metadata=spec['map_metadata'],
                      coordinate_frame=spec.get('coordinate_frame','source_map'),map_aligned=spec.get('map_aligned',True),metrics=metrics)
    except Exception as e:result.update(state='FAILED',error=str(e),finished_at=time.time());save();raise
    save();return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--package',required=True);p.add_argument('--binary',required=True)
    p.add_argument('--device',default='auto');p.add_argument('--iterations',type=int,default=6000);a=p.parse_args()
    print(json.dumps(run(a.package,a.binary,a.device,a.iterations),ensure_ascii=False))
