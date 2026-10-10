"""One transactional output contract for measured and Gaussian grid backends."""
import argparse,json,os,uuid,zipfile
from pathlib import Path
from recording import atomic
from artifact_cache import digest,signature
from result_contract import navigation_assessment

BACKENDS=('rgbd','gaussian_surface')
DEFAULT_BACKEND='gaussian_surface'


def build(context,destination=None):
    cfg=json.loads(Path(context).read_text());fusion=Path(cfg['output'])
    backend=cfg.get('grid_backend',DEFAULT_BACKEND)
    if backend not in BACKENDS:raise ValueError('unknown grid backend: '+backend)
    dest=Path(destination) if destination else fusion/'artifacts/navigation_candidate'
    identity=signature(dict(context=digest(context),backend=backend,contract='object_clearance_binary_v1'))
    if dest.exists():
        receipt=dest/'grid_commit.json'
        if receipt.exists():
            committed=json.loads(receipt.read_text())
            if committed.get('identity')==identity and all((dest/name).is_file() and digest(dest/name)==value for name,value in committed['hashes'].items()):
                return json.loads((dest/'navigation_update.json').read_text())
        raise ValueError('grid destination has no matching verified commit; use a new task')
    attempts=fusion/'grid_attempts';attempts.mkdir(exist_ok=True)
    work=attempts/uuid.uuid4().hex;work.mkdir()
    atomic(work/'attempt.json',dict(state='RUNNING',backend=backend))
    try:
        from navigation_candidate import run as measured
        measured(context,work/'measured')
        final=work/'measured'
        if backend=='gaussian_surface':
            from gaussian_surface_grid import run as surface
            from semantic_gaussian_grid import run as semantic
            surface(context,final,work/'surface')
            semantic(context,work/'surface',work/'semantic')
            final=work/'semantic'
        report=json.loads((final/'navigation_update.json').read_text())
        from binary_grid import export
        report=export(final,report)
        if backend=='gaussian_surface':
            from object_clearance_grid import apply
            report=apply(context,final,report)
        registration=json.loads((Path(cfg['alignment'])/'similarity_alignment.json').read_text()) if cfg.get('alignment') else {}
        report['alignment_accepted']=registration.get('accepted')
        report['preview_only']=registration.get('preview_only',False)
        report.update(schema_version=2,grid_backend=backend,grid_pipeline_version='object_clearance_binary_v1',
            candidate_map_sha256=digest(final/'map.png'),navigation_assessment=navigation_assessment(report))
        # Keep the cleanup audit bound to the committed candidate.
        if backend=='gaussian_surface':
            report['contour_cleanup_version']=report['contour_cleanup']['version']
        atomic(final/'navigation_update.json',report)
        # Common black/white preview and download names for both backends.
        from PIL import Image
        old=Image.open(final/'original_map.png').convert('RGB');new=Image.open(final/'map.png').convert('RGB')
        preview=Image.new('RGB',(old.width*2,old.height),'white');preview.paste(old,(0,0));preview.paste(new,(old.width,0));preview.save(final/'grid_comparison.png')
        (final/'index.html').write_text('<meta charset="utf-8"><h1>栅格候选（未发布）</h1><p>后端：'+backend+'</p><p>左：原图；右：候选。仅两类：黑色不可通行，白色可通行；不确定格归为不可通行。导航验收尚未通过。</p><img style="max-width:100%" src="grid_comparison.png"><p><a href="navigation_update.json">验收与增删审计</a> · <a href="hole_repair.json">补洞审计</a> · <a href="navigation_candidate.zip">下载候选</a></p>',encoding='utf-8')
        if backend=='gaussian_surface':
            with (final/'index.html').open('a',encoding='utf-8') as page:
                page.write('<p><a href="contour_cleanup.json">轮廓整理审计</a> · <a href="object_clearance.json">设备连接与参考净空审计</a> · <a href="model_occupancy_preview_NOT_NAVIGATION.png">纯形状预览（不可用于导航）</a></p>')
        with zipfile.ZipFile(final/'navigation_candidate.zip','w',zipfile.ZIP_DEFLATED) as z:
            for name in ('map.png','map.yaml','config.yaml','navigation_update.json','hole_repair.json','binary_export.json'):z.write(final/name,name)
            if backend=='gaussian_surface':
                for name in ('contour_cleanup.json','object_clearance.json'):z.write(final/name,name)
        required=('binary_export.json','map.png','map.yaml','config.yaml','navigation_update.json','hole_repair.json','navigation_candidate.zip','index.html','grid_comparison.png')
        if backend=='gaussian_surface':required+=('object_clearance.json','object_clearance_evidence.npz','contour_cleanup.json','contour_cleanup_evidence.npz','model_occupancy_preview_NOT_NAVIGATION.png')
        atomic(final/'grid_commit.json',dict(identity=identity,hashes={name:digest(final/name) for name in required}))
        dest.parent.mkdir(parents=True,exist_ok=True)
        os.rename(final,dest)
        atomic(work/'attempt.json',dict(state='COMPLETE',backend=backend,output=str(dest)))
        return report
    except Exception as e:
        atomic(work/'attempt.json',dict(state='FAILED',backend=backend,error=str(e)))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--context',type=Path,required=True);p.add_argument('--destination',type=Path)
    a=p.parse_args();build(a.context,a.destination)
