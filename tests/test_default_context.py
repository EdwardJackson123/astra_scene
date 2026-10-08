import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from astra_scene.io import ROOT,write_json
from astra_scene.object_candidate import resolve_backend
from astra_scene.automatic import execute

class DefaultContextTests(unittest.TestCase):
    def test_old_capture_without_setting_uses_new_default(self):
        self.assertEqual(resolve_backend({},dict(base_sh_degree=0,output_sh_degree=0)),('context',None))

    def test_explicit_legacy_rollback(self):
        self.assertEqual(resolve_backend(dict(fusion_backend='scoped'),{}),('scoped',None))

    def test_higher_sh_preserved_by_compatibility_backend(self):
        for degree in (1,2):
            spec=dict(base_sh_degree=0,output_sh_degree=degree)
            backend,note=resolve_backend({},spec)
            self.assertEqual(backend,'scoped');self.assertTrue(note)
            with self.assertRaises(ValueError):resolve_backend({},spec,'context')

    def test_automatic_routes_default_and_keeps_legacy_guard(self):
        for backend in ('context','scoped'):
            with self.subTest(backend=backend),tempfile.TemporaryDirectory(dir=ROOT/'runs') as tmp:
                root=Path(tmp);source=root/'capture';source.mkdir();run=root/'auto'
                cfg=dict(vps_python='python',game_python='python')
                if backend=='scoped':cfg['fusion_backend']=backend
                write_json(source/'config.json',cfg)
                region=dict(id='area',bounds=[[0,0,0],[1,1,1]],mode='replace')
                stages=[]
                def stage(folder,name,*args,**kwargs):
                    stages.append(name)
                    if name=='auto_fusion':write_json(run/'artifacts/fusion/report.json',dict(regions=[]))
                    if name=='object_policy':write_json(run/'artifacts/object_policy/report.json',dict(regions=[region],blocked={}))
                    if name=='fusion_guard':write_json(run/'artifacts/fusion_guard/report.json',dict(regions=[region],deferred=[]))
                def ai(folder):write_json(run/'analysis/ai.json',dict(items=[]))
                with patch('astra_scene.spirula.asset',return_value=dict(base_sh_degree=0,output_sh_degree=0)),patch('astra_scene.automatic.run_stage',side_effect=stage),patch('astra_scene.automatic.ai_stage',side_effect=ai),patch('astra_scene.automatic.render_report'),patch('astra_scene.object_candidate.build_candidate') as build:
                    execute(run,source=source)
                self.assertEqual('fusion_guard' in stages,backend=='scoped')
                self.assertEqual(build.call_args.kwargs['backend'],backend)
                self.assertEqual(json.loads((run/'status.json').read_text())['state'],'needs_review')

    def test_context_progress_follows_internal_candidate(self):
        from astra_scene.progress import details
        with tempfile.TemporaryDirectory(dir=ROOT/'runs') as tmp:
            root=Path(tmp);child=root/'runs/task_candidate'
            write_json(child/'status.json',dict(state='running',stage='context_fusion'))
            write_json(child/'context_fusion/progress.json',dict(step=80,total=320))
            result=details(root,'task',dict(state='running',stage='candidate'),'automatic')
            self.assertIn('80 / 320',result['label'])
