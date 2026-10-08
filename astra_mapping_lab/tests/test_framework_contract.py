import json,sys,tempfile,unittest,subprocess
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from result_contract import navigation_assessment,status_summary
from grid_pipeline import build
from stage_runtime import run

class FrameworkTests(unittest.TestCase):
    def test_ready_does_not_accept_failed_fusion(self):
        s=status_summary(dict(state='READY',reconstruction_ready=True,map_update=dict(state='FAILED')))
        self.assertEqual(s['reconstruction'],'READY');self.assertEqual(s['fusion'],'FAILED')
        self.assertFalse(s['navigation']['accepted'])

    def test_scale_and_geometry_are_separate(self):
        r=navigation_assessment(dict(outside_allowed_unchanged=True,metric_scale_conflict=True))
        self.assertIn('METRIC_SCALE_CONFLICT_OR_UNKNOWN',r['blockers']);self.assertFalse(r['accepted'])

    def test_timeout_is_bounded(self):
        with self.assertRaises(subprocess.TimeoutExpired):
            run([sys.executable,'-c','import time; time.sleep(10)'],timeout=.1)

    def test_grid_failure_can_retry_without_reusing_partial_outputs(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);context=root/'context.json';context.write_text(json.dumps(dict(output=tmp,grid_backend='rgbd')))
            calls=[]
            def backend(ctx,dest):
                calls.append(dest);dest.mkdir()
                if len(calls)==1:
                    (dest/'partial').write_text('incomplete');raise RuntimeError('injected')
                Image.new('L',(4,4),255).save(dest/'map.png');Image.new('L',(4,4),255).save(dest/'original_map.png')
                Image.new('L',(4,4),255).save(dest/'allowed_scope.png')
                (dest/'map.yaml').write_text('image: map.png\nresolution: 0.05\n')
                (dest/'navigation_update.json').write_text(json.dumps(dict(outside_allowed_unchanged=True)))
                (dest/'hole_repair.json').write_text('{}')
            with patch('navigation_candidate.run',side_effect=backend):
                with self.assertRaisesRegex(RuntimeError,'injected'):build(context)
                self.assertFalse((root/'artifacts/navigation_candidate').exists())
                r=build(context)
                self.assertEqual(build(context),r) # crash after atomic commit is resumable
            self.assertNotEqual(calls[0],calls[1]);self.assertEqual(r['grid_backend'],'rgbd')
            self.assertTrue((root/'artifacts/navigation_candidate/navigation_candidate.zip').exists())

    def test_unknown_backend_fails_before_work(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'context.json';p.write_text(json.dumps(dict(output=tmp,grid_backend='bad')))
            with self.assertRaises(ValueError):build(p)

    def test_stale_fusion_outcome_is_archived_before_new_attempt(self):
        from automatic_update import execute
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);out=root/'out';out.mkdir()
            (out/'automatic_outcome.json').write_text('{"state":"NEEDS_CAPTURE"}')
            context=root/'context.json';context.write_text(json.dumps(dict(output=str(out),alignment=str(root/'align'),source=str(root/'source'),base_map='base',python='python',game_python='python')))
            def fail(cmd,**kwargs):
                if '--build-only' in cmd:raise subprocess.CalledProcessError(1,cmd)
            with self.assertRaises(subprocess.CalledProcessError):execute(context,lambda s:None,runner=fail,phase='late',checkpointing=False)
            self.assertFalse((out/'automatic_outcome.json').exists())
            self.assertEqual(len(list((out/'logs').glob('previous_outcome_*.json'))),1)
