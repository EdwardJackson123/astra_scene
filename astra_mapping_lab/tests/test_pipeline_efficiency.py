import json, sys, tempfile, unittest, zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from automatic_update import execute, source_identity
from cloud_server import Cloud
from collection import export_spirula
from demo_dataset import create_demo


class EfficiencyTests(unittest.TestCase):
    def test_profile_in_archive_matches_disk(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);m=create_demo(root/'source')
            result=export_spirula(root/'source',root/'package',m['suggested_roi'],[20,20,80,90],pipeline_profile='test_profile')
            with zipfile.ZipFile(root/'package/reconstruction.zip') as z:
                name=next(n for n in z.namelist() if n.endswith('package.json'))
                self.assertEqual(json.loads(z.read(name)),result)
            self.assertEqual(result['pipeline_profile'],'test_profile')

    def test_source_content_change_invalidates_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);create_demo(root);before=source_identity(root)
            m=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
            (root/m['frames'][0]['depth']).write_bytes(b'changed')
            after=source_identity(root)
            self.assertEqual(before['manifest'],after['manifest']);self.assertNotEqual(before['assets'],after['assets'])

    def test_commands_use_snapshot_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);snapshot=root/'snapshot';snapshot.mkdir()
            cfg=dict(output=str(root/'out'),alignment=str(root/'alignment'),source=str(root/'source'),
                     base_map='base.ply',python='python',game_python='game',code_root=str(snapshot))
            ctx=root/'context.json';ctx.write_text(json.dumps(cfg));calls=[]
            execute(ctx,lambda s:None,lambda cmd,**kw:calls.append((cmd,kw)),checkpointing=False)
            for cmd,kwargs in calls:
                self.assertEqual(Path(cmd[1]).parent,snapshot);self.assertEqual(kwargs['cwd'],snapshot)
            cfg['code_snapshot_sha256']='invalid';ctx.write_text(json.dumps(cfg))
            with self.assertRaisesRegex(ValueError,'快照'):
                execute(ctx,lambda s:None,lambda *a,**kw:self.fail('must not execute modified snapshot'))

    def test_data_updates_do_not_split_stage_timing(self):
        with tempfile.TemporaryDirectory() as tmp:
            cloud=Cloud(tmp,'token',runner=lambda key:None);key=cloud.create({'iterations':100})['id']
            with patch('cloud_server.time.time',return_value=100):cloud.update(key,state='PACKAGING')
            n=len(cloud.state(key)['stage_timings'])
            with patch('cloud_server.time.time',return_value=105):cloud.update(key,detail='progress',state='PACKAGING')
            self.assertEqual(len(cloud.state(key)['stage_timings']),n)
            with patch('cloud_server.time.time',return_value=112):cloud.update(key,state='TRAINING')
            self.assertEqual(cloud.state(key)['stage_timings'][-1]['elapsed_s'],12)


if __name__=='__main__':unittest.main()
