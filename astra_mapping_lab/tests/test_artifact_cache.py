import sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from artifact_cache import Checkpoints,signature

class CacheTests(unittest.TestCase):
    def test_verified_resume_rejects_changed_input_and_corrupt_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);file=root/'result';calls=[];cache=Checkpoints(root/'checkpoints')
            def build():calls.append(1);file.write_text('valid')
            self.assertFalse(cache.run('stage',{'input':'a'},[file],build)['cached'])
            self.assertTrue(cache.run('stage',{'input':'a'},[file],build)['cached']);self.assertEqual(len(calls),1)
            with self.assertRaises(ValueError):cache.run('stage',{'input':'b'},[file],build)
            file.write_text('corrupt')
            with self.assertRaises(ValueError):cache.run('stage',{'input':'a'},[file],build)

    def test_failed_stage_can_retry_but_missing_outputs_never_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);file=root/'result';cache=Checkpoints(root/'checkpoints')
            with self.assertRaises(ValueError):cache.run('stage',{},[file],lambda:None)
            self.assertFalse(cache.run('stage',{},[file],lambda:file.write_text('ok'))['cached'])
            self.assertNotEqual(signature({'image':'same','keep':True}),signature({'image':'same','keep':False}))

if __name__=='__main__':unittest.main()
