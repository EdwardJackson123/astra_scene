import json,os,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from instance_policy import enabled,policy


class PolicyTests(unittest.TestCase):
    def test_bounded_replay_never_enables_seed_only_instance_segmentation(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'context.json'
            with patch.dict(os.environ,{'ASTRA_UPDATE_CONTEXT':str(p)}):
                for name,wanted in [('bounded_context_v1',False),('bounded_context_v2',False),('reviewed_instance_v1',True),('',False)]:
                    p.write_text(json.dumps({'fusion_policy':name}))
                    self.assertEqual(policy(),name)
                    self.assertEqual(enabled(),wanted)

    def test_missing_context_preserves_default(self):
        with patch.dict(os.environ,{},clear=True):
            self.assertFalse(enabled())
            self.assertEqual(policy(),'')
