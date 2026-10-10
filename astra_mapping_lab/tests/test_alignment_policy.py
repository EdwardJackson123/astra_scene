import sys,unittest,copy
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from alignment_policy import preview_allowed,usable
class AlignmentPolicyTests(unittest.TestCase):
 def report(self):return dict(accepted=False,scale=.892,rotation=[[1,0,0],[0,1,0],[0,0,1]],translation=[0,0,0],fit=dict(inliers=458),holdout=dict(matches=138))
 def test_failed_quality_can_preview_without_acceptance(self):
  r=self.report();self.assertTrue(preview_allowed(r,'preview'));self.assertFalse(usable(r));r['preview_only']=True;self.assertTrue(usable(r));self.assertFalse(r['accepted'])
 def test_strict_and_missing_policy_refuse(self):
  for policy in ['strict',None,'invalid']:self.assertFalse(preview_allowed(self.report(),policy))
 def test_invalid_or_unsupported_transform_remains_blocked(self):
  for patch in [dict(scale=float('nan')),dict(scale=2),dict(rotation=[[0]*3]*3),dict(translation=[0,0]),dict(fit=dict(inliers=2)),dict(holdout=dict(matches=2))]:
   r=self.report();r.update(patch);r['preview_only']=True;self.assertFalse(usable(r))
 def test_new_cloud_request_records_policy(self):
  import tempfile,json
  from cloud_server import Cloud
  with tempfile.TemporaryDirectory() as tmp:
   c=Cloud(Path(tmp),'x'*40,runner=lambda key:None);k=c.create({})['id'];self.assertEqual(json.loads((c.directory(k)/'request.json').read_text())['alignment_policy'],'preview')
   with self.assertRaises(ValueError):c.create(dict(alignment_policy='bypass_everything'))
if __name__=='__main__':unittest.main()
