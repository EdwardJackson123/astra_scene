import unittest,tempfile,json
from pathlib import Path
from unittest.mock import patch
from astra_scene.io import ROOT,write_json
from astra_scene.task_review import analyze
class TaskReviewTests(unittest.TestCase):
 def run_case(self,views,accept=True,label="chair",nav="unknown",vps="potential",scope="whole"):
  with tempfile.TemporaryDirectory(dir=ROOT/"runs") as temp:
   r=Path(temp);(r/"analysis").mkdir()
   write_json(r/"analysis/evidence.json",dict(proposals=[dict(id="p",source_label=label,novelty_points=100,images=[dict(view=1),dict(view=2)])]))
   first=[dict(id="p",visual_change="clear",change_type="add",scope_completeness=scope,navigation_relevance=nav,localization_relevance=vps,reason="stable structural difference",box_advice="same",needs_evidence=[])]
   second=[dict(id="p",accept_candidate=accept,scope_complete=True,supporting_views=views,reason="two views",needs_evidence=[])]
   with patch("astra_scene.task_review._call",side_effect=[(first,{}),(second,{})]):
    analyze(r)
   return json.loads((r/"analysis/ai.json").read_text())["items"][0]
 def test_vps_only_can_nominate_without_measured_regression(self):
  x=self.run_case([1,2]);self.assertEqual(x["decision"],"review_update");self.assertEqual(x["localization_effect"],"unmeasured")
 def test_single_view_or_audit_rejection_cannot_pass(self):
  self.assertEqual(self.run_case([1])["decision"],"defer")
  self.assertEqual(self.run_case([1,2],False)["decision"],"defer")
 def test_robot_never_permanent_candidate(self):self.assertEqual(self.run_case([1,2],label="robot")["decision"],"defer")
 def test_fragment_cannot_be_accepted_as_complete(self):
  self.assertEqual(self.run_case([1,2],scope="fragment")["decision"],"defer")
 def test_unobserved_view_rejected(self):
  self.assertEqual(self.run_case([1,99])["decision"],"defer")
if __name__=="__main__":unittest.main()
