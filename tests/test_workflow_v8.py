import unittest,tempfile,json
from pathlib import Path
from unittest.mock import patch
from astra_scene.io import ROOT,write_json
from astra_scene.task_review import analyze,_batched_call,NOMINATE,AUDIT
class WorkflowV8Tests(unittest.TestCase):
 def test_invalid_id_defers_entire_batch(self):
  ps=[dict(id="a",images=[dict(view=1)])]
  with patch("astra_scene.task_review._call",return_value=([dict(id="invented")],{})):
   items,_=_batched_call(NOMINATE,ps,Path("."))
  self.assertTrue(items[0]["ai_validation_failed"])
  self.assertEqual(items[0]["id"],"a")
 def test_all_batches_are_processed(self):
  ps=[dict(id=str(i),images=[]) for i in range(25)]
  def call(prompt,proposals,images):
   return [dict(id=p["id"],accept_candidate=False,scope_complete=False,supporting_views=[],reason="defer") for p in proposals],{}
  with patch("astra_scene.task_review._call",side_effect=call) as mock:
   items,_=_batched_call(AUDIT,ps,Path("."))
  self.assertEqual(len(items),25);self.assertEqual(mock.call_count,5)
 def test_old_scope_never_becomes_replace(self):
  with tempfile.TemporaryDirectory(dir=ROOT/"runs") as temp:
   r=Path(temp);(r/"analysis").mkdir()
   write_json(r/"analysis/evidence.json",dict(proposals=[dict(id="old_a",evidence_side="old",source_label="easel",novelty_points=100,images=[dict(view=1),dict(view=2)])]))
   first=[dict(id="old_a",visual_change="clear",navigation_relevance="potential",localization_relevance="potential",scope_completeness="whole",change_type="remove",missing_parts=[],reason="old object absent")]
   second=[dict(id="old_a",accept_candidate=True,scope_complete=True,supporting_views=[1,2],reason="two views")]
   with patch("astra_scene.task_review._call",side_effect=[(first,{}),(second,{})]):analyze(r)
   item=json.loads((r/"analysis/ai.json").read_text())["items"][0]
   self.assertEqual(item["decision"],"defer");self.assertEqual(item["evidence_side"],"old")
