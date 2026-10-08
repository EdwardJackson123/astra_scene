import json,tempfile,unittest
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch,Mock
from astra_scene.io import ROOT,write_json
from astra_scene.workflow import snapshot,save_draft,draft_key,enqueue,task_path
from astra_scene.web import create_app

class WorkflowTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory(dir=ROOT/"runs");self.root=Path(self.tmp.name)
  (self.root/"runs/capture/job").mkdir(parents=True);(self.root/"logs").mkdir()
  self.source=self.root/"runs/capture"
  (self.source/"job/ready.json").write_text("{}");(self.source/"job/views.npz").write_bytes(b"test geometry")
  write_json(self.source/"status.json",dict(state="geometry_ready",stage="geometry_check"))
  self.app=create_app(self.root);self.client=self.app.test_client()
  self.headers={"X-Astra-Request":"1"}
 def tearDown(self):self.tmp.cleanup()
 def workspace(self):
  return dict(mode="add",activeId="region_1",regions=[dict(id="region_1",center=[0,0,1],size=[1,1,1],quaternion_xyzw=[0,0,0,1],label="cart")])
 def test_draft_roundtrip_and_conflict(self):
  endpoint="/api/regions/capture/draft"
  r=self.client.post(endpoint,json=dict(revision=0,workspace=self.workspace()),headers=self.headers)
  self.assertEqual(r.status_code,200);self.assertEqual(r.json["revision"],1)
  self.assertEqual(self.client.get(endpoint).json["workspace"],self.workspace())
  stale=self.client.post(endpoint,json=dict(revision=0,workspace=self.workspace()),headers=self.headers)
  self.assertEqual(stale.status_code,409);self.assertEqual(self.client.get(endpoint).json["revision"],1)
 def test_draft_geometry_change_detected(self):
  save_draft(self.source,dict(revision=0,workspace=self.workspace()))
  (self.source/"job/views.npz").write_bytes(b"new geometry changed")
  self.assertTrue(self.client.get("/api/regions/capture/draft").json["stale"])
 def test_draft_invalid_rejected_without_overwrite(self):
  w=self.workspace();w["regions"][0]["size"]=[-1,1,1]
  self.assertEqual(self.client.post("/api/regions/capture/draft",json=dict(workspace=w),headers=self.headers).status_code,400)
  self.assertFalse((self.source/"editor_draft.json").exists())
 def test_draft_csrf(self):
  self.assertEqual(self.client.post("/api/regions/capture/draft",json={}).status_code,403)
 def test_candidate_not_presented_as_capture(self):
  run=self.root/"runs/child";(run/"job").mkdir(parents=True);(run/"job/ready.json").write_text("{}")
  write_json(run/"status.json",dict(state="needs_review"))
  write_json(run/"manual_scope.json",dict(source_run="capture",regions=[{}],mode="delete"))
  s=snapshot(self.root,"child");self.assertEqual(s["parent"],"capture");self.assertFalse(s["can_select"])
 def test_task_traversal(self):
  with self.assertRaises(ValueError):snapshot(self.root,"../capture")
 def test_concurrent_duplicate_submission_starts_once(self):
  with patch("astra_scene.workflow.subprocess.Popen",return_value=Mock(pid=None)) as proc,patch("astra_scene.workflow.environment",return_value={}):
   def send(i):return enqueue(self.root,dict(id="task_"+str(i),kind="candidate"),["python","safe"],"same_scope","python")
   with ThreadPoolExecutor(max_workers=4) as pool:result=list(pool.map(send,range(4)))
   self.assertEqual(len({x["id"] for x in result}),1);self.assertEqual(proc.call_count,1)
 def test_worker_loss_is_reported(self):
  write_json(task_path(self.root,"lost"),dict(state="running",kind="candidate",pid=999999999))
  self.assertEqual(snapshot(self.root,"lost")["state"],"failed")
 def test_retry_reuses_source_but_new_output(self):
  path=task_path(self.root,"failed")
  command=["python","-m","astra_scene.pipeline","--output",str(self.root/"runs/failed"),"--video","saved.mp4"]
  write_json(path,dict(id="failed",kind="capture",state="failed",command=command))
  with patch("astra_scene.workflow.enqueue",return_value=dict(id="new",reused=False)) as submit:
   r=self.client.post("/api/tasks/failed/retry",headers=self.headers)
   self.assertEqual(r.status_code,202)
   args=submit.call_args.args;self.assertIn("saved.mp4",args[2]);self.assertNotIn(str(self.root/"runs/failed"),args[2])
 def test_upload_forwards_default_and_explicit_SH(self):
  import io
  (self.root/'configs').mkdir()
  write_json(self.root/'configs/xichuang0915.json',dict(vps_python='python'))
  with patch('astra_scene.workflow.enqueue',return_value=dict(id='new',reused=False)) as submit:
   for choice in ['auto','0','1','2']:
    r=self.client.post('/api/upload',headers=self.headers,data=dict(video=(io.BytesIO(b'mock video'),'video.mp4'),sh_degree=choice))
    self.assertEqual(r.status_code,202)
    cmd=submit.call_args.args[2];self.assertIn('--spirula',cmd)
    self.assertEqual(cmd[cmd.index('--sh-degree')+1],choice)
 def test_invalid_SH_rejected_before_queue(self):
  import io
  r=self.client.post('/api/upload',headers=self.headers,data=dict(video=(io.BytesIO(b'mock video'),'video.mp4'),sh_degree='99'))
  self.assertEqual(r.status_code,400)
 def test_children_grouped(self):
  run=self.root/"runs/child";run.mkdir();write_json(run/"status.json",dict(state="needs_review"))
  write_json(run/"manual_scope.json",dict(source_run="capture",mode="add",regions=[]))
  d=self.client.get("/api/tasks/capture").json
  self.assertEqual([x["id"] for x in d["children"]],["child"])

if __name__=="__main__":unittest.main()
