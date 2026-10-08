import json,tempfile,unittest
from pathlib import Path
import numpy as np
from astra_scene.io import ROOT,rigid,intrinsic,independent,output_path
from astra_scene.geometry import evidence,grid_cells,grid_centers,update_grid
from astra_scene.postprocess import verify
from astra_scene.web import create_app

def row(depth=2,pose=None):
    return dict(depth=np.full((9,9),depth,dtype=float),valid=np.ones((9,9),bool),clear=np.ones((9,9),bool),
        K=np.array([[4,0,4],[0,4,4],[0,0,1]],float),pose=np.eye(4) if pose is None else pose)
class EvidenceTests(unittest.TestCase):
    def test_occlusion_is_not_deletion(self):
        s,f,o=evidence(np.array([[0,0,3.]]),[row()], [0])
        self.assertEqual((s[0],f[0],o[0]),(0,0,1))
    def test_free_requires_clear_valid_depth(self):
        r=row();r["clear"][:]=False
        self.assertEqual(evidence(np.array([[0,0,1.]]),[r],[0])[1][0],0)
        r["clear"][:]=True
        self.assertEqual(evidence(np.array([[0,0,1.]]),[r],[0])[1][0],1)
    def test_invalid_depth_no_evidence(self):
        r=row(0);r["valid"][:]=False;r["clear"][:]=False
        self.assertEqual(sum(int(x[0]) for x in evidence(np.array([[0,0,1.]]),[r],[0])),0)
    def test_distinct_positions_not_repeated_frames(self):
        self.assertEqual(len(independent([row() for _ in range(10)],range(10),.15)),1)
    def test_behind_camera_no_evidence(self):
        self.assertEqual(sum(int(x[0]) for x in evidence(np.array([[0,0,-1.]]),[row()],[0])),0)
    def test_rotated_grid_roundtrip(self):
        meta=dict(origin=[2,-4,.7],resolution=.05)
        uv=np.array([[0,0],[4,3],[20,10]])
        xy=grid_centers(uv,meta,(30,40))
        actual,ok=grid_cells(np.c_[xy,np.zeros(len(xy))],meta,(30,40))
        np.testing.assert_array_equal(uv,actual);self.assertTrue(ok.all())
    def test_unobserved_grid_is_preserved(self):
        image=np.array([[205,0,254],[0,205,254]],np.uint8)
        cfg=dict(floor_z=0,obstacle_height_m=[.1,1.8],view_baseline_m=.15)
        new,report=update_grid(image,dict(origin=[0,0,0],resolution=.05),np.empty((0,3)),np.empty((0,3)),np.empty((0,3)),[],[],cfg)
        np.testing.assert_array_equal(new,image);self.assertEqual(report["changed_cells"],0)
    def test_removed_obstacle_clears_only_with_full_height_evidence(self):
        image=np.zeros((1,1),np.uint8);meta=dict(origin=[-.05,-.05,0],resolution=.1)
        cfg=dict(floor_z=0,obstacle_height_m=[.1,1.8],view_baseline_m=.15,clear_height_step_m=.1,surface_tolerance_m=.08,min_clear_views=4)
        rows=[]
        for x in (-.3,-.1,.1,.3):
            pose=np.eye(4);pose[:3,3]=[x,0,-2]
            rows.append(row(depth=6,pose=pose))
        point=np.array([[0,0,.5]]);empty=np.empty((0,3))
        cleared,report=update_grid(image,meta,empty,point,empty,rows,list(range(4)),cfg)
        self.assertEqual(int(cleared[0,0]),254);self.assertEqual(report["cleared_columns"],1)
        rows[0]["clear"][:]=False
        kept,_=update_grid(image,meta,empty,point,empty,rows,list(range(4)),cfg)
        self.assertEqual(int(kept[0,0]),0)
    def test_small_obstacle_not_eroded(self):
        image=np.full((10,10),254,np.uint8);cfg=dict(floor_z=0,obstacle_height_m=[.1,1.8],view_baseline_m=.15)
        point=np.array([[.225,.225,.5]])
        new,_=update_grid(image,dict(origin=[0,0,0],resolution=.05),point,np.empty((0,3)),np.empty((0,3)),[],[],cfg)
        self.assertEqual(np.count_nonzero(new==0),1)
    def test_rigid_rejects_scale(self):
        p=np.eye(4);p[0,0]=2
        with self.assertRaises(ValueError):rigid(p)
    def test_rigid_rejects_reflection(self):
        p=np.eye(4);p[0,0]=-1
        with self.assertRaises(ValueError):rigid(p)
    def test_output_escape(self):
        with self.assertRaises(ValueError):output_path(ROOT/"../outside")
    def test_intrinsic_rejects_nan(self):
        k=np.eye(3);k[0,0]=np.nan
        with self.assertRaises(ValueError):intrinsic(k)
    def test_bundle_detects_tampering(self):
        from astra_scene.io import sha256
        with tempfile.TemporaryDirectory(dir=ROOT/"runs") as tmp:
            p=Path(tmp);(p/"a").write_text("before")
            (p/"bundle.json").write_text(json.dumps(dict(files={"a":dict(sha256=sha256(p/"a"))},deployable=False)))
            self.assertEqual(verify(p/"bundle.json")["verified_files"],1)
            (p/"a").write_text("after")
            with self.assertRaises(ValueError):verify(p/"bundle.json")

class WebTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir=ROOT/"runs");self.root=Path(self.tmp.name)
        (self.root/"runs").mkdir();(self.root/"astra_scene").mkdir()
        (self.root/"astra_scene/dashboard.html").write_text("dashboard")
        self.app=create_app(self.root);self.client=self.app.test_client()
    def tearDown(self):self.tmp.cleanup()
    def test_passwordless_access(self):
        self.assertEqual(self.client.get("/api/runs").status_code,200)
        self.assertEqual(self.client.get("/").status_code,200)
        self.assertFalse((self.root/"web_access_token.txt").exists())
    def test_old_login_redirects_home(self):
        self.assertEqual(self.client.post("/login",data={}).status_code,302)
        self.assertEqual(self.client.get("/api/runs").json,[])
    def test_upload_requires_request_marker(self):
        self.client.post("/login",data={})
        self.assertEqual(self.client.post("/api/upload").status_code,403)
    def test_reject_unsupported_upload(self):
        import io
        self.client.post("/login",data={})
        r=self.client.post("/api/upload",headers={"X-Astra-Request":"1"},data={"video":(io.BytesIO(b"evil"),"a.py")})
        self.assertEqual(r.status_code,400)
    def test_result_traversal(self):
        self.client.post("/login",data={})
        self.assertEqual(self.client.get("/results/foo/../../web_access_token.txt").status_code,404)

if __name__=="__main__":unittest.main()
