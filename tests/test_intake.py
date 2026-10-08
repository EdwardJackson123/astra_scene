import io,json,unittest,tempfile,zipfile
from pathlib import Path
from unittest.mock import patch,Mock
import cv2,numpy as np
from werkzeug.datastructures import FileStorage
from astra_scene.io import ROOT,write_json
from astra_scene.web import create_app
from astra_scene.maps import extract_references,register
from astra_scene.capture import import_photos
class IntakeTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory(dir=ROOT/"runs");self.root=Path(self.tmp.name)
  (self.root/"configs/secrets").mkdir(parents=True);(self.root/"uploads").mkdir()
  (self.root/"configs/secrets/deepseek.key").write_text("test")
  write_json(self.root/"configs/xichuang0915.json",dict(map_id="xichuang0915",map_ply="fake.ply",vps_python="python"))
  self.c=create_app(self.root).test_client()
  self.h={"X-Astra-Request":"1"}
 def tearDown(self):self.tmp.cleanup()
 def file(self,name,data):return FileStorage(stream=io.BytesIO(data),filename=name)
 def test_registry(self):
  self.assertTrue(self.c.get("/api/maps").json["maps"][0]["ready"])
 def test_missing_ai_fails_before_upload(self):
  (self.root/"configs/secrets/deepseek.key").unlink()
  r=self.c.post("/api/generate",headers=self.h,data=dict(map_id="xichuang0915"))
  self.assertEqual(r.status_code,400);self.assertIn("AI",r.json["error"])
 def test_generate_csrf(self):
  self.assertEqual(self.c.post("/api/generate").status_code,403)
 def test_bad_map_traversal(self):
  self.assertEqual(self.c.post("/api/generate",headers=self.h,data=dict(map_id="../x")).status_code,400)
 def test_photos_count_gate(self):
  with patch("astra_scene.spirula.ply_degree",return_value=0):
   r=self.c.post("/api/generate",headers=self.h,data=dict(input_kind="images",inputs=(io.BytesIO(b"x"),"one.jpg")))
   self.assertEqual(r.status_code,400)
 def test_video_queues_automatic_and_dedup_cleans_upload(self):
  with patch("astra_scene.spirula.ply_degree",return_value=0),patch("astra_scene.intake.enqueue",return_value=dict(id="job",reused=True)) as en:
   r=self.c.post("/api/generate",headers=self.h,data=dict(input_kind="video",inputs=(io.BytesIO(b"test video"),"clip.mp4")))
   self.assertEqual(r.status_code,202);self.assertIn("astra_scene.automatic",en.call_args.args[2])
   self.assertEqual(list((self.root/"uploads").iterdir()),[])
 def test_invalid_suffix_cleans_upload(self):
  with patch("astra_scene.spirula.ply_degree",return_value=0):
   r=self.c.post("/api/generate",headers=self.h,data=dict(input_kind="video",inputs=(io.BytesIO(b"x"),"a.py")))
   self.assertEqual(r.status_code,400);self.assertEqual(list((self.root/"uploads").iterdir()),[])
 def test_zip_slip(self):
  f=self.root/"bad.zip"
  with zipfile.ZipFile(f,"w") as z:z.writestr("../escaped.txt","bad")
  with self.assertRaises(ValueError):extract_references(f,self.root/"refs")
  self.assertFalse((self.root/"escaped.txt").exists())
 def test_import_without_coordinate_confirmation(self):
  with self.assertRaises(ValueError):register(self.root,{},"map",False)
 def test_map_saves_pending_when_refs_missing(self):
  ok,png=cv2.imencode(".png",np.full((10,10),254,np.uint8))
  files=dict(ply=self.file("scene.ply",b"ply"),grid=self.file("map.png",png.tobytes()),yaml=self.file("map.yaml",b"resolution: 0.05\norigin: [0, 0, 0]\n"))
  with patch("astra_scene.maps.ply_degree",return_value=0),patch("astra_scene.postprocess.read_gaussians",return_value=(np.zeros(100),None,0)):
   m=register(self.root,files,"new",True)
  self.assertFalse(m["ready"])
  r=self.c.post("/api/generate",headers=self.h,data=dict(map_id=m["id"]))
  self.assertEqual(r.status_code,400);self.assertIn("补齐",r.json["error"])
 def test_photos_are_direct_capture_and_duplicates_rejected(self):
  images=[]
  for i in range(14):
   path=self.root/f"{i}.png";cv2.imwrite(str(path),np.random.default_rng(i).integers(0,256,(180,240,3),dtype=np.uint8));images.append(str(path))
  manifest=self.root/"photos.json";write_json(manifest,dict(images=images))
  report=import_photos(manifest,self.root/"capture")
  self.assertEqual(report["source"],"phone_rgb_images");self.assertEqual(len(report["frames"]),14)
  self.assertEqual(len(set(f["source_sha256"] for f in report["frames"])),14)
  write_json(manifest,dict(images=[images[0]]*14))
  with self.assertRaises(ValueError):import_photos(manifest,self.root/"duplicate_capture")

 def test_photo_folder_preserves_paths_and_duplicate_basenames(self):
  paths=[f"trip/part{i}/same.jpg" for i in range(14)]
  with patch("astra_scene.spirula.ply_degree",return_value=0),patch("astra_scene.intake.enqueue",return_value=dict(id="job",reused=False)):
   r=self.c.post("/api/generate",headers=self.h,data=dict(input_kind="images",relative_paths=json.dumps(paths),inputs=[(io.BytesIO(str(i).encode()),"same.jpg") for i in range(14)]))
  self.assertEqual(r.status_code,202)
  manifest=json.loads(next((self.root/"uploads").glob("*/photos.json")).read_text())
  self.assertEqual(manifest["original_relative_paths"],paths)
  self.assertEqual(len(set(manifest["images"])),14)
  self.assertEqual([Path(p).read_text() for p in manifest["images"]],[str(i) for i in range(14)])
 def test_bad_photo_folder_manifest_cleans_upload(self):
  for bad in ['["one"]','{}','"oops"','null']:
   with patch("astra_scene.spirula.ply_degree",return_value=0):
    r=self.c.post("/api/generate",headers=self.h,data=dict(input_kind="images",relative_paths=bad,inputs=[(io.BytesIO(b"x"),"same.jpg") for i in range(14)]))
   self.assertEqual(r.status_code,400)
   self.assertEqual(list((self.root/"uploads").iterdir()),[])

if __name__=="__main__":unittest.main()
