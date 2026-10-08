import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import yaml
from astra_scene.io import ROOT
from astra_scene.maps import maps,config_path
class VPSCatalogTests(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory(dir=ROOT/"runs");self.r=Path(self.temp.name)
  (self.r/"configs").mkdir();(self.r/"vps/configs").mkdir(parents=True)
  self.source=self.r/"vps/configs/default.yaml"
  (self.r/"configs/vps_catalog.json").write_text(json.dumps(dict(source=str(self.source))))
  engine=self.r/"configs/engine.yaml";engine.write_text("pose: {method: preserved}\nservice: {maps: []}\n")
  (self.r/"configs/xichuang0915.json").write_text(json.dumps(dict(map_id="xichuang0915",vps_config=str(engine),floor_z=9)))
  a=self.r/"vps/assets";a.mkdir();(a/"train/rgb").mkdir(parents=True);(a/"train/poses").mkdir()
  for f in ["scene.ply","map.png","config.yaml","ref.h5"]:(a/f).write_text("fixture")
  self.asset=dict(ref_data_path="assets/train",ref_descriptors_path="assets/ref.h5",gaussian_ply_path="assets/scene.ply",nav_map_path="assets/map.png",nav_yaml_path="assets/map.yaml")
 def tearDown(self):self.temp.cleanup()
 def save(self,rows):self.source.write_text(yaml.safe_dump(dict(service=dict(maps=rows))))
 def test_variants_and_immutable_local_snapshot(self):
  self.save([dict(id="scene",light_variants=dict(bright=self.asset,dark=self.asset))])
  before=self.source.read_bytes()
  with patch("astra_scene.spirula.ply_degree",return_value=0):
   self.assertEqual([m["id"] for m in maps(self.r)],["scene__bright","scene__dark"])
   p=config_path(self.r,"scene__dark");cfg=json.loads(p.read_text())
   self.assertEqual(cfg["map_ply"],str(self.r/"vps/assets/scene.ply"))
   self.assertEqual(cfg["nav_yaml"],str(self.r/"vps/assets/config.yaml"))
   self.assertIsNone(cfg["floor_z"]);self.assertFalse(cfg["floor_verified"])
   engine=yaml.safe_load(Path(cfg["vps_config"]).read_text())
   self.assertEqual(engine["service"]["maps"][0]["id"],"scene__dark")
   self.assertEqual(engine["pose"]["method"],"preserved")
   self.assertEqual(config_path(self.r,"scene__dark"),p)
  self.assertEqual(self.source.read_bytes(),before)
 def test_missing_assets_not_ready_and_removal_takes_effect(self):
  self.save([dict(id="scene",**self.asset)])
  (self.r/"vps/assets/ref.h5").unlink()
  row=maps(self.r)[0];self.assertFalse(row["ready"]);self.assertIn("参考索引",row["reason"])
  self.save([])
  self.assertEqual(maps(self.r),[])
  with self.assertRaises(ValueError):config_path(self.r,"scene")
 def test_legacy_map_yaml_is_not_fallback(self):
  self.save([dict(id="scene",**self.asset)])
  (self.r/"vps/assets/config.yaml").unlink()
  (self.r/"vps/assets/map.yaml").write_text("legacy")
  row=maps(self.r)[0]
  self.assertFalse(row["ready"]);self.assertIn("config.yaml",row["reason"])
 def test_flat_entry_and_actual_sh(self):
  self.save([dict(id="scene",**self.asset)])
  with patch("astra_scene.spirula.ply_degree",return_value=2):
   row=maps(self.r)[0];self.assertEqual(row["id"],"scene");self.assertEqual(row["sh_degree"],2)
   self.assertTrue(row["ready"])
if __name__=="__main__":unittest.main()
