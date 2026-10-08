import unittest,tempfile,importlib.util,json
from pathlib import Path
import cv2,numpy as np
from astra_scene.io import ROOT
class VideoSamplingTests(unittest.TestCase):
 @unittest.skipUnless((ROOT/"vendor/phone/phone_video.py").is_file(), "Optional external phone backend is not installed")
 def test_duration_driven_sampling_has_no_32_frame_cap(self):
  spec=importlib.util.spec_from_file_location("phone_sampling",ROOT/"vendor/phone/phone_video.py")
  mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
  with tempfile.TemporaryDirectory(dir=ROOT/"runs") as tmp:
   p=Path(tmp);video=p/"test.avi"
   w=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*"MJPG"),10,(160,120))
   self.assertTrue(w.isOpened())
   rng=np.random.default_rng(1)
   for i in range(210):w.write(rng.integers(30,225,(120,160,3),dtype=np.uint8))
   w.release();mod.extract(video,p/"capture",max_frames=0)
   report=json.loads((p/"capture/capture.json").read_text())
   self.assertEqual(len(report["frames"]),42)
   self.assertEqual(len(report["frames"]),report["sampled_usable_frames"])
if __name__=="__main__":unittest.main()
