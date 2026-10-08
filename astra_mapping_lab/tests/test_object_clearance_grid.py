import sys,unittest
from pathlib import Path
import numpy as np
from scipy.ndimage import distance_transform_edt
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from object_clearance_grid import preserve_cores
class ClearanceTests(unittest.TestCase):
 def test_reject_narrowing_reference_corridor(self):
  free=np.zeros((25,25),bool);free[2:23,8:17]=True
  p=np.zeros_like(free);p[12,8:12]=True
  added,rejected=preserve_cores(free,p,[2,3,4])
  self.assertFalse(added.any());self.assertTrue(rejected)
 def test_allow_small_hole_without_reference_center(self):
  free=np.zeros((25,25),bool);free[2:10,2:10]=True;free[17,17:20]=True
  p=np.zeros_like(free);p[17,18]=True
  added,_=preserve_cores(free,p,[2,3,4]);self.assertTrue(added[17,18])
  for r in [2,3,4]:np.testing.assert_array_equal(distance_transform_edt(free)>=r,distance_transform_edt(free&~added)>=r)
if __name__=='__main__':unittest.main()
