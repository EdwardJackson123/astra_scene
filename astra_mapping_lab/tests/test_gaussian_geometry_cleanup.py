import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gaussian_geometry_cleanup import joint_clearance

class GaussianCleanupTests(unittest.TestCase):
    def test_all_probes_need_clearance_and_any_surface_vetoes(self):
        free=np.full((7,4),4);surface=np.zeros((7,4),int)
        free[6,1]=3;surface[4,2]=1;free[:,3]=0
        np.testing.assert_array_equal(joint_clearance(free,surface),[True,False,False,False])
    def test_empty_candidates(self):
        self.assertEqual(len(joint_clearance(np.empty((7,0)),np.empty((7,0)))),0)

if __name__=='__main__':unittest.main()
