import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gaussian_grid import supported_voxels
from rgbd_grid_submap import fuse_view_voxels

class GaussianGridTests(unittest.TestCase):
    def test_thin_connected_structure_survives_isolated_point_does_not(self):
        points=np.array([[.01,.01,z] for z in [.01,.06,.11,.16]]+[[1.,1.,1.]])
        keep,report=supported_voxels(points)
        np.testing.assert_array_equal(keep,[True,True,True,True,False])
        self.assertEqual(report['rejected_voxels'],1)

    def test_duplicate_pixels_cannot_inflate_independent_view_count(self):
        a=np.tile([.011,.011,.011],(100,1));b=np.array([[.012,.012,.012]])
        points,votes=fuse_view_voxels([a,b])
        self.assertEqual(len(points),1);self.assertEqual(votes[0],2)

    def test_empty_geometry(self):
        keep,_=supported_voxels(np.empty((0,3)));self.assertEqual(len(keep),0)
        p,v=fuse_view_voxels([]);self.assertEqual(p.shape,(0,3));self.assertEqual(len(v),0)

if __name__=='__main__':unittest.main()
