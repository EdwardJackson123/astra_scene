import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from protected_takeover import in_volumes


class DeviceVolumesTests(unittest.TestCase):
    def test_device_footprint_does_not_extrude_through_ceiling_or_background(self):
        regions=[dict(bounds=[[1,1,.1],[2,2,1.8]])]
        p=np.array([[1.5,1.5,1],[1.5,1.5,2.7],[1.5,1.5,-.1],[1.5,3,1]])
        np.testing.assert_array_equal(in_volumes(p,regions),[True,False,False,False])

    def test_separate_old_and_new_positions_do_not_fill_the_gap(self):
        regions=[dict(bounds=[[0,0,.1],[1,1,1.8]]),dict(bounds=[[3,0,.1],[4,1,1.8]])]
        p=np.array([[.5,.5,1],[2,.5,1],[3.5,.5,1]])
        np.testing.assert_array_equal(in_volumes(p,regions),[True,False,True])


if __name__=='__main__':unittest.main()
