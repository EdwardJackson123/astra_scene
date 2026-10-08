import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from regional_takeover import owner,sample_field


class RegionalOwnershipTests(unittest.TestCase):
    def test_disconnected_regions_leave_the_gap_old_owned(self):
        field=np.array([[1,0,0,1]],bool)
        points=np.array([[.5,.5,1],[1.5,.5,1],[2.5,.5,1],[3.5,.5,1]])
        fresh=owner(points,field,[0,0],1,[-.2,3])
        np.testing.assert_array_equal(fresh,[True,False,False,True])
        retained_old=~owner(points,field,[0,0],1,[-.2,3])
        self.assertFalse((fresh&retained_old).any())
        self.assertTrue((fresh|retained_old).all())

    def test_unknown_columns_and_outside_height_stay_unchanged(self):
        field=np.array([[1,0]],bool)
        p=np.array([[.5,.5,-.3],[.5,.5,3.1],[1.5,.5,1],[-.01,.5,1],[2,.5,1],[.5,.5,1]])
        np.testing.assert_array_equal(owner(p,field,[0,0],1,[-.2,3]),[False,False,False,False,False,True])

    def test_nonzero_origin_and_metric_resolution(self):
        p=np.array([[2.01,-.99,1],[2.06,-.99,1],[2.11,-.99,1]])
        np.testing.assert_array_equal(sample_field(p,np.array([[4,9]]),[2,-1],.05),[4,9,0])


if __name__=='__main__':unittest.main()
