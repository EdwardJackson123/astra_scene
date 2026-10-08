import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from map_update import roi_weights,fresh_weights,old_weights,weighted_opacity,fit_similarity


class MapUpdateTests(unittest.TestCase):
    def test_similarity_recovers_shared_scale_without_pose_deformation(self):
        a=np.random.default_rng(4).normal(size=(60,3));R=np.array([[0,-1,0],[1,0,0],[0,0,1.]])
        b=.91*a@R.T+[4,7,1];s,r,t=fit_similarity(a,b)
        self.assertAlmostEqual(s,.91);np.testing.assert_allclose(r,R,atol=1e-10);np.testing.assert_allclose(t,[4,7,1])
        with self.assertRaises(ValueError):fit_similarity(np.zeros((4,3)),np.zeros((4,3)))
    def test_roi_fades_only_inside_selected_region(self):
        meta=dict(resolution=1.,height=10,origin=[0,0,0])
        w=roi_weights(np.array([[1,5,0],[2,5,0],[2.25,5,0],[4,5,0],[9,5,0]]),meta,[2,2,8,8],.5)
        np.testing.assert_allclose(w,[0,0,.5,1,0])
    def test_unknown_occluded_and_conflicted_old_points_are_preserved(self):
        # First two have no surface/free evidence; the third has contradictory surface.
        w=old_weights([0,0,1,0,5,5],[0,0,8,4,0,0],[False,False,False,False,False,True],np.ones(6))
        np.testing.assert_allclose(w,[0,0,0,1,0,1])
    def test_fresh_requires_multiple_views_and_no_repeated_contradictions(self):
        np.testing.assert_allclose(fresh_weights([2,3,5,6],[0,0,0,2],[1,.6,1,1]),[0,.2,1,0])
    def test_opacity_is_blended_as_probability_not_logit(self):
        x=weighted_opacity(np.array([0.]),np.array([.5]))
        np.testing.assert_allclose(1/(1+np.exp(-x)),[.25])


if __name__=='__main__':unittest.main()
