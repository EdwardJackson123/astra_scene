import unittest
import numpy as np
from astra_scene.roi import validate_bounds,contains,make_boxes
class ThreeDScopeTests(unittest.TestCase):
    def test_preserves_user_bounds_exactly(self):
        bounds=[[7.2,3.1,.05],[8.1,4.2,1.3]]
        actual=validate_bounds(dict(bounds=bounds))
        np.testing.assert_array_equal(actual,bounds)
        scope=dict(bounds=actual,mode="add",selections=[dict(view=3)],coverage=dict(supported_views=[3,4,5]))
        box=make_boxes(scope)["current"][0]
        center=np.asarray(box["center"]);half=np.asarray(box["size"])/2
        np.testing.assert_allclose(center-half,actual[0]);np.testing.assert_allclose(center+half,actual[1])
        self.assertEqual(make_boxes(scope)["old"],[])
    def test_rejects_invalid_and_oversize_boxes(self):
        for bounds in (None,[[0,0,0],[1,1,float("nan")]],[[0,0,0],[1,1,float("inf")]],
                       [[1,1,1],[0,0,0]],[[0,0,0],[.01,1,1]],[[0,0,0],[4,4,4]],[[0,0],[1,1]]):
            with self.subTest(bounds=bounds),self.assertRaises(ValueError):validate_bounds(dict(bounds=bounds))
    def test_rejects_unknown_update_mode(self):
        with self.assertRaises(ValueError):validate_bounds(dict(bounds=[[0,0,0],[1,1,1]],mode="delete_everything"))
    def test_only_inside_points_allowed(self):
        scope=dict(bounds=[[1,2,3],[2,3,4]])
        np.testing.assert_array_equal(contains(np.array([[1,2,3],[2,3,4],[1.5,2.5,3.5],[0,2.5,3.5],[1.5,2.5,4.1]]),scope),[True,True,True,False,False])
if __name__=="__main__":unittest.main()
