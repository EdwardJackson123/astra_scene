import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from semantic_gaussian_grid import close_object_gaps
from grid_shape_semantics import validate

class SemanticGaussianTests(unittest.TestCase):
    def test_open_frame_hole_and_existing_obstacles_are_preserved(self):
        m=np.zeros((15,15),bool);m[4:11,4:11]=True;m[7,7]=False
        result,added=close_object_gaps(m,np.ones_like(m),solid=False)
        self.assertFalse(result[7,7]);self.assertTrue(result[m].all())
        solid,_=close_object_gaps(m,np.ones_like(m),solid=True);self.assertTrue(solid[7,7])

    def test_no_change_outside_object_or_across_wide_corridor(self):
        m=np.zeros((15,15),bool);m[3:12,2:5]=True;m[3:12,8:11]=True
        scope=np.zeros_like(m);scope[2:13,1:12]=True
        result,added=close_object_gaps(m,scope)
        self.assertFalse(result[:,5:8].any());self.assertFalse((added&~scope).any())
        np.testing.assert_array_equal(result[~scope],m[~scope])

    def test_model_cannot_invent_coordinates_or_evidence(self):
        batch=[dict(id='x',images=[dict(view=1),dict(view=3)])]
        item=dict(id='x',footprint='open_frame',operation='close_small_gaps',supporting_views=[1,3],reason='same frame')
        self.assertEqual(validate([item],batch),[item])
        for change in [dict(operation='delete'),dict(supporting_views=[1]),dict(supporting_views=[1,2]),dict(footprint='uncertain')]:
            with self.assertRaises(ValueError):validate([dict(item,**change)],batch)

if __name__=='__main__':unittest.main()
