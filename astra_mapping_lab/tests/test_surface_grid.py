import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from surface_grid import retire_cells
from semantic_grid_review import validate

class SurfaceGridTests(unittest.TestCase):
    def test_removal_does_not_imply_free_and_current_obstacle_wins(self):
        total=np.full(6,10);cleared=np.array([10,10,10,0,10,10]);stable=np.array([0,0,0,0,1,0])
        semantic=np.array([True,True,True,True,True,False]);occupied=np.array([False,False,True,False,False,False]);ground=np.array([3,0,3,3,3,3])
        free,unknown,retired=retire_cells(total,cleared,stable,semantic,occupied,ground,np.ones(6,bool))
        np.testing.assert_array_equal(free,[True,False,False,False,False,False]);np.testing.assert_array_equal(unknown,[False,True,False,False,False,False])
        self.assertTrue(retired[2])
    def test_missing_depth_never_clears(self):
        a=np.zeros(4);free,unknown,retired=retire_cells(a,a,a,np.ones(4,bool),np.zeros(4,bool),a,np.ones(4,bool));self.assertFalse(retired.any())
    def test_partial_old_surface_is_unknown_even_with_ground(self):
        total=np.array([100,100,100]);clear=np.array([85,99,100]);zero=np.zeros(3)
        free,unknown,retired=retire_cells(total,clear,zero,np.ones(3,bool),np.zeros(3,bool),np.full(3,20),np.ones(3,bool))
        np.testing.assert_array_equal(free,[False,False,True])
        np.testing.assert_array_equal(unknown,[True,True,False])
        self.assertTrue(retired.all())
    def test_semantic_output_ids_views_and_evidence_are_checked(self):
        batch=[dict(id='old1',images=[dict(view=1),dict(view=2)])]
        item=dict(id='old1',old_state='gone',structure='movable',supporting_views=[1,2],reason='two views show background')
        self.assertEqual(validate([item],batch),[item])
        for change in [dict(supporting_views=[1]),dict(supporting_views=[1,3]),dict(id='invented'),dict(old_state='delete'),dict(supporting_views=[1,1])]:
            with self.assertRaises(ValueError):validate([dict(item,**change)],batch)

if __name__=='__main__':unittest.main()
