import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from navigation_candidate import classes,constrained_result

class NavigationTests(unittest.TestCase):
    def test_outside_scope_is_preserved_and_unknown_is_not_free(self):
        old=np.array([[0,0,255,0]],np.uint8);new=np.array([[254,147,0,254]],np.uint8)
        result,r=constrained_result(old,new,np.array([[True,True,True,False]]),{})
        np.testing.assert_array_equal(result,[[254,147,0,0]])
        self.assertEqual(r['cleared_obstacles'],1);self.assertEqual(r['occupied_to_unknown'],1);self.assertEqual(r['added_obstacles'],1)
        occ,free,unknown=classes(result,{})
        self.assertTrue(unknown[0,1]);self.assertFalse(free[0,1]);self.assertTrue(r['outside_allowed_unchanged'])

    def test_negate_and_empty_scope(self):
        old=np.array([[255,0]],np.uint8);new=np.array([[1,255]],np.uint8)
        result,r=constrained_result(old,new,np.ones_like(old,bool),dict(negate=1))
        self.assertEqual(r['cleared_obstacles'],1);self.assertEqual(r['added_obstacles'],1)
        result,r=constrained_result(old,new,np.zeros_like(old,bool),dict(negate=1));np.testing.assert_array_equal(result,old);self.assertEqual(r['changed_cells'],0)

if __name__=='__main__':unittest.main()
