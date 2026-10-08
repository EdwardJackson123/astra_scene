import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from confirmed_grid import confirmed_result

class ConfirmedGridTests(unittest.TestCase):
    def test_uncertainty_preserves_original_and_confirmed_changes_survive(self):
        original=np.array([[0,255,0,255,147,0]],np.uint8)
        candidate=np.array([[147,147,254,0,147,254]],np.uint8)
        allowed=np.array([[True,True,True,True,True,False]])
        expected=np.array([[0,255,254,0,147,0]],np.uint8)
        for negate in [0,1]:
            o,c,w=(original,candidate,expected) if not negate else (255-original,255-candidate,255-expected)
            actual,counts=confirmed_result(o,c,allowed,dict(negate=negate,free_thresh=.196,occupied_thresh=.65))
            np.testing.assert_array_equal(actual,w)
            self.assertEqual(counts['cleared_obstacles'],1);self.assertEqual(counts['added_obstacles'],1)
            self.assertEqual(counts['uncertain_edits_not_applied'],2)

if __name__=='__main__':unittest.main()
