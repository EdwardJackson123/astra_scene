import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from structured_grid import conservative_blocks,topology_audit

class StructuredGridTests(unittest.TestCase):
    def test_odd_dimensions_no_erasure_or_outside_changes(self):
        m=np.zeros((7,9),bool);m[2,2]=True;m[6,8]=True
        allowed=np.zeros_like(m);allowed[1:6,1:7]=True
        result=conservative_blocks(m,allowed)
        self.assertEqual(result.shape,m.shape);self.assertTrue(result[m].all())
        np.testing.assert_array_equal(result[~allowed],m[~allowed])
        np.testing.assert_array_equal(conservative_blocks(result,allowed),result)

    def test_detects_closed_corridor_between_free_cores(self):
        before=np.zeros((11,23),bool);before[2:9,2:9]=True;before[2:9,14:21]=True;before[5,9:14]=True
        after=before.copy();after[5,11]=False
        audit=topology_audit(before,after,.1)
        self.assertEqual(len(audit['split_free_space_cores']),1)
        self.assertFalse(topology_audit(before,before,.1)['split_free_space_cores'])

if __name__=='__main__':unittest.main()
