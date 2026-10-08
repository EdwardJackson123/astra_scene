import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from binary_grid import convert
from navigation_candidate import classes
class BinaryTests(unittest.TestCase):
 def test_both_encodings_no_unknown_or_new_free(self):
  for negate in (0,1):
   a=np.arange(256,dtype=np.uint8);meta=dict(negate=negate)
   b,r=convert(a,meta)
   self.assertEqual(set(b.tolist()),{0,255});self.assertFalse(classes(b,meta)[2].any())
   np.testing.assert_array_equal(classes(a,meta)[1],classes(b,meta)[1])
   self.assertEqual(r['unknown_cells'],0)
 def test_idempotent(self):
  a=np.array([0,205,254,255],np.uint8);b,_=convert(a,{})
  c,_=convert(b,{});np.testing.assert_array_equal(b,c)
if __name__=='__main__':unittest.main()
