import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from model_primary_contour import refine,split_count
from navigation_candidate import classes
class ModelPrimaryTests(unittest.TestCase):
 def test_unknown_not_free_and_preserve_model_legacy_scope(self):
  for negate in (0,1):
   for thresholds in ({},{'free_thresh':.3,'occupied_thresh':.6}):
    meta=dict(negate=negate,**thresholds);free=1 if negate else 254;occ=255 if negate else 0
    before=np.full((9,9),free,np.uint8);old=before.copy();allowed=np.ones((9,9),bool);zero=np.zeros((9,9),bool);hits=np.zeros((9,9),int);model=zero.copy()
    before[2,2]=occ;hits[2,2]=3
    before[3,3]=old[3,3]=occ;before[4,4]=occ;model[4,4]=True
    before[5,5]=occ;allowed[5,5]=False
    result,_,report,_=refine(before,old,allowed,meta,model,zero,zero,hits,hits*0,hits*0,zero,{})
    self.assertTrue(classes(result,meta)[2][2,2]);self.assertEqual(report['newly_free_cells'],0)
    for y,x in [(3,3),(4,4),(5,5)]:self.assertEqual(result[y,x],occ)
 def test_split_detector_catches_one_cell_door_closure(self):
  free=np.ones((9,9),bool);free[:,4]=False;free[4,4]=True
  after=free.copy();after[4,4]=False
  self.assertEqual(split_count(free,after),1)
 def test_no_unobserved_shape_addition(self):
  a=np.full((9,9),254,np.uint8);a[3:6,3:6]=0;a[4,4]=254
  z=np.zeros_like(a,bool);ones=np.ones_like(z);hits=np.zeros_like(a,int)
  result,_,report,_=refine(a,a,ones,{},a==0,z,z,hits,hits,hits,z,{'object':ones})
  np.testing.assert_array_equal(result,a);self.assertEqual(report['added_cells'],0)
if __name__=='__main__':unittest.main()
