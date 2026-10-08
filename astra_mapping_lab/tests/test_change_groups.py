import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from change_groups import build_groups,removal_masks

class ChangeGroupTests(unittest.TestCase):
    def test_floor_and_same_label_alone_cannot_merge_objects(self):
        rng=np.random.default_rng(4);points=rng.uniform(0,1,(100,3));points[:,2]=0
        regions=[dict(id='a'),dict(id='b')];proposals={k:dict(evidence_side='current',source_label='chair') for k in ('a','b')}
        masks={k+'__'+str(i):np.ones((8,8),bool) for k in ('a','b') for i in range(2)}
        r=build_groups(regions,dict(a=points,b=points),proposals,dict(a={},b={}),masks,0)
        self.assertEqual(len(r['groups']),2)
        points[:,2]=1;r=build_groups(regions,dict(a=points,b=points),proposals,dict(a={},b={}),masks,0)
        self.assertEqual(len(r['groups']),1)
        masks['b__1'][:]=False;r=build_groups(regions,dict(a=points,b=points),proposals,dict(a={},b={}),masks,0)
        self.assertEqual(len(r['groups']),2)

    def test_occlusion_thin_rods_and_no_replacement_are_preserved(self):
        old,new=removal_masks(np.array([0,4,0,4]),np.array([0,0,3,1]),np.array([0,0,3,0]),np.array([3,3,3,3]),np.array([.1,.01,.1,.1]))
        np.testing.assert_array_equal(old,[False,True,True,False]);np.testing.assert_array_equal(new,[False,False,False,False])

if __name__=='__main__':unittest.main()
