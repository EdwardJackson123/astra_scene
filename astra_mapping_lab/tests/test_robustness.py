import sys, unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from robustness import classify_rays,decision,alignment_review


class RobustnessTests(unittest.TestCase):
    def test_occluded_and_invalid_depth_do_not_authorize_removal(self):
        points=np.array([[0,0,2],[0,0,2],[0,0,2],[0,0,2.]])
        row=dict(depth=np.array([[3.,1.,2.,np.nan]]),valid=np.array([[True,True,True,False]]),keep=np.ones((1,4),bool))
        def project(p,row):return np.arange(4),np.arange(4),np.zeros(4,int),p[:,2]
        c=classify_rays(points,[row]*3,[0,1,2],project)
        np.testing.assert_array_equal(c['clear'],[3,0,0,0])
        np.testing.assert_array_equal(c['occluded'],[0,3,0,0])
        np.testing.assert_array_equal(c['support'],[0,0,3,0])
        n=dict(support=np.full(4,3),clear=np.zeros(4,int),observed_views=3)
        d=decision(c,n,np.zeros(4,bool));self.assertFalse(d['accepted']);self.assertEqual(d['strict_free_fraction'],.25)
        with self.assertRaises(ValueError):classify_rays(points,[row],[0,0],project)

    def test_region_can_pass_without_weakening_other_region(self):
        old=dict(support=np.zeros(10,int),clear=np.full(10,3),observed_views=3)
        new=dict(support=np.full(10,3),clear=np.zeros(10,int),observed_views=3)
        self.assertTrue(decision(old,new,np.zeros(10,bool))['accepted'])
        self.assertFalse(decision(old,new,np.zeros(10,bool),alignment_ok=False)['accepted'])
        old['clear'][:]=0
        self.assertFalse(decision(old,new,np.zeros(10,bool))['accepted'])
        self.assertTrue(decision(old,new,np.ones(10,bool))['accepted'])

    def test_spatially_degenerate_alignment_rejected(self):
        x=np.linspace(0,3,240);a=np.column_stack([x,np.zeros_like(x),np.zeros_like(x)])
        reg=dict(scale=1.,rotation=np.eye(3).tolist(),translation=[0,0,0],accepted=True)
        report=alignment_review(a,a,np.arange(240)%12,reg)
        self.assertFalse(report['candidate_alignment_accepted']);self.assertIn('spatially_concentrated_anchors',report['reasons'])

    def test_pure_addition_has_no_fictitious_removal_requirement(self):
        old=dict(support=np.zeros(0,int),clear=np.zeros(0,int),observed_views=0)
        new=dict(support=np.full(10,3),clear=np.zeros(10,int),observed_views=3)
        self.assertTrue(decision(old,new,np.zeros(0,bool))['accepted'])

    def test_distributed_consistent_alignment_passes(self):
        rng=np.random.default_rng(7);a=rng.uniform(-2,2,(480,3));a[:,2]=0
        reg=dict(scale=1.,rotation=np.eye(3).tolist(),translation=[0,0,0],accepted=True,metric_scale_conflict=True)
        report=alignment_review(a,a,np.arange(480)%12,reg)
        self.assertTrue(report['candidate_alignment_accepted']);self.assertTrue(report['metric_scale_conflict'])
        self.assertFalse(report['navigation_deployment_accepted'])

if __name__=='__main__':unittest.main()
