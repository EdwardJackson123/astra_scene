import unittest
from astra_scene.object_policy import accept_trial,chair_bounds
class ObjectPolicyTests(unittest.TestCase):
 def records(self,gains,pixels=150):
  return [dict(changed_pixels=pixels,before_l1=.1,after_l1=.1-g) for g in gains]
 def test_multiple_views_required(self):
  self.assertFalse(accept_trial(self.records([.02,.02])))
 def test_noop_rejected(self):
  self.assertFalse(accept_trial(self.records([0,0,0,0])))
 def test_single_good_view_cannot_hide_regression(self):
  self.assertFalse(accept_trial(self.records([.1,.01,-.04,.01])))
 def test_consistent_improvement(self):
  self.assertTrue(accept_trial(self.records([.02,.01,.01,-.001])))
 def test_tiny_effect_not_enough(self):
  self.assertFalse(accept_trial(self.records([.1]*5,40)))
 def test_chair_bounds_do_not_assume_world_floor(self):
  b=chair_bounds([[1,2,8],[2,3,9]])
  self.assertEqual(b[0][2],8)
  self.assertAlmostEqual(b[0][0],.9)
