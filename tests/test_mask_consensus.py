import unittest,numpy as np
from astra_scene.mask_consensus import select_points,edit_regression
class MaskConsensusTests(unittest.TestCase):
 def test_occluded_or_supported_old_point_not_deleted(self):
  selected=select_points([4,4,4],[4,4,4],[0,4,4],[0,1,0])
  self.assertEqual(selected.tolist(),[False,False,True])
 def test_inconsistent_masks_rejected(self):
  self.assertFalse(select_points([3],[10])[0])
 def fixture(self):
  before=np.zeros((40,40,3),np.uint8);mask=np.zeros((40,40),bool);mask[10:30,10:30]=True
  truth=before.copy();truth[mask]=200
  return before,mask,truth
 def test_no_op_not_accepted_as_background_preservation(self):
  before,mask,truth=self.fixture()
  self.assertIn("insufficient_target_improvement",edit_regression(before,before,truth,mask)["reasons"])
 def test_background_damage_blocks_even_good_foreground(self):
  before,mask,truth=self.fixture();after=np.full_like(before,200)
  self.assertIn("outside_edit_region_changed",edit_regression(before,after,truth,mask)["reasons"])
 def test_local_improvement_with_unchanged_background_passes(self):
  before,mask,truth=self.fixture()
  self.assertTrue(edit_regression(before,truth,truth,mask)["passed"])

class MissingMaskTests(unittest.TestCase):
 def test_missing_mask_is_not_negative_vote(self):
  from astra_scene.mask_consensus import mask_observation
  eligible,positive=mask_observation([True,False])
  self.assertEqual(eligible.tolist(),[False,False]);self.assertEqual(positive.tolist(),[False,False])
 def test_valid_mask_can_vote_against(self):
  from astra_scene.mask_consensus import mask_observation
  eligible,positive=mask_observation([True,True],[True,False])
  self.assertEqual(eligible.tolist(),[True,True]);self.assertEqual(positive.tolist(),[True,False])
