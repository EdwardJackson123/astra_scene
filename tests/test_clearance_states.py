import unittest,numpy as np
from astra_scene.geometry import partial_clearance_unknown
class ClearanceStateTests(unittest.TestCase):
 def test_unknown_requires_partial_observation(self):
  np.testing.assert_array_equal(partial_clearance_unknown([0,4,6],10,[0,0,0],[False]*3,[False]*3),[False,False,True])
 def test_current_obstacle_or_surface_blocks_unknown(self):
  np.testing.assert_array_equal(partial_clearance_unknown([8,8],10,[1,0],[False,True],[False,False]),[False,False])
 def test_confirmed_free_not_reclassified(self):
  self.assertFalse(partial_clearance_unknown([10],10,[0],[False],[True])[0])
