import unittest
from astra_scene.instance_evidence import existence_score,retire_surface
class InstanceEvidenceTests(unittest.TestCase):
 def test_occlusion_is_neutral(self):
  self.assertAlmostEqual(float(existence_score([0],[0],[100])[0]),.9)
 def test_single_miss_cannot_retire(self):
  score=existence_score([0],[1],[0])
  self.assertFalse(retire_surface([0],[1],score,[True])[0])
 def test_repeated_free_evidence_retires_owned_surface(self):
  score=existence_score([0],[10],[20])
  self.assertTrue(retire_surface([0],[10],score,[True])[0])
 def test_current_surface_conflict_blocks(self):
  score=existence_score([1],[16],[0])
  self.assertFalse(retire_surface([1],[16],score,[True])[0])
 def test_unowned_geometry_is_preserved(self):
  self.assertFalse(retire_surface([0],[16],existence_score([0],[16],[0]),[False])[0])
