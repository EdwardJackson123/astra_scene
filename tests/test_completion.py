import unittest
from astra_scene.completion import clusters,overlap
class CompletionTests(unittest.TestCase):
 def test_adjacent_objects_are_not_merged(self):
  a=dict(id="a",bounds=[[0,0,0],[1,1,1]])
  b=dict(id="b",bounds=[[1.01,0,0],[2,1,1]])
  self.assertEqual(len(clusters([a,b])),2)
 def test_overlapping_component_is_grouped(self):
  a=dict(id="a",bounds=[[0,0,0],[1,1,2]])
  b=dict(id="b",bounds=[[.1,.1,.3],[.8,.8,1.8]])
  self.assertEqual(len(clusters([a,b])),1)
 def test_no_overlap_has_zero_score(self):
  self.assertEqual(overlap([[0,0,0],[1,1,1]],[[2,2,2],[3,3,3]]),0)
