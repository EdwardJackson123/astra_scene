import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from enclosed_grid_repair import repair, apply_to_grid


class RepairTests(unittest.TestCase):
    def setUp(self):
        self.occ = np.zeros((9, 9), bool)
        self.occ[2:7, 2:7] = True
        self.occ[3:6, 3:6] = False
        self.scope = np.ones((9, 9), bool)
        self.hits = np.full((9, 9), 2)
        self.ground = np.zeros((9, 9), int)

    def run_repair(self):
        return repair(self.occ, ~self.occ, self.scope, self.hits, self.ground)[0]

    def test_enclosed_supported_hole_only(self):
        added = self.run_repair()
        self.assertEqual(int(added.sum()), 9)
        self.assertFalse((added & self.occ).any())

    def test_diagonal_exit_remains_open(self):
        self.occ[2, 2] = False
        self.assertFalse(self.run_repair().any())

    def test_ground_scope_and_missing_hits_veto_whole_hole(self):
        self.ground[4, 4] = 3
        self.assertFalse(self.run_repair().any())
        self.ground[:] = 0
        self.scope[4, 4] = False
        self.assertFalse(self.run_repair().any())
        self.scope[:] = True
        self.hits[4, 4] = 0
        self.assertFalse(self.run_repair().any())

    def test_map_negate_and_resolution(self):
        for negate in (0,1):
            grid=np.where(self.occ,255 if negate else 0,0 if negate else 255).astype('uint8')
            result,added,report=apply_to_grid(grid,dict(resolution=.05,negate=negate),self.scope,self.hits,self.ground)
            self.assertEqual(int(added.sum()),9)
            self.assertTrue((result[added]==(255 if negate else 0)).all())
            self.assertTrue(np.array_equal(result[~added],grid[~added]))
            _,added,_=apply_to_grid(grid,dict(resolution=.1,negate=negate),self.scope,self.hits,self.ground)
            self.assertFalse(added.any())

    def test_misaligned_evidence_rejected(self):
        with self.assertRaises(ValueError):repair(self.occ,~self.occ,self.scope,self.hits[:2],self.ground)
