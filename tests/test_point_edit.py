import unittest
import numpy as np
from astra_scene.point_edit import decisions, accept


class PointEditTests(unittest.TestCase):
    def test_unknown_and_occlusion_do_not_delete(self):
        reject, restore = decisions([0,0], [0,3], [1.,1.])
        self.assertFalse(reject.any())
        self.assertFalse(restore.any())

    def test_surface_contradiction_blocks_removal(self):
        reject, _ = decisions([0,1], [4,20], [0.,0.])
        np.testing.assert_array_equal(reject, [True,False])

    def test_restore_requires_support_no_free_and_no_replacement(self):
        _, restore = decisions([3,3,2,3], [0,1,0,0], [.09,.2,.2,.04])
        np.testing.assert_array_equal(restore, [True,False,False,False])

    def records(self):
        return [dict(pixels=200, before_l1=.1, after_l1=.09,
                     background_delta=0, missing_increase=0,
                     foreground_increase=0) for _ in range(3)]

    def test_improvement_requires_three_views(self):
        self.assertTrue(accept(self.records()))
        self.assertFalse(accept(self.records()[:2]))

    def test_background_or_structure_regression_rejects_good_average(self):
        for key in ['background_delta','missing_increase','foreground_increase']:
            rows=self.records(); rows[0][key]=.03
            self.assertFalse(accept(rows))

    def test_nonfinite_and_worst_view_regression_rejected(self):
        for value in [np.nan,.12]:
            rows=self.records(); rows[0]['after_l1']=value
            self.assertFalse(accept(rows))


if __name__ == '__main__':
    unittest.main()
