import unittest
from astra_scene.fusion_guard import apply_decision


class FusionGuardTests(unittest.TestCase):
    def test_defer_only_target_and_preserve_original_input(self):
        regions=[dict(id='a',bounds=[1]),dict(id='b',bounds=[2])]
        self.assertEqual(apply_decision(regions,'a',True),[regions[1]])
        self.assertEqual(len(regions),2)

    def test_rejected_trial_keeps_all_regions(self):
        regions=[dict(id='a'),dict(id='b')]
        self.assertEqual(apply_decision(regions,'a',False),regions)

    def test_unknown_or_ambiguous_identity_is_an_error(self):
        for regions,ident in [([dict(id='a')],'b'),([dict(id='a'),dict(id='a')],'a')]:
            with self.assertRaises(ValueError):apply_decision(regions,ident,True)

    def test_last_region_can_be_deferred_without_fake_no_change(self):
        self.assertEqual(apply_decision([dict(id='a')],'a',True),[])
