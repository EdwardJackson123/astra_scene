import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from fusion_visibility import flat_floor,retirement_mask,replacement_visible


class VisibilityTests(unittest.TestCase):
    def test_image_box_prompt_rejects_missing_views_and_room_sized_boxes(self):
        from instance_box_refinement import validate
        self.assertEqual(len(validate([dict(view=1,valid=True,xyxy=[10,20,400,900])],[1])),1)
        for boxes in ([],[dict(view=1,valid=True,xyxy=[0,0,1000,1000])],[dict(view=1,valid=True,xyxy=[50,20,10,100])]):
            with self.assertRaises(ValueError):validate(boxes,[1])

    def test_boundary_review_rejects_partial_or_uncertain_instances(self):
        from instance_boundary_review import accepted
        a=dict(decision='update',scope_complete=True,change_type='layout',supporting_views=[1,2],missing_parts=[])
        self.assertTrue(accepted(a))
        for override in (dict(scope_complete=False),dict(decision='defer'),dict(missing_parts=['upper beam']),dict(supporting_views=[1]),dict(ai_validation_failed=True)):
            self.assertFalse(accepted(dict(a,**override)))

    def test_instance_volumes_keep_interior_without_bridging_two_objects(self):
        from instance_volume import InstanceVolumes
        corners=np.array([[x,y,z] for x in (0.,1.) for y in (0.,1.) for z in (0.,1.)])
        field=InstanceVolumes({'a':corners,'b':corners+[3,0,0]})
        np.testing.assert_array_equal(field.contains(np.array([[.5,.5,.5],[2,.5,.5],[3.5,.5,.5],[.5,.5,2]])),[True,False,True,False])

    def test_foreground_coverage_does_not_authorize_background_retirement(self):
        np.testing.assert_array_equal(replacement_visible(np.array([True,True,False]),np.array([2.,4.,4.]),np.array([4.,4.,4.])),[False,True,False])

    def test_window_sheet_centered_at_floor_is_not_floor(self):
        np.testing.assert_array_equal(flat_floor(np.array([True,True,False]),np.array([.02,.8,.02])),[True,False,False])

    def test_completion_needs_instance_views_coverage_and_permission(self):
        distance=np.array([.2,.2,.5,.2,.2,.2]);axis=np.array([.01,.01,.01,.3,.01,.01])
        votes=np.array([3,2,4,4,4,4]);covered=np.array([3,4,4,4,2,4]);allowed=np.array([True]*5+[False])
        np.testing.assert_array_equal(retirement_mask(distance,axis,votes,covered,allowed),[True,False,False,False,False,False])
