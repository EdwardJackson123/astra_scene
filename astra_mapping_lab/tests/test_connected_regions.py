import sys, unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from connected_regions import components, scope_weights, validate_review,confirmed_stale


class ConnectedRegionTests(unittest.TestCase):
    def test_disconnected_regions_never_fill_their_bounding_box(self):
        meta = dict(resolution=1., height=10, origin=[0, 0, 0])
        supports = np.array([[2, 5, 1], [8, 5, 1]])
        points = np.array([[2,5,1], [5,5,1], [8,5,1], [-.1,5,1]])
        np.testing.assert_allclose(scope_weights(points, supports, meta, [0,0,10,10]), [1,0,1,0])
        np.testing.assert_array_equal(scope_weights(points, [], meta, [0,0,10,10]), np.zeros(4))

    def test_permission_roi_cannot_authorize_unobserved_or_outside_edits(self):
        meta = dict(resolution=1., height=10, origin=[0,0,0])
        p = np.array([[1.99,5,0], [2.05,5,0], [7,5,0]])
        w = scope_weights(p, [[2.05,5,0]], meta, [2,2,8,8])
        self.assertEqual(w[0], 0); self.assertGreater(w[1], 0); self.assertEqual(w[2], 0)
        np.testing.assert_array_equal(scope_weights(p, [[1.99,5,0]], meta, [2,2,8,8]), np.zeros(3))

    def test_components_require_geometric_connectivity(self):
        p = np.array([[0,0,0],[.1,0,0],[.2,0,0],[3,0,0],[3.1,0,0],[7,0,0]])
        self.assertEqual([g.tolist() for g in components(p, .15, 2)], [[0,1,2],[3,4]])

    def test_deferred_neighbor_is_not_overwritten_by_transition(self):
        meta = dict(resolution=1., height=10, origin=[0,0,0])
        points = np.array([[2,5,1],[2.25,5,1]])
        w = scope_weights(points, [[2,5,1]], meta, [0,0,10,10], protected_supports=[[2.25,5,1]])
        self.assertGreater(w[0],0);self.assertEqual(w[1],0)

    def test_semantic_review_cannot_invent_geometry_or_evidence(self):
        proposal = [dict(id='r1', images=[dict(view=1),dict(view=4)])]
        good = dict(id='r1', decision='update', reason='two visible changes', supporting_views=[1,4], scope_complete=True)
        self.assertEqual(validate_review([good], proposal), [good])
        for patch in (dict(supporting_views=[1,9]), dict(supporting_views=[1,1]), dict(scope_complete=False), dict(bounds=[[0,0,0],[1,1,1]]), dict(id='r2')):
            with self.assertRaises(ValueError): validate_review([good | patch], proposal)
        with self.assertRaises(ValueError): validate_review([], proposal)

    def test_layout_nomination_is_distinct_from_publication_approval(self):
        from context_rgbd import nominate
        proposals=[dict(id=str(i),novelty_points=40,clearance_points=0) for i in range(5)]
        common=dict(decision='defer',change_type='layout',supporting_views=[1,4])
        items=[dict(common,id='0'),dict(common,id='1',decision='ignore'),dict(common,id='2',change_type='uncertain'),
               dict(common,id='3',ai_validation_failed=True),dict(common,id='4',supporting_views=[1])]
        self.assertEqual(nominate(items,proposals),['0'])
        self.assertEqual(items[0]['decision'],'defer')

    def test_confirmed_stale_is_not_a_transition_or_unknown_surface(self):
        np.testing.assert_array_equal(confirmed_stale([0,0,0,1,5],[0,5,6,20,0]),[False,False,True,False,False])


if __name__ == '__main__': unittest.main()
