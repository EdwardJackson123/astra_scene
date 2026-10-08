import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from grid_review_report import decision_reasons,observation_targets
from semantic_grid_review import validate

class GridReviewTests(unittest.TestCase):
    def test_current_obstacle_reason_wins_and_unknown_is_separate(self):
        shape=(1,4);old=np.zeros(shape,np.uint8);result=np.array([[0,0,147,254]],np.uint8)
        e=dict(current_obstacle=np.array([[True,False,False,False]]),semantic=np.array([[False,False,True,True]]),
               total=np.full(shape,10),stable=np.zeros(shape),cleared=np.array([[0,0,9,10]]),retired=np.array([[False,False,True,True]]))
        meta=dict(negate=0,free_thresh=.196,occupied_thresh=.65)
        np.testing.assert_array_equal(decision_reasons(old,result,np.ones(shape,bool),meta,e),[[2,1,6,8]])
        # The same probabilities encoded with negate=1 must have the same explanations.
        np.testing.assert_array_equal(decision_reasons(255-old,255-result,np.ones(shape,bool),dict(meta,negate=1),e),[[2,1,6,8]])

    def test_targets_exclude_buildings_positive_obstacles_and_free(self):
        labels=np.array([[1,1,1,0,5,5,5],[2,2,2,0,6,6,6],[8,8,8,0,0,0,0]],np.uint8)
        allowed=np.ones(labels.shape,bool);allowed[0,6]=False
        result=observation_targets(labels,{},allowed,dict(origin=[10,20,np.pi/2],resolution=.05))
        self.assertFalse(result['executable_navigation_goals'])
        self.assertEqual(len(result['targets']),1)
        target=result['targets'][0];self.assertEqual(target['cells'],5)
        self.assertEqual(target['unknown_cells'],3)
        self.assertTrue(np.isfinite(target['target_map_xy']).all())

    def test_invalid_semantic_items_fail_validation(self):
        with self.assertRaises(ValueError):validate(['not an object'],[dict(id='x',images=[])])

if __name__=='__main__':unittest.main()
