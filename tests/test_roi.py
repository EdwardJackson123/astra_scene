import json,tempfile,unittest
from pathlib import Path
import numpy as np
from plyfile import PlyData,PlyElement
from astra_scene.io import ROOT
from astra_scene.roi import contains,validate_selections,make_boxes,audit,scope_mask
from test_core import row

class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.frames=[dict(view=0,crop=[0,50,100,150]),dict(view=2,crop=[0,50,100,150])]
        self.payload=dict(selections=[dict(view=0,rect=[.1,.1,.9,.9]),dict(view=2,rect=[.1,.1,.9,.9])])
    def test_normalized_rect_to_camera_pixels(self):
        s,_=validate_selections(self.payload,self.frames)
        np.testing.assert_array_equal(s[0]["pixels"],[10,60,90,140])
    def test_duplicate_view_rejected(self):
        self.payload["selections"][1]["view"]=0
        with self.assertRaises(ValueError):validate_selections(self.payload,self.frames)
    def test_nontraining_view_rejected(self):
        self.payload["selections"][1]["view"]=1
        with self.assertRaises(ValueError):validate_selections(self.payload,self.frames)
    def test_rect_outside_image_rejected(self):
        self.payload["selections"][1]["rect"][2]=2
        with self.assertRaises(ValueError):validate_selections(self.payload,self.frames)
    def test_default_add_has_no_deletable_box(self):
        scope=dict(bounds=[[0,0,0],[1,1,1]],selections=[dict(view=0)],coverage=dict(supported_views=[0,2,4]),mode="add")
        b=make_boxes(scope);self.assertEqual(b["old"],[]);self.assertTrue(b["manual_scope"]);self.assertEqual(b["margin_m"],0)
    def test_center_scope_bounds(self):
        np.testing.assert_array_equal(contains(np.array([[.5,.5,.5],[2,0,0]]),dict(bounds=[[0,0,0],[1,1,1]])),[True,False])
    def test_projected_scope_excludes_distant_pixels(self):
        mask=scope_mask(row(),dict(bounds=[[-.1,-.1,1],[.1,.1,2]]))
        self.assertTrue(mask[4,4]);self.assertFalse(mask[0,0])
    def test_export_audit_catches_outside_change(self):
        with tempfile.TemporaryDirectory(dir=ROOT/"runs") as temp:
            p=Path(temp);(p/"polish").mkdir();(p/"update/boxer_native_game_v1/guarded").mkdir(parents=True)
            v=np.array([(0.,0.,1.,.5),(3.,3.,3.,.6)],dtype=[("x","f4"),("y","f4"),("z","f4"),("opacity","f4")])
            PlyData([PlyElement.describe(v,"vertex")]).write(p/"base.ply")
            (p/"config.json").write_text(json.dumps(dict(map_ply=str(p/"base.ply"))))
            (p/"manual_scope.json").write_text(json.dumps(dict(bounds=[[-1,-1,0],[1,1,2]],mode="replace")))
            np.savez(p/"update/boxer_native_game_v1/guarded/lineage.npz",original_ids=np.array([0,1]))
            candidate=v.copy();candidate["opacity"][0]=.9
            PlyData([PlyElement.describe(candidate,"vertex")]).write(p/"polish/game_candidate.ply")
            audit(p)
            candidate["opacity"][1]=.9
            PlyData([PlyElement.describe(candidate,"vertex")]).write(p/"polish/game_candidate.ply")
            with self.assertRaises(ValueError):audit(p)
if __name__=="__main__":unittest.main()
