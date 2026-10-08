import unittest,importlib.util
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scipy.spatial.transform import Rotation
from astra_scene.roi import validate_oriented_box,box_corners,box_geometry,contains,make_boxes,scope_mask
from astra_scene.io import ROOT

class OrientedScopeTests(unittest.TestCase):
    def scope(self,angles=(0,0,45)):
        return dict(oriented_box=dict(center=[2,3,1],size=[2,.2,.4],
            quaternion_xyzw=Rotation.from_euler("XYZ",angles,degrees=True).as_quat().tolist()),
            mode="replace",selections=[dict(view=0)],coverage=dict(supported_views=[0,1,2]))
    def test_world_enclosure_is_not_update_region(self):
        scope=self.scope();corners=box_corners(scope)
        # This point lies in the world AABB but outside the actual rotated box.
        p=np.array([[2.6,2.4,1]])
        self.assertTrue(((p>=corners.min(0))&(p<=corners.max(0))).all())
        self.assertFalse(contains(p,scope)[0])
    def test_all_axis_rotation_membership(self):
        scope=self.scope((17,-23,41));center,size,rot=box_geometry(scope)
        local=np.array([[0,0,0],[.99,.09,.19],[0,.101,0],[1.01,0,0]])
        np.testing.assert_array_equal(contains(local@rot.T+center,scope),[True,True,False,False])
    def test_native_boxes_keep_rotation_and_dimensions(self):
        scope=self.scope((17,-23,41));center,size,rot=box_geometry(scope);b=make_boxes(scope)["current"][0]
        np.testing.assert_allclose(np.asarray(b["pose"])[:3,:3],rot)
        np.testing.assert_allclose(b["size"],size)
        np.testing.assert_allclose(np.asarray(b["bounds"]).reshape(3,2),np.c_[-size/2,size/2])
        np.testing.assert_allclose(b["center"],center)
    def test_invalid_orientation_rejected(self):
        for q in ([0,0,0,0],[0,0,0,2],[0,0,float("nan"),1],[0,1,0]):
            obb=self.scope()["oriented_box"];obb["quaternion_xyzw"]=q
            with self.subTest(q=q),self.assertRaises(ValueError):validate_oriented_box(dict(oriented_box=obb))
    def test_rotation_does_not_enlarge_local_size_gate(self):
        obb=dict(center=[0,0,0],size=[4,.1,.1],quaternion_xyzw=Rotation.from_euler("z",45,degrees=True).as_quat())
        result=validate_oriented_box(dict(oriented_box=obb));self.assertEqual(result["size"],[4,.1,.1])
    def test_projected_rotated_box_mask(self):
        scope=self.scope();scope["oriented_box"]["center"]=[0,0,3]
        row=dict(pose=np.eye(4),K=np.array([[100,0,100],[0,100,100],[0,0,1]]),depth=np.ones((200,200)))
        mask=scope_mask(row,scope);self.assertTrue(mask[100,100]);self.assertFalse(mask[80,120])
    def test_multiple_regions_do_not_fill_the_gap(self):
        a=self.scope((0,0,0));b=self.scope((0,0,0));b["oriented_box"]["center"]=[8,3,1]
        scope=dict(regions=[a,b],mode="replace")
        np.testing.assert_array_equal(contains(np.array([[2,3,1],[5,3,1],[8,3,1]]),scope),[True,False,True])
        boxes=make_boxes(scope);self.assertEqual(len(boxes["old"]),2);self.assertEqual(len(boxes["current"]),2)
    def test_delete_preserves_surviving_ply_rows(self):
        import tempfile
        from plyfile import PlyData,PlyElement
        from astra_scene.deletion import write_filtered
        with tempfile.TemporaryDirectory(dir=ROOT/"runs") as folder:
            path=Path(folder);v=np.array([(2.,3.,1.,7.),(5.,3.,1.,8.),(8.,3.,1.,9.)],dtype=[("x","f4"),("y","f4"),("z","f4"),("opacity","f4")])
            PlyData([PlyElement.describe(v,"vertex")],text=False).write(str(path/"base.ply"))
            removed,xyz=write_filtered(path/"base.ply",self.scope(),path/"candidate.ply")
            np.testing.assert_array_equal(removed,[True,False,False])
            actual=PlyData.read(str(path/"candidate.ply"))["vertex"].data
            self.assertEqual(actual.tobytes(),v[1:].tobytes())
            original=PlyData.read(str(path/"base.ply"))["vertex"].data
            self.assertEqual(original.tobytes(),v.tobytes())
    @unittest.skipUnless((ROOT/"vendor/phone/legacy_phone_v2/boxer_game_guard.py").is_file(), "Optional external Boxer backend is not installed")
    def test_native_guard_membership_and_clamping(self):
        import torch
        spec=importlib.util.spec_from_file_location("native_boxer_guard",ROOT/"vendor/phone/legacy_phone_v2/boxer_game_guard.py")
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        scope=self.scope((17,-23,41));center,size,rot=box_geometry(scope)
        g=module.BoxerGuard.__new__(module.BoxerGuard);g.nold=0;g.boxes=[make_boxes(scope)["current"][0]]
        g.rot=torch.tensor(rot[None],dtype=torch.float32);g.inv=torch.linalg.inv(g.rot)
        g.center=torch.tensor(center[None],dtype=torch.float32)
        g.low=torch.tensor(-size[None]/2,dtype=torch.float32);g.high=-g.low
        points=np.array([[0,0,0],[0,.15,0]])@rot.T+center
        xyz=torch.tensor(points,dtype=torch.float32)
        self.assertEqual(g.membership(xyz).tolist(),[0,-1])
        # Force an active new center outside the oriented face and run real post-step clamp.
        g.owner=torch.tensor([0,0]);g.ids=torch.tensor([-1,-1]);g.anchor=xyz.clone()
        g.scale_anchor=torch.zeros((2,3));g.model=SimpleNamespace(_xyz=xyz.clone(),_scaling=torch.zeros((2,3)))
        g.audit={"optimizer_steps":0};g.after_step(None,None,None)
        self.assertTrue(contains(g.model._xyz.numpy(),scope).all())

if __name__=="__main__":unittest.main()
