import unittest
import numpy as np
from astra_scene.quality import structural_metrics
from astra_scene.training_views import resize_geometry
from astra_scene.reconstruction import owners

class QualityTests(unittest.TestCase):
    def setup_scene(self):
        depth=np.ones((32,32),np.float32)*2
        row=dict(depth=depth,valid=np.ones_like(depth,bool),K=np.array([[100,0,16],[0,100,16],[0,0,1]]),
            pose=np.eye(4),rgb=np.ones((32,32,3),np.float32)*.5)
        scope=dict(bounds=[[-.4,-.4,1.9],[.4,.4,2.1]])
        return row,scope
    def test_exact_surface_passes(self):
        r,s=self.setup_scene()
        self.assertTrue(structural_metrics(r['rgb'],r['depth'],np.ones((32,32)),r,s)['passed'])
    def test_opaque_background_cannot_hide_holes(self):
        r,s=self.setup_scene()
        m=structural_metrics(r['rgb'],r['depth']+1,np.ones((32,32)),r,s)
        self.assertFalse(m['passed']);self.assertEqual(m['missing_surface_fraction'],1)
    def test_floaters_fail_even_if_color_matches(self):
        r,s=self.setup_scene()
        m=structural_metrics(r['rgb'],r['depth']-.5,np.ones((32,32)),r,s)
        self.assertFalse(m['passed']);self.assertEqual(m['foreground_disagreement_fraction'],1)
    def test_transparent_surface_fails(self):
        r,s=self.setup_scene()
        self.assertFalse(structural_metrics(r['rgb'],r['depth'],np.zeros((32,32)),r,s)['passed'])
    def test_invalid_rgb_does_not_pass_comparisons(self):
        r,s=self.setup_scene();rgb=r['rgb'].copy();rgb[0,0]=np.nan
        self.assertFalse(structural_metrics(rgb,r['depth'],np.ones((32,32)),r,s)['passed'])
    def test_small_bad_region_not_hidden_by_large_good_region(self):
        r,s=self.setup_scene();small=dict(bounds=[[.1,.1,1.9],[.3,.3,2.1]])
        rgb=r['rgb'].copy();rgb[22:31,22:31]=0
        m=structural_metrics(rgb,r['depth'],np.ones((32,32)),r,dict(regions=[s,small]),dict(min_pixels=10))
        self.assertFalse(m['passed']);self.assertFalse(m['regions'][1]['passed'])
    def test_resize_preserves_camera_rays(self):
        K=np.array([[400,0,259],[0,420,259],[0,0,1.]])
        new=resize_geometry(K,(518,518),(112,406,0,518),(576,1024))
        uv=np.array([123.,234.,1.])
        transformed=np.array([(uv[0]+.5)*1024/518-.5,(uv[1]-112+.5)*576/294-.5,1.])
        np.testing.assert_allclose(np.linalg.inv(K)@uv,np.linalg.inv(new)@transformed)
    def test_fusion_obeys_rotated_boxes(self):
        angle=np.pi/4;c,s=np.cos(angle),np.sin(angle);pose=np.eye(4);pose[:2,:2]=[[c,-s],[s,c]]
        boxes=dict(old=[],current=[dict(pose=pose.tolist(),bounds=[-1,1,-.1,.1,-.1,.1])])
        self.assertEqual(owners(np.array([[.5,.5,0],[.5,-.5,0]]),boxes).tolist(),[0,-1])

if __name__=='__main__':unittest.main()
