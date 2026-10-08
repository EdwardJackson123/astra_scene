import unittest
import numpy as np
from astra_scene.context_fusion import field,context_scope,verified_prepared
from astra_scene.roi import contains


class ContextFieldTests(unittest.TestCase):
    def setUp(self):
        self.scope=dict(mode='replace',bounds=[[0,0,0],[1,1,1]])

    def test_core_frozen_exterior_and_smooth_transition(self):
        p=np.array([[.5,.5,.5],[1,.5,.5],[1.15,.5,.5],[1.3,.5,.5],[2,0,0]])
        np.testing.assert_allclose(field(p,self.scope,.3),[1,1,.5,0,0],atol=1e-7)

    def test_union_does_not_fill_large_gap(self):
        s=dict(regions=[self.scope,dict(bounds=[[3,0,0],[4,1,1]])])
        self.assertEqual(field(np.array([[2,.5,.5]]),s,.3)[0],0)

    def test_expanded_scope_contains_transition_support(self):
        rng=np.random.default_rng(7);p=rng.uniform(-1,2,(5000,3))
        expanded=context_scope(self.scope,.3)
        self.assertTrue(contains(p[field(p,self.scope,.3)>0],expanded).all())
        self.assertEqual(self.scope['bounds'],[[0,0,0],[1,1,1]])

    def test_rotated_region_and_expansion(self):
        s=dict(bounds=[[-1,-1,-1],[1,1,1]],oriented_box=dict(center=[0,0,0],size=[2,1,1],quaternion_xyzw=[0,0,np.sin(np.pi/8),np.cos(np.pi/8)]))
        p=np.array([[0,0,0],[.8,.8,0],[2,2,0]])
        w=field(p,s,.3)
        self.assertEqual(w[0],1);self.assertGreater(w[1],0);self.assertEqual(w[2],0)
        self.assertTrue(contains(p[w>0],context_scope(s,.3)).all())

    def test_nonpositive_band_rejected(self):
        with self.assertRaises(ValueError):field(np.zeros((1,3)),self.scope,0)

    def test_prepared_rejects_changed_split_or_candidate(self):
        import tempfile,json
        from pathlib import Path
        from astra_scene.io import ROOT,sha256
        with tempfile.TemporaryDirectory(dir=ROOT/'runs') as directory:
            root=Path(directory);(root/'job').mkdir()
            split=dict(train_views=[0],validation_views=[1],test_views=[2])
            (root/'job/refit_report.json').write_text(json.dumps(split))
            (root/'scene.ply').write_bytes(b'candidate');(root/'baseline.ply').write_bytes(b'baseline')
            spec=dict(ply_sha256='fresh',views_sha256='views',base_sha256='base')
            inputs=dict(fresh_sha256='fresh',views_sha256='views',base_sha256='base',baseline_sha256=sha256(root/'baseline.ply'),split=split)
            (root/'inputs.json').write_text(json.dumps(inputs))
            (root/'report.json').write_text(json.dumps(dict(candidate_sha256=sha256(root/'scene.ply'))))
            verified_prepared(root,spec,root/'baseline.ply',root)
            (root/'job/refit_report.json').write_text(json.dumps(dict(train_views=[1],validation_views=[0],test_views=[2])))
            with self.assertRaises(ValueError):verified_prepared(root,spec,root/'baseline.ply',root)
            (root/'job/refit_report.json').write_text(json.dumps(split))
            (root/'scene.ply').write_bytes(b'changed')
            with self.assertRaises(ValueError):verified_prepared(root,spec,root/'baseline.ply',root)
