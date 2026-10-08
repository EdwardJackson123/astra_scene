import json,sys,tempfile,unittest
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from depth_io import save_depth,load_depth,quantize_mm,validate_depth_manifest
from recording import atomic

class DepthStorageTests(unittest.TestCase):
    def setUp(self):self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
    def tearDown(self):self.tmp.cleanup()
    def test_sensor_millimetres_roundtrip_and_invalid(self):
        mm=np.array([[0,1,250,6000,65535]],np.uint16);d=mm.astype(np.float32)/1000
        f=save_depth(self.root,'frame',d);self.assertTrue(np.array_equal(load_depth(self.root/f['depth'],f),d))
        q=quantize_mm(np.array([[np.nan,np.inf,-1,0,1.2344]],np.float32));self.assertEqual(q.tolist(),[[0,0,0,0,1234]])
        for v in [66.,.0001]:
            with self.assertRaises(ValueError):quantize_mm(np.array([[v]]))
    def test_encoding_scale_and_legacy_compatibility(self):
        f=save_depth(self.root,'new',np.ones((4,4),np.float32))
        with self.assertRaises(ValueError):load_depth(self.root/f['depth'],{**f,'depth_scale_m':1})
        with self.assertRaises(ValueError):validate_depth_manifest(dict(depth_unit='m',frames=[f]))
        Image.new('L',(4,4),2).save(self.root/'bad.png')
        with self.assertRaises(ValueError):load_depth(self.root/'bad.png',f)
        np.save(self.root/'old.npy',np.full((4,4),2.5,np.float32))
        self.assertAlmostEqual(float(load_depth(self.root/'old.npy',dict(depth='old.npy'))[0,0]),2.5)
        np.save(self.root/'uint.npy',np.ones((4,4),np.uint16))
        with self.assertRaises(ValueError):load_depth(self.root/'uint.npy',dict(depth='uint.npy'))
    def convert(self,root,m,filename):
        for f in m['frames']:f.update(save_depth(root,f['id'],np.load(root/f['depth'],allow_pickle=False)))
        m['depth_unit']='mm';atomic(root/filename,m)
    def test_png_tracking_and_spirula_export_keep_metric_scale(self):
        from tracking_demo import create_tracking_demo
        from trajectory import track
        from collection import export_spirula
        raw,_=create_tracking_demo(self.root/'raw',self.root/'ref');self.convert(self.root/'raw',raw,'capture_manifest.json')
        m=track(self.root/'raw',self.root/'tracked')
        self.assertEqual(len(m['frames']),8);self.assertAlmostEqual(m['frames'][-1]['pose_c2w'][0][3],.56,delta=.015)
        p=export_spirula(self.root/'tracked',self.root/'package',[48,48,52,52],[45,45,55,55]);self.assertGreater(p['initial_points'],20)
        lines=(self.root/'package/dataset/sparse/0/points3D.txt').read_text().splitlines()
        self.assertAlmostEqual(float(lines[0].split()[3]),2,delta=.02)
    def test_png_and_npy_replay_have_same_evidence(self):
        from demo_dataset import create_demo
        from replay import replay
        root=self.root/'demo';m=create_demo(root);old=replay(root,m['suggested_roi'],self.root/'old')
        self.convert(root,m,'manifest.json');new=replay(root,m['suggested_roi'],self.root/'new')
        self.assertEqual(old['regions'],new['regions']);self.assertEqual(old['accepted_cells'],new['accepted_cells'])
