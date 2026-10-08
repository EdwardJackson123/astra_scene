import io,json,tempfile,unittest,zipfile
from pathlib import Path
import numpy as np
from PIL import Image
from demo_dataset import create_demo
from replay import replay,load_dataset,unpack_dataset,ReplayCancelled
from server import Lab


class ReplayTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name);self.dataset=self.root/'input';self.m=create_demo(self.dataset)
    def tearDown(self):self.temp.cleanup()
    def write(self):
        (self.dataset/'manifest.json').write_text(json.dumps(self.m),encoding='utf-8')
    def run_case(self,roi=None,**kwargs):return replay(self.dataset,roi or self.m['suggested_roi'],self.root/'output',**kwargs)
    def test_large_static_box_only_and_repeated_view_not_support(self):
        r=self.run_case();self.assertEqual(r['frames_used'],3);self.assertEqual(len(r['regions']),1)
        self.assertEqual(r['regions'][0]['bbox_pixels'],[44,60,56,61]);self.assertEqual(r['frames'][-1]['status'],'SKIPPED_REPEATED_VIEW')
        self.assertGreater(r['candidate_cells'],r['accepted_cells']);self.assertIsNone(r['replacement_region'])
    def test_region_outside_roi_never_triggered(self):
        r=self.run_case([60,35,65,70]);self.assertEqual(r['regions'],[])
    def test_small_projected_change_filtered(self):
        r=self.run_case([44,58,46,64]);self.assertEqual(r['regions'],[])
        self.assertGreater(len(r['suppressed_regions']),0)
    def test_transient_changes_not_persistent(self):
        self.m['frames']=self.m['frames'][:2];self.write();r=self.run_case();self.assertGreater(r['candidate_cells'],0);self.assertEqual(r['regions'],[])
    def test_unknown_grid_not_treated_as_free(self):
        Image.new('L',(100,100),205).save(self.dataset/'map.png');r=self.run_case();self.assertEqual(r['candidate_cells'],0)
    def test_missing_depth_does_not_clear_old_occupancy(self):
        for f in self.m['frames']:np.save(self.dataset/f['depth'],np.zeros((120,160),np.float32))
        r=self.run_case();self.assertEqual(r['regions'],[]);self.assertFalse(r['capabilities']['removal_detection'])
    def test_masked_observations_do_not_become_map_changes(self):
        for f in self.m['frames']:
            f['mask']=f['id']+'.mask.png';Image.new('L',(160,120),0).save(self.dataset/f['mask'])
        self.write();r=self.run_case();self.assertEqual(r['candidate_cells'],0)
        self.assertFalse(r['capabilities']['removal_detection'])
    def test_bad_pose_skipped(self):
        for f in self.m['frames']:f['pose_valid']=False
        self.write();r=self.run_case();self.assertEqual(r['frames_used'],0)
    def test_undeclared_calibration_rejected(self):
        self.m['depth_rectified']=False;self.write()
        with self.assertRaises(ValueError):load_dataset(self.dataset)
    def test_nonrigid_pose_rejected(self):
        self.m['frames'][0]['pose_c2w'][0][0]=2;self.write()
        with self.assertRaises(ValueError):load_dataset(self.dataset)
    def test_paths_cannot_escape(self):
        self.m['frames'][0]['depth']='../outside.npy';self.write()
        with self.assertRaises(ValueError):load_dataset(self.dataset)
    def test_zip_traversal_rejected(self):
        b=io.BytesIO()
        with zipfile.ZipFile(b,'w') as z:z.writestr('../outside.json','{}')
        with self.assertRaises(ValueError):unpack_dataset(b.getvalue(),self.root/'unpack')
        self.assertFalse((self.root/'outside.json').exists())
    def test_cancelled_replay_not_published(self):
        with self.assertRaises(ReplayCancelled):self.run_case(cancelled=lambda:True)
        self.assertFalse((self.root/'output'/'report.json').exists())
    def test_server_mode_and_map_binding(self):
        lab=Lab(self.root/'lab','http://not-used');ds=lab.register_dataset()
        p=dict(dataset_id=ds['id'],map_id=ds['map_id'],roi=[40,35,65,70])
        with self.assertRaises(ValueError):lab.run_replay(p)
        lab.mode_change(True);result=lab.run_replay(p);self.assertTrue(result['synthetic']);self.assertEqual(len(result['regions']),1)
        blob=io.BytesIO();Image.new('L',(100,100),0).save(blob,format='PNG');other=lab.store_map(blob.getvalue(),self.m['map_metadata'],'other',dict(kind='uploaded'))
        p['map_id']=other['id']
        with self.assertRaisesRegex(ValueError,'不一致'):lab.run_replay(p)
        self.assertFalse(lab.replay_running)


if __name__=='__main__':unittest.main()
