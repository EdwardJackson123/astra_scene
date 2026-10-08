import sys,tempfile,json,unittest
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from masks import read_mask,find_mask
from tracking_demo import create_tracking_demo
from trajectory import track
from collection import export_spirula


class MaskTests(unittest.TestCase):
    def test_binary_semantics_and_bad_shape(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);Image.fromarray(np.array([[0,255],[255,0]],np.uint8)).save(p/'x.png')
            self.assertEqual(read_mask(p/'x.png',(2,2)).tolist(),[[False,True],[True,False]])
            self.assertEqual(find_mask(p,'x.jpg'),p/'x.png')
            with self.assertRaises(ValueError):read_mask(p/'x.png',(3,2))
            Image.fromarray(np.full((2,2),128,np.uint8)).save(p/'x.png')
            with self.assertRaises(ValueError):read_mask(p/'x.png',(2,2))
    def test_person_pixels_do_not_penalize_static_metrics(self):
        from masked_metrics import appearance
        gt=np.full((80,80,3),100,np.uint8);pred=gt.copy();pred[25:55,25:55]=255
        keep=np.ones((80,80),bool);keep[25:55,25:55]=False
        result=appearance(pred,gt,keep)
        self.assertAlmostEqual(result['static']['rgb_l1'],0)
        self.assertAlmostEqual(result['static']['ssim'],1)
        self.assertGreater(result['full']['rgb_l1'],0)

    def test_masks_export_and_remove_initial_points(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);create_tracking_demo(p/'raw',p/'ref');m=track(p/'raw',p/'local')
            original=export_spirula(p/'local',p/'original',[1,1,2,2],[0,0,3,3])
            for f in m['frames']:
                with Image.open(p/'local'/f['rgb']) as image:w,h=image.size
                a=np.full((h,w),255,np.uint8);a[:,:w//2]=0
                f['mask']=f['id']+'.mask.png';Image.fromarray(a).save(p/'local'/f['mask'])
            (p/'local/manifest.json').write_text(json.dumps(m))
            result=export_spirula(p/'local',p/'masked',[1,1,2,2],[0,0,3,3])
            self.assertEqual(result['mask_policy'],'ignore_black_observations')
            self.assertLess(result['initial_points'],original['initial_points']);self.assertGreater(result['initial_points'],20)
            self.assertEqual(len(list((p/'masked/dataset/masks').glob('*.png'))),8)


if __name__=='__main__':unittest.main()
