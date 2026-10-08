import json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from sfm_rgbd import convert
from keyframes import select


class SfMDepthTests(unittest.TestCase):
    def exercise(self,bad_eval=False,mask_dynamic=False):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder);source=p/'source';source.mkdir();package=p/'package';package.mkdir()
            Image.new('L',(40,40),255).save(source/'map.png');Image.new('RGB',(40,40)).save(source/'rgb.jpg')
            K=[[100,0,20],[0,100,20],[0,0,1]];frames=[];splits=[];images=[];points={};obs=[]
            for y in range(4,35,3):
                for x in range(4,35,3):
                    key=len(points);points[key]=dict(xyz=np.array([(x-20)/100,(y-20)/100,1]),error=0);obs.append((x,y,key))
            maskdir=p/'masks';maskdir.mkdir()
            for i in range(8):
                np.save(source/f'{i}.npy',np.full((40,40),5 if bad_eval and i>=6 else 3,np.float32))
                if mask_dynamic:
                    depth=np.full((40,40),3,np.float32);depth[:,:18]=6;np.save(source/f'{i}.npy',depth)
                    keep=np.full((40,40),255,np.uint8);keep[:,:18]=0;Image.fromarray(keep).save(maskdir/f'{i}.png')
                frames.append(dict(id=str(i),rgb='rgb.jpg',depth=f'{i}.npy',K=K,timestamp_s=i,pose_valid=True))
                splits.append(dict(frame_id=str(i),name=f'{i}.png',split='train' if i<6 else 'eval'))
                images.append(dict(name=f'{i}.png',camera_id=1,w2c=np.eye(4),observations=obs))
            (source/'manifest.json').write_text(json.dumps(dict(kind='posed_rgbd',frames=frames,map_image='map.png')))
            (package/'package.json').write_text(json.dumps(dict(frames=splits)))
            model=({1:dict(K=K,width=40,height=40)},images,points)
            with patch('sfm_rgbd.read_model',return_value=model):
                if bad_eval:
                    with self.assertRaises(ValueError):convert(source,package,'unused',p/'out')
                    r=json.loads((p/'out/sfm_rgbd_report.json').read_text(encoding='utf-8'))
                    self.assertAlmostEqual(r['scale_m_per_sfm_unit'],3)
                    self.assertFalse((p/'out/manifest.json').exists())
                else:
                    m=convert(source,package,'unused',p/'out',mask_folder=maskdir if mask_dynamic else None);r=m['sfm_rgbd_report']
                    self.assertTrue(r['accepted']);self.assertAlmostEqual(r['scale_m_per_sfm_unit'],3)
                    self.assertAlmostEqual(r['heldout_depth']['median_error_m'],0)
                    selected,_=select(p/'out',m['frames'])
                    self.assertEqual([s for _,s in selected],['train']*6+['eval']*2)
    def test_metric_scale_and_preserved_splits(self):self.exercise()
    def test_masked_dynamic_depth_does_not_bias_scale(self):self.exercise(mask_dynamic=True)
    def test_eval_depth_never_changes_training_scale(self):self.exercise(True)


if __name__=='__main__':unittest.main()
