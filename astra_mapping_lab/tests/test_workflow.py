import json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import Mock
import numpy as np
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from core import normalize_metadata,make_plan
from collection import collection_route,export_spirula,rotation_quaternion
from demo_dataset import create_demo
from recording import Recorder,validate_calibration
from server import Lab


def calibration():return dict(verified=True,camera_to_base=np.eye(4).tolist(),ground_z_m=0,depth_aligned_to_color=True,rgb_topic='/rgb',depth_topic='/depth',info_topic='/info',pose_topic='/pose')


class WorkflowTests(unittest.TestCase):
    def setUp(self):self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
    def tearDown(self):self.tmp.cleanup()
    def test_route_cannot_cross_wall(self):
        pixels=np.full((100,100),255,np.uint8);pixels[:,49:52]=0;image=Image.fromarray(pixels);meta=normalize_metadata(dict(resolution=.1,origin=[0,0,0]),100,100)
        plan=dict(preview_clearance_m=.1,viewpoints=[dict(id='left',pixel=[20.5,20.5],yaw=0),dict(id='right',pixel=[80.5,20.5],yaw=0)])
        r=collection_route(image,meta,plan,[2,7]);self.assertEqual([s['view_id'] for s in r['segments']],['left']);self.assertEqual(r['unreachable'],['right']);self.assertFalse(r['execution_allowed'])
        for segment in r['segments']:
            for x,y in segment['pixels']:self.assertEqual(pixels[int(y),int(x)],255)
    def test_route_rejects_invalid_start(self):
        image=Image.new('L',(20,20),0);meta=normalize_metadata(dict(resolution=.1,origin=[0,0,0]),20,20)
        with self.assertRaises(ValueError):collection_route(image,meta,dict(preview_clearance_m=.1,viewpoints=[]),[1,1])
    def test_colmap_pose_and_holdout(self):
        d=self.root/'demo';m=create_demo(d);target=self.root/'package';r=export_spirula(d,target,m['suggested_roi'],[20,20,80,90]);self.assertTrue((target/'reconstruction.zip').is_file());self.assertGreater(r['initial_points'],20)
        lines=(target/'dataset/sparse/0/images.txt').read_text().splitlines();head=lines[0].split();q=np.array(list(map(float,head[1:5])));w,x,y,z=q
        R=np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
        pose=np.array(m['frames'][0]['pose_c2w']);self.assertTrue(np.allclose(R,pose[:3,:3].T));self.assertTrue(np.allclose(np.array(list(map(float,head[5:8]))),-R@pose[:3,3]))
        self.assertEqual(lines[-1],'');self.assertEqual(r['frames'][-1]['split'],'eval');self.assertFalse(r['quality_accepted'])
    def test_unverified_calibration_rejected(self):
        c=calibration();c['verified']=False
        with self.assertRaises(ValueError):validate_calibration(c)
        c['verified']=True;c['camera_to_base'][0][0]=2
        with self.assertRaises(ValueError):validate_calibration(c)
    def recorder(self):
        im=self.root/'map.png';Image.new('L',(100,100),255).save(im)
        return Recorder(self.root/'record',dict(metadata=normalize_metadata(dict(resolution=.1,origin=[0,0,0]),100,100),source={}),im,[10,10,50,50],calibration())
    def test_quality_repeat_and_stop(self):
        r=self.recorder();rgb=np.full((24,32,3),100,np.uint8);d=np.ones((24,32));K=np.array([[20,0,16],[0,20,12],[0,0,1]])
        self.assertFalse(r.add(rgb,d*0,K,np.eye(4),1));self.assertTrue(r.add(rgb,d,K,np.eye(4),1));self.assertFalse(r.add(rgb,d,K,np.eye(4),2))
        pose=np.eye(4);pose[0,3]=.3;self.assertTrue(r.add(rgb,d,K,pose,2));r.stop();pose[0,3]=.6;self.assertFalse(r.add(rgb,d,K,pose,3));self.assertEqual(len(r.frames),2)
    def test_mode_exit_requests_only_owned_capture_stop(self):
        lab=Lab(self.root/'lab','http://127.0.0.1:9');lab.mode_change(True);lab.workflow.capture='a'*32
        # Model owned session record so status remains inspectable.
        (lab.data/'captures'/('a'*32+'.json')).write_text(json.dumps(dict(state='ACTIVE')))
        lab.mode_change(False);self.assertTrue((lab.data/'datasets'/('a'*32+'.stop')).exists());self.assertEqual(lab.mode,'OFF')
    def test_recording_off_rejected_before_process(self):
        lab=Lab(self.root/'lab','http://127.0.0.1:9')
        with self.assertRaises(ValueError):lab.workflow.start({})
        self.assertIsNone(lab.workflow.process)
    def test_reconstruction_package_map_binding(self):
        lab=Lab(self.root/'lab','http://127.0.0.1:9');lab.mode_change(True);d=lab.register_dataset();task=lab.plan(dict(map_id=d['map_id'],roi=d['suggested_roi'],task_type='inspect'))
        p=lab.workflow.package(dict(task_id=task['id'],dataset_id=d['id']));self.assertTrue(p['synthetic']);self.assertEqual(lab.workflow.job(p['id'])['state'],'PACKAGE_READY')
        lab.mode_change(False)
        with self.assertRaises(ValueError):lab.workflow.reconstruct(dict(id=p['id']))
    def test_package_path_rejects_traversal(self):
        lab=Lab(self.root/'lab','http://127.0.0.1:9')
        with self.assertRaises(ValueError):lab.workflow.package_path('../outside')


if __name__=='__main__':unittest.main()
