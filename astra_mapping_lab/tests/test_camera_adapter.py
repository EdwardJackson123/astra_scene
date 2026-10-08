import copy,sys,unittest
from pathlib import Path
from types import SimpleNamespace as NS
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from recording import validate_capture_config,require_capture_ready,static_camera_info

def config():
    return dict(pose_source='rgbd_odometry',intrinsics_source='static',rgb_topic='/camera/captured_image',depth_topic='/camera/captured_depth',intrinsics=dict(verified=True,serial='test',width=640,height=480,K=[[400,0,320],[0,400,240],[0,0,1]],D=[0]*5,distortion_model='plumb_bob',rgb_frame_id='camera_test',depth_frame_id='camera_test_depth',depth_aligned_to_color=True,provenance='synthetic unit test'))

def message(frame):return NS(width=640,height=480,header=NS(frame_id=frame,stamp=NS(sec=1,nanosec=2)))

class AdapterTests(unittest.TestCase):
    def test_recording_limits_are_opt_in(self):
        c=config();v=validate_capture_config(c)
        self.assertEqual((v['max_frames'],v['max_duration_s']),(0,0))
        c.update(max_frames=1000,max_duration_s=1800)
        self.assertEqual(validate_capture_config(c)['max_frames'],1000)
        for bad in [-1,True,1.5]:
            c['max_frames']=bad
            with self.assertRaises(ValueError):validate_capture_config(c)
    def test_capture_rate_validation(self):
        self.assertEqual(validate_capture_config(config())['capture_hz'],5)
        for hz in [0,11,float('nan'),True]:
            c=config();c['capture_hz']=hz
            with self.assertRaises(ValueError):validate_capture_config(c)
    def test_exact_camera_and_stamp(self):
        c=config();a=message('camera_test');b=message('camera_test_depth')
        self.assertEqual(static_camera_info(c,a,b).width,640)
        for field,value in [('frame_id','camera_other'),('stamp',NS(sec=1,nanosec=3))]:
            bad=copy.deepcopy(b);setattr(bad.header,field,value)
            with self.assertRaises(ValueError):static_camera_info(c,a,bad)
        b.width=320
        with self.assertRaises(ValueError):static_camera_info(c,a,b)

    def test_unverified_saved_but_not_recordable(self):
        c=config();c['intrinsics']['verified']=False
        self.assertFalse(validate_capture_config(c)['intrinsics']['verified'])
        with self.assertRaises(ValueError):require_capture_ready(c)

    def test_bad_geometry_rejected(self):
        for key,value in [('K',[[0,0,0]]*3),('D',[float('nan')]*5),('depth_aligned_to_color',False),('serial','other')]:
            c=config();c['intrinsics'][key]=value
            with self.assertRaises(ValueError):validate_capture_config(c)

    def test_missing_distortion_never_assumed_zero(self):
        c=config();c['intrinsics'].update(D=None,verified=False)
        self.assertIsNone(validate_capture_config(c)['intrinsics']['D'])
        with self.assertRaises(ValueError):require_capture_ready(c)
        c['intrinsics']['verified']=True
        with self.assertRaises(ValueError):validate_capture_config(c)

    def test_standard_camera_info_retained(self):
        c=dict(pose_source='rgbd_odometry',rgb_topic='/rgb',depth_topic='/depth',info_topic='/info')
        self.assertEqual(require_capture_ready(c)['intrinsics_source'],'camera_info')
