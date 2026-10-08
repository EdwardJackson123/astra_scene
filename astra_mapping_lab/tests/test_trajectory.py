import json,sys,tempfile,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from recording import RawRecorder
from trajectory import fit_rigid,robust_rigid,track,align_anchors,reference_anchors,raw_manifest
from tracking_demo import create_tracking_demo
from replay import load_dataset,fingerprint
from collection import export_spirula
from PIL import Image


class TrajectoryTests(unittest.TestCase):
    def setUp(self):self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
    def tearDown(self):self.tmp.cleanup()
    def test_connection_requires_distinct_frames_on_both_sides(self):
        from trajectory import connection_consensus
        T=np.eye(4)
        self.assertIsNone(connection_consensus([(0,5,T),(0,6,T),(0,7,T)]))
        bad=T.copy();bad[0,3]=1
        self.assertIsNone(connection_consensus([(0,5,T),(1,6,T),(2,7,bad)]))
        self.assertIsNotNone(connection_consensus([(0,5,T),(1,6,T),(2,7,T)]))

    def test_incomplete_trajectory_cannot_train(self):
        from trajectory import trajectory_quality,require_trajectory_quality
        frames=[dict(timestamp_s=i*.25) for i in range(487)]
        q=trajectory_quality(frames,list(range(34)),[34,453])
        self.assertFalse(q['accepted'])
        with self.assertRaises(ValueError):require_trajectory_quality(dict(kind='posed_rgbd',trajectory_report=dict(quality_gate=q)))
        self.assertTrue(trajectory_quality(frames,list(range(487)),[487])['accepted'])

    def test_partial_but_connected_trajectory_can_build(self):
        from trajectory import require_trajectory_quality
        frames=[dict(id=str(i),pose_c2w=np.eye(4).tolist()) for i in range(8)]
        q=dict(accepted=False,edge_checks=12,inconsistent_edge_fraction=.01)
        m=dict(kind='posed_rgbd',frames=frames,trajectory_report=dict(quality_gate=q,components=[dict(frames=[f['id'] for f in frames])]))
        require_trajectory_quality(m)
        q['inconsistent_edge_fraction']=.3
        with self.assertRaises(ValueError):require_trajectory_quality(m)
        q['inconsistent_edge_fraction']=.01;m['trajectory_report']['components']=[dict(frames=['0','1']),dict(frames=['2','3','4','5','6','7'])]
        with self.assertRaises(ValueError):require_trajectory_quality(m)

    def test_keyframes_reduce_repeats_and_keep_independent_holdouts(self):
        from keyframes import select
        from unittest.mock import patch
        rng=np.random.default_rng(1);im=Image.fromarray(rng.integers(0,256,(30,30),dtype=np.uint8))
        frames=[dict(id=str(i),pose_c2w=np.eye(4).tolist(),timestamp_s=i*.25,rgb='x',depth='y') for i in range(100)]
        with patch('keyframes.asset',return_value='unused'),patch('keyframes.Image.open',return_value=im),patch('keyframes.load_depth',return_value=np.ones((30,30))):
            selected,report=select(self.root,frames)
        train=[f for f,split in selected if split=='train'];holdout=[f for f,split in selected if split=='eval']
        self.assertLess(len(train),30);self.assertEqual(len(holdout),12)
        self.assertFalse({f['id'] for f in train}&{f['id'] for f in holdout})
        self.assertEqual(train[0]['id'],'0');self.assertEqual(train[-1]['id'],'99')
        self.assertLessEqual(max(b['timestamp_s']-a['timestamp_s'] for a,b in zip(train,train[1:])),1.5)

    def test_failed_gate_prevents_package_creation(self):
        from unittest.mock import patch
        from collection import export_spirula
        m=dict(kind='posed_rgbd',world_convention='camera_local',trajectory_report=dict(quality_gate=dict(accepted=False,reasons=['incomplete'])))
        with patch('collection.load_dataset',return_value=(m,None,{})):
            with self.assertRaises(ValueError):export_spirula('unused',self.root/'package',[],[])
        self.assertFalse((self.root/'package').exists())

    def test_segment_recovery_and_disconnected_frames_remain_separate(self):
        from unittest.mock import patch
        from trajectory import track
        raw=self.root/'raw';raw.mkdir();Image.new('L',(10,10),255).save(raw/'map.png')
        frames=[]
        for i in range(10):
            (raw/f'{i}.jpg').write_bytes(b'image');(raw/f'{i}.npy').write_bytes(b'depth')
            frames.append(dict(id=str(i),timestamp_s=i*.25,rgb=f'{i}.jpg',depth=f'{i}.npy'))
        (raw/'capture_manifest.json').write_text(json.dumps(dict(kind='raw_rgbd',depth_aligned_to='color',depth_rectified=True,depth_unit='m',map_image='map.png',frames=frames)))
        def feature(root,f):return np.full((30,3),int(f['id'])),None
        def match(a,b):
            i,j=int(a[0][0,0]),int(b[0][0,0])
            # Adjacent loss at 3->4; three independent bridges can recover it.
            if j>=8 or (i,j)==(3,4):raise ValueError('no match')
            return np.array([i]),np.array([j])
        def estimate(a,b):
            T=np.eye(4);T[0,3]=-.05*(int(b[0])-int(a[0]));return T,dict(inliers=30)
        with patch('trajectory.features',side_effect=feature),patch('trajectory.correspond',side_effect=match),patch('trajectory.robust_rigid',side_effect=estimate):
            m=track(raw,self.root/'tracked')
        self.assertEqual(len(m['frames']),8)
        self.assertAlmostEqual(m['frames'][-1]['pose_c2w'][0][3],.35)
        self.assertEqual(len(m['trajectory_report']['connections']),1)
        self.assertFalse(m['trajectory_report']['quality_gate']['accepted'])

    def test_ransac_rejects_outliers_without_scale_change(self):
        rng=np.random.default_rng(2);a=rng.normal(size=(100,3));R=np.array([[0,-1,0],[1,0,0],[0,0,1]]);b=a@R.T+[.3,.1,-.2];b[:20]=rng.normal(size=(20,3))*4
        T,r=robust_rigid(a,b);self.assertTrue(np.allclose(T[:3,:3],R));self.assertTrue(np.allclose(T[:3,3],[.3,.1,-.2]));self.assertEqual(r['inliers'],80)
    def test_collinear_geometry_rejected(self):
        a=np.zeros((30,3));a[:,0]=np.arange(30)
        with self.assertRaises(ValueError):robust_rigid(a,a)
    def test_raw_recording_needs_no_body_pose(self):
        m,_=create_tracking_demo(self.root/'raw',self.root/'ref');record=dict(metadata=m['map_metadata'],source={})
        cfg=dict(pose_source='rgbd_odometry',rgb_topic='/rgb',depth_topic='/depth',info_topic='/info')
        r=RawRecorder(self.root/'record',record,self.root/'raw/map.png',m['suggested_roi'],cfg)
        f=m['frames'][0];rgb=np.asarray(Image.open(self.root/'raw'/f['rgb']));d=np.load(self.root/'raw'/f['depth'])
        self.assertTrue(r.add(rgb,d,f['K'],1));r.stop();self.assertNotIn('pose_c2w',r.frames[0]);self.assertFalse(r.manifest['map_alignment']['accepted'])
    def test_long_recording_can_be_saved_and_loaded(self):
        image=self.root/'map.png';Image.new('L',(10,10),255).save(image)
        cfg=dict(pose_source='rgbd_odometry',rgb_topic='/rgb',depth_topic='/depth',info_topic='/info')
        r=RawRecorder(self.root/'record',dict(metadata={'resolution':.1,'origin':[0,0,0]},source={}),image,[0,0,5,5],cfg)
        rgb=np.full((8,8,3),100,np.uint8);d=np.ones((8,8),np.float32);K=[[10,0,4],[0,10,4],[0,0,1]]
        for i in range(205):self.assertTrue(r.add(rgb,d,K,1+i*.5))
        r.stop();self.assertEqual(len(raw_manifest(r.root)['frames']),205)
        r2=RawRecorder(self.root/'limited',dict(metadata={},source={}),image,[0,0,5,5],{**cfg,'max_frames':2})
        self.assertTrue(r2.add(rgb,d,K,1));self.assertTrue(r2.add(rgb,d,K,2));self.assertFalse(r2.add(rgb,d,K,3));self.assertEqual(r2.state,'STOPPED')
    def test_five_hz_capture_and_preview_isolation(self):
        from server import Lab
        lab=Lab(self.root/'lab','http://unused');lab.mode_change(True)
        key='a'*32;image=self.root/'map.png';Image.new('L',(10,10),255).save(image)
        cfg=dict(pose_source='rgbd_odometry',rgb_topic='/rgb',depth_topic='/depth',info_topic='/info')
        r=RawRecorder(lab.data/'datasets'/key,dict(metadata={},source={}),image,[0,0,5,5],cfg)
        from recording import atomic
        atomic(lab.data/'captures'/(key+'.json'),dict(id=key,state='ACTIVE'))
        lab.workflow.capture=key
        with self.assertRaises(ValueError):lab.workflow.preview(key)
        rgb=np.full((8,8,3),100,np.uint8);d=np.ones((8,8),np.float32);K=[[10,0,4],[0,10,4],[0,0,1]]
        for i in range(10):self.assertTrue(r.add(rgb,d,K,1+i*.2))
        self.assertFalse(r.add(rgb,d,K,2.85));self.assertFalse(r.counts)
        self.assertEqual(lab.workflow.preview(key),(r.root/'000009.jpg').read_bytes())
        self.assertIsNotNone(lab.workflow.capture_state()['frame_age_s'])
        with self.assertRaises(ValueError):lab.workflow.preview('b'*32)
        lab.mode_change(False)
        with self.assertRaises(ValueError):lab.workflow.preview(key)
    def test_complete_no_extrinsic_path(self):
        create_tracking_demo(self.root/'raw',self.root/'ref');local=track(self.root/'raw',self.root/'local')
        self.assertEqual(len(local['frames']),8);self.assertAlmostEqual(local['frames'][-1]['pose_c2w'][0][3],.56,delta=.015)
        with self.assertRaises(ValueError):load_dataset(self.root/'local')
        p=export_spirula(self.root/'local',self.root/'package',[48,48,52,52],[45,45,55,55]);self.assertFalse(p['map_aligned']);self.assertGreater(p['initial_points'],20)
        anchors=reference_anchors(self.root/'local',self.root/'ref');self.assertGreaterEqual(len(anchors['anchors']),4)
        aligned=align_anchors(self.root/'local',self.root/'aligned',anchors);self.assertTrue(aligned['map_alignment']['accepted']);load_dataset(self.root/'aligned')
        expected=np.array([[1,0,0,0],[0,0,1,0],[0,-1,0,1],[0,0,0,1]])
        self.assertTrue(np.allclose(aligned['map_alignment']['map_from_local'],expected,atol=.015))
    def test_alignment_wrong_map_and_holdout_rejected(self):
        create_tracking_demo(self.root/'raw',self.root/'ref');track(self.root/'raw',self.root/'local');spec=reference_anchors(self.root/'local',self.root/'ref')
        wrong={**spec,'map_fingerprint':'bad'}
        with self.assertRaises(ValueError):align_anchors(self.root/'local',self.root/'bad',wrong)
        spec['anchors'][-1]['pose_c2w'][0][3]+=1
        with self.assertRaises(ValueError):align_anchors(self.root/'local',self.root/'bad',spec)
        self.assertFalse((self.root/'bad').exists())
    def test_cancel_before_tracking_publishes_nothing(self):
        create_tracking_demo(self.root/'raw',self.root/'ref')
        with self.assertRaises(ValueError):track(self.root/'raw',self.root/'local',lambda:True)
        self.assertFalse((self.root/'local/manifest.json').exists())


if __name__=='__main__':unittest.main()
