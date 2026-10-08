import json,sys,tempfile,unittest,subprocess
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from automatic_update import execute,select_map,sha256
from cloud_server import Cloud
from automatic_source import complete_holdouts

class AutomaticUpdateTests(unittest.TestCase):
    def test_new_job_freezes_surface_backend_without_mutating_request(self):
        with tempfile.TemporaryDirectory() as tmp:
            cloud=Cloud(Path(tmp),'token',runner=lambda key:None)
            request={'iterations':100}
            key=cloud.create(request)['id']
            saved=json.loads((cloud.directory(key)/'request.json').read_text())
            self.assertEqual(saved['grid_backend'],'gaussian_surface')
            self.assertEqual(saved['fusion_policy'],'bounded_context_v2')
            self.assertNotIn('grid_backend',request)
            self.assertNotIn('fusion_policy',request)
            key=cloud.create(dict(iterations=100,grid_backend='rgbd'))['id']
            self.assertEqual(json.loads((cloud.directory(key)/'request.json').read_text())['grid_backend'],'rgbd')
            key=cloud.create(dict(iterations=100,fusion_policy='legacy_geometry'))['id']
            self.assertEqual(json.loads((cloud.directory(key)/'request.json').read_text())['fusion_policy'],'legacy_geometry')
            with self.assertRaises(ValueError):cloud.create(dict(fusion_policy='unreviewed'))

    def test_selected_source_does_not_alias_same_basename(self):
        from automatic_update import selected_source
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);source=p/'source';source.mkdir();package=p/'package';package.mkdir()
            for folder,content in [('rgb',b'color'),('mask',b'mask')]:
                (source/folder).mkdir();(source/folder/'000.png').write_bytes(content)
            (source/'map.png').write_bytes(b'map')
            frames=[dict(id=str(i),rgb='rgb/000.png',mask='mask/000.png') for i in range(26)]
            (source/'manifest.json').write_text(json.dumps(dict(frames=frames,map_image='map.png')))
            (package/'package.json').write_text(json.dumps(dict(frames=[dict(frame_id=str(i),split='train' if i<20 else 'eval') for i in range(26)])))
            target=p/'target';selected_source(source,package,target,[0,0,1,1])
            f=json.loads((target/'manifest.json').read_text())['frames'][0]
            self.assertNotEqual(f['rgb'],f['mask'])
            self.assertEqual((target/f['rgb']).read_bytes(),b'color')
            self.assertEqual((target/f['mask']).read_bytes(),b'mask')
            self.assertEqual((target/'map.png').read_bytes(),b'map')
            self.assertEqual(json.loads((target/'manifest.json').read_text())['map_image'],'map.png')

    def test_small_only_update_skips_training_at_existing_size_boundary(self):
        from automatic_update import preflight
        for size,expected in [(.39,False),(.4,True)]:
            with self.subTest(size=size),tempfile.TemporaryDirectory() as tmp:
                p=Path(tmp);out=p/'out';(out/'analysis').mkdir(parents=True);(out/'artifacts').mkdir();alignment=p/'align';alignment.mkdir()
                (out/'analysis/evidence.json').write_text(json.dumps(dict(proposals=[dict(id='region',oriented_box=dict(size=[size,.2,.1]))])))
                (out/'analysis/ai.json').write_text(json.dumps(dict(items=[dict(id='region',decision='update')])) )
                (alignment/'similarity_alignment.json').write_text(json.dumps(dict(metric_scale_conflict=False)))
                context=p/'context.json';context.write_text(json.dumps(dict(output=str(out),alignment=str(alignment))))
                cloud=Cloud(p/'jobs','token',runner=lambda key:None);key=cloud.create({'iterations':100})['id']
                with patch('automatic_update.initialize',return_value=(context,{'results_url':'http://example/results'})),patch('automatic_update.execute'),patch('context_rgbd.nominate',return_value=['region']):
                    self.assertEqual(preflight(cloud,key,p,p),expected)
                report=json.loads((out/'preflight.json').read_text())
                self.assertEqual(report['eligible_regions'],int(expected))

    def test_no_supported_region_distinguishes_failure_and_capture(self):
        from automatic_update import preflight
        for items,expected in [([dict(decision='defer',ai_validation_failed=True)],'SEMANTIC_UNAVAILABLE'),([dict(decision='defer')],'NEEDS_CAPTURE'),([dict(decision='ignore')],'NO_SUPPORTED_CHANGE')]:
            with self.subTest(expected=expected),tempfile.TemporaryDirectory() as tmp:
                p=Path(tmp);out=p/'out';(out/'analysis').mkdir(parents=True);(out/'artifacts').mkdir();alignment=p/'align';alignment.mkdir()
                (out/'analysis/evidence.json').write_text(json.dumps(dict(proposals=[])))
                (out/'analysis/ai.json').write_text(json.dumps(dict(items=items)))
                (alignment/'similarity_alignment.json').write_text(json.dumps(dict(metric_scale_conflict=False)))
                context=p/'context.json';context.write_text(json.dumps(dict(output=str(out),alignment=str(alignment))))
                cloud=Cloud(p/'jobs','token',runner=lambda key:None);key=cloud.create({'iterations':100})['id']
                with patch('automatic_update.initialize',return_value=(context,{'results_url':'http://example/results'})),patch('automatic_update.execute'),patch('context_rgbd.nominate',return_value=[]):
                    self.assertFalse(preflight(cloud,key,p,p))
                state=cloud.state(key);self.assertEqual(state['map_update']['state'],expected)
                self.assertNotIn('ply',state);self.assertFalse(state['map_update']['published'])

    def test_preflight_and_post_training_commands_are_separate(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);cfg=dict(output=str(p/'out'),alignment=str(p/'align'),source=str(p/'source'),base_map='original.ply',python='python',game_python='game')
            ctx=p/'context.json';ctx.write_text(json.dumps(cfg));early=[];late=[]
            execute(ctx,lambda stage:None,lambda cmd,**kw:early.append(cmd),phase='early',checkpointing=False)
            execute(ctx,lambda stage:None,lambda cmd,**kw:late.append(cmd),phase='late',checkpointing=False)
            self.assertFalse(any('--build-only' in c or 'segment' in c for c in early))
            self.assertTrue(any('retry-review' in c for c in early))
            self.assertFalse(any('update_matches.py' in ' '.join(c) for c in late))
            phases=[c[c.index('--phase')+1] for c in late if '--phase' in c]
            self.assertEqual(phases,['segment','fuse','evaluate','geometry-review'])
            grid=next(i for i,c in enumerate(late) if Path(c[1]).name=='grid_pipeline.py')
            report=next(i for i,c in enumerate(late) if Path(c[1]).name=='automatic_report.py')
            self.assertLess(grid,report)
            self.assertFalse(any(Path(c[1]).name=='grid_pipeline.py' for c in early))

    def test_retry_requires_saved_reconstruction_and_failed_update(self):
        with tempfile.TemporaryDirectory() as tmp:
            c=Cloud(tmp,'t'*40,runner=lambda key:None);key=c.create({'iterations':100})['id']
            c.update(key,state='FAILED')
            with self.assertRaises(ValueError):c.retry(key)
            c.update(key,state='READY',reconstruction_ready=True,map_update={'state':'NEEDS_REVIEW'})
            with self.assertRaises(ValueError):c.retry(key)
            c.update(key,map_update={'state':'FAILED'})
            self.assertEqual(c.retry(key)['state'],'QUEUED')

    def test_missing_sfm_holdout_replaced_before_training_without_duplication(self):
        entries=[dict(name=str(i),frame_id=str(i),split='eval' if i%5==0 else 'train') for i in range(40)]
        registered={str(i) for i in range(40)}-{'10','11','20'}
        rows,report=complete_holdouts(entries,registered,{str(i):i for i in range(40)})
        held={e['name'] for e in rows if e['split']=='eval'}
        self.assertEqual(len(held),8);self.assertTrue(held<=registered)
        self.assertEqual({r['missing_frame'] for r in report['replacements']},{'10','20'})
        self.assertEqual(entries[10]['split'],'eval') # preserve original input audit
        self.assertFalse(held & {e['name'] for e in rows if e['split']=='train'})
        with self.assertRaises(ValueError):complete_holdouts(entries,{'1','2'},{str(i):i for i in range(40)})

    def test_stage_failure_stops_before_fusion(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);cfg=dict(output=str(p/'out'),alignment=str(p/'align'),source=str(p/'source'),base_map='original.ply',python='python',game_python='game',backend_python='backend')
            ctx=p/'context.json';ctx.write_text(json.dumps(cfg));seen=[]
            def failing(command,**kw):
                if 'update_alignment.py' in ' '.join(command):raise subprocess.CalledProcessError(1,command)
            with self.assertRaises(subprocess.CalledProcessError):execute(ctx,seen.append,failing,checkpointing=False)
            self.assertEqual(seen,['MAP_LOCALIZING','MAP_MATCHING','MAP_ALIGNING'])
            self.assertFalse((p/'out/artifacts/scene.ply').exists())

    def test_order_and_protected_default_no_experiment_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);cfg=dict(output=str(p/'out'),alignment=str(p/'align'),source=str(p/'source'),base_map='original.ply',python='python',game_python='game',backend_python='backend')
            ctx=p/'context.json';ctx.write_text(json.dumps(cfg));calls=[];seen=[]
            execute(ctx,seen.append,lambda cmd,**kw:calls.append(cmd),checkpointing=False)
            self.assertEqual(seen[-1],'MAP_REPORT')
            fusion=next(c for c in calls if '--build-only' in c);self.assertEqual(fusion[fusion.index('--fusion-mode')+1],'geometry')
            self.assertEqual(fusion[0],cfg['python']) # renderer imports VPS/hloc
            self.assertEqual(fusion[fusion.index('--source')+1],cfg['source'])
            self.assertNotIn('masked_boxer_complete',' '.join(sum(calls,[])))

    def test_failed_update_preserves_reconstruction_and_is_visible(self):
        with tempfile.TemporaryDirectory() as tmp:
            c=Cloud(tmp,'t'*40,runner=lambda key:None);key=c.create({'iterations':100})['id']
            result=dict(map_aligned=False,coordinate_frame='camera_local',metrics={},ply='training/splat.ply')
            with patch('automatic_update.run',side_effect=ValueError('alignment rejected')):
                c.finish_reconstruction(key,Path(tmp)/'source',result)
            s=c.state(key)
            self.assertEqual(s['state'],'READY');self.assertTrue(s['reconstruction_ready'])
            self.assertEqual(s['map_update']['state'],'FAILED');self.assertIn('alignment rejected',s['map_update']['error'])
            self.assertFalse(s['quality_accepted']);self.assertNotIn('candidate_url',s['map_update'])

    def test_completed_update_still_needs_review(self):
        with tempfile.TemporaryDirectory() as tmp:
            c=Cloud(tmp,'t'*40,runner=lambda key:None);key=c.create({'iterations':100})['id']
            result=dict(map_aligned=False,coordinate_frame='camera_local',metrics={},ply='training/splat.ply')
            with patch('automatic_update.run',return_value=dict(state='NEEDS_REVIEW',result_url='http://example/result',published=False)):
                c.finish_reconstruction(key,Path(tmp)/'source',result)
            self.assertEqual(c.state(key)['map_update']['state'],'NEEDS_REVIEW')
            self.assertFalse(c.state(key)['quality_accepted'])

    def test_map_identity_and_version_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'base';p.write_bytes(b'old');entry=dict(map_fingerprint='current',base_map=str(p),base_sha256=sha256(p))
            with patch('replay.read_dataset',return_value=({},None,{})),patch('replay.fingerprint',return_value='current'):
                self.assertEqual(select_map(tmp,dict(maps=[entry])),entry)
                p.write_bytes(b'new')
                with self.assertRaises(ValueError):select_map(tmp,dict(maps=[entry]))
                with self.assertRaises(ValueError):select_map(tmp,dict(maps=[]))

if __name__=='__main__':unittest.main()
