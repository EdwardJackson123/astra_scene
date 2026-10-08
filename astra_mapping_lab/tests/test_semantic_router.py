import json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from semantic_router import route,SemanticClient,validate

class RouterTests(unittest.TestCase):
    def setUp(self):
        self.p=dict(id='r',oriented_box={'size':[1,1,1]},images=[{'view':1},{'view':4}],source_label='chair')
        self.g=dict(observed_fraction=.95,stable_fraction=.95,changed_fraction=0,current_supported_fraction=1,evidence_views=[1,4,7,9])
        self.answer=dict(id='r',decision='update',scope_complete=True,supporting_views=[1,4],change_type='layout',reason='visible',missing_parts=[])
    def test_unknown_depth_does_not_mean_unchanged(self):
        self.assertIsNone(route(self.p,dict(self.g,observed_fraction=.2,stable_fraction=.2)))
        self.assertEqual(route(self.p,self.g)['decision'],'ignore')
    def test_clear_change_still_needs_completion(self):
        r=route(self.p,dict(self.g,stable_fraction=0,changed_fraction=.9))
        self.assertEqual(r['decision'],'defer');self.assertFalse(r['scope_complete'])
        self.assertIsNone(route(self.p,dict(self.g,stable_fraction=0,changed_fraction=.9,current_supported_fraction=.2)))
    def test_small_and_protected_regions_skip_model(self):
        self.assertEqual(route(dict(self.p,oriented_box={'size':[.1,.1,.1]}),self.g)['decision_source'],'geometry_small_item')
        self.assertEqual(route(dict(self.p,source_label='window'),self.g)['decision_source'],'geometry_protected')
    def test_cache_changes_with_image_and_stage(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);image=p/'image.jpg';image.write_bytes(b'first');calls=[]
            def transport(payload):
                calls.append(payload);return {'choices':[{'message':{'content':json.dumps({'items':[self.answer]})}}]}
            c=SemanticClient(p,p,transport);images=[(self.p['images'][0],image),(self.p['images'][1],image)]
            a,r=c.call(self.p,images,{});self.assertFalse(r['cache_hit'])
            a,r=c.call(self.p,images,{});self.assertTrue(r['cache_hit']);self.assertEqual(len(calls),1)
            image.write_bytes(b'changed');c.call(self.p,images,{});self.assertEqual(len(calls),2)
            c.call(self.p,images,{},'completion');self.assertEqual(len(calls),3)
    def test_invalid_response_is_not_cached_or_authorized(self):
        with tempfile.TemporaryDirectory() as d:
            calls=[]
            def transport(payload):
                calls.append(1);return {'choices':[{'message':{'content':json.dumps({'items':[dict(self.answer,supporting_views=[1,999])]})}}]}
            c=SemanticClient(d,d,transport);item,audit=c.call(self.p,[],{})
            self.assertEqual(len(calls),2);self.assertTrue(item['ai_validation_failed']);self.assertEqual(item['decision'],'defer')
            self.assertEqual(list(c.cache.glob('*.json')),[])
    def test_complete_answer_cannot_have_missing_parts(self):
        with self.assertRaises(ValueError):validate(dict(self.answer,missing_parts=['base']),self.p)

    def test_missing_parts_do_not_veto_supported_local_update(self):
        from semantic_router import completed_regions
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'artifacts').mkdir();ctx=(root,None,None,root,None,{},None,None,None,None)
            p=dict(self.p,evidence_side='current');item=dict(self.answer,decision='defer',scope_complete=False,missing_parts=['top'])
            report=dict(id='r',mask_views=3,seeded_gaussians=100,contradicted_fraction=0,search_boundary_reached=True)
            with patch('semantic_router.SemanticClient.call',side_effect=AssertionError('no semantic conflict')):
                self.assertEqual(completed_regions(ctx,[p],[item],{'r':[1,2]},[report]),{'r'})
            record=json.loads((root/'artifacts/completion_check.json').read_text(encoding='utf-8'))['regions'][0]
            self.assertEqual(record['status'],'bounded_update');self.assertFalse(record['scope_complete'])
    def test_uncertain_depth_conflict_retains_region(self):
        from semantic_router import completed_regions,fallback
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'artifacts').mkdir();ctx=(root,None,None,root,None,{'depth_multiplier':1},None,None,None,None)
            p=dict(self.p,evidence_side='current',images=[]);report=dict(id='r',mask_views=3,seeded_gaussians=100,contradicted_fraction=.4)
            with patch('semantic_router.SemanticClient.call',return_value=(fallback(p,'uncertain'),{})) as call:
                self.assertEqual(completed_regions(ctx,[p],[self.answer],{'r':[1,2]},[report]),set());self.assertEqual(call.call_count,1)

    def test_local_quality_guard_rejects_repeated_foreground_regression(self):
        import numpy as np
        from semantic_router import guard_regions
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);old=root/'old';new=root/'new';old.mkdir();new.mkdir();(root/'artifacts').mkdir()
            mask=np.ones((24,24),bool);np.savez(root/'boxer_sam_masks.npz',r__0=mask,r__1=mask)
            rows=[dict(valid=mask,depth=np.ones((24,24))*2,rgb=np.ones((24,24,3))*100/255) for _ in range(2)]
            for i in range(2):
                np.savez(old/f'{i:03d}.npz',depth=rows[i]['depth'],rgb=np.ones((24,24,3),dtype='uint8')*100,alpha=np.ones((24,24)))
                np.savez(new/f'{i:03d}.npz',depth=rows[i]['depth']*.5,rgb=np.ones((24,24,3),dtype='uint8')*255,alpha=np.ones((24,24)))
            ctx=(root,None,None,root,None,None,None,rows,[0,1],[])
            self.assertEqual(guard_regions(ctx,[{'id':'r'}],old,new),[])
            self.assertEqual(guard_regions(ctx,[{'id':'r'}],old,old),[{'id':'r'}])
            with self.assertRaises(ValueError):guard_regions(ctx[:-1]+([1],),[{'id':'r'}],old,new)
