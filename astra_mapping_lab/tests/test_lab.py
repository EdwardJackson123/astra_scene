import io,json,math,tempfile,threading,unittest,sys
from pathlib import Path
from urllib.request import Request,urlopen
from urllib.error import HTTPError
from http.server import ThreadingHTTPServer
from unittest.mock import Mock
import numpy as np
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from core import pixel_to_world,world_to_pixel,normalize_metadata,make_plan,clearance_mask
from server import Lab,handler


def raw_image(value=255):
    image=Image.new('L',(100,80),value);b=io.BytesIO();image.save(b,format='PNG');return b.getvalue()


class GeometryTests(unittest.TestCase):
    def setUp(self):self.meta=normalize_metadata(dict(resolution=.1,origin=[2,-3,math.pi/2]),100,80)
    def test_rotated_coordinate_roundtrip_and_y_axis(self):
        self.assertTrue(np.allclose(pixel_to_world(0,80,self.meta),[2,-3]))
        self.assertTrue(np.allclose(pixel_to_world(10,80,self.meta),[2,-2]))
        for x,y in [(0,0),(.5,.5),(37.25,61.5),(100,80)]:
            self.assertTrue(np.allclose(world_to_pixel(*pixel_to_world(x,y,self.meta),self.meta),[x,y]))
    def test_occupied_unknown_transparent_cannot_be_viewpoints(self):
        for color in [(0,0,0,255),(205,205,205,255),(255,255,255,0)]:
            p=make_plan(Image.new('RGBA',(100,80),color),self.meta,[30,25,60,50])
            self.assertEqual(p['viewpoints'],[]);self.assertIsNone(p['replacement_region'])
    def test_candidates_outside_region_and_no_route(self):
        p=make_plan(Image.new('L',(100,80),255),self.meta,[30,25,60,50])
        self.assertGreater(len(p['viewpoints']),0);self.assertIsNone(p['route']);self.assertFalse(p['execution_allowed'])
        for q in p['viewpoints']:
            x,y=q['pixel'];self.assertFalse(30<=x<=60 and 25<=y<=50)
            self.assertTrue(np.allclose(q['world_xy'],pixel_to_world(x,y,self.meta)))
    def test_invalid_bounds_and_nonfinite(self):
        for rect in [[-1,1,10,10],[0,0,101,10],[0,0,float('nan'),10],[0,0,1,1]]:
            with self.assertRaises(ValueError):make_plan(Image.new('L',(100,80)),self.meta,rect)
    def test_clearance_blocks_unknown_and_exterior(self):
        free=np.ones((9,9),bool);free[4,4]=False;safe=clearance_mask(free,1)
        self.assertFalse(safe[3:6,3:6].any());self.assertFalse(safe[0].any());self.assertTrue(safe[1,1])
        self.assertFalse(clearance_mask(free,1000).any())


class StateTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.lab=Lab(self.temp.name,'http://unused')
        self.meta=dict(width=100,height=80,resolution=.1,origin=[0,0,0],map_id='real',map_version='75',revision=0)
        self.lab.get_nav=Mock(side_effect=lambda kind:dict(self.meta) if kind=='metadata' else raw_image())
        self.record=self.lab.import_current();self.payload=dict(map_id=self.record['id'],roi=[30,25,60,50],task_type='inspect')
    def tearDown(self):self.temp.cleanup()
    def test_off_exit_and_restart(self):
        with self.assertRaises(ValueError):self.lab.plan(self.payload)
        self.lab.mode_change(True);result=self.lab.plan(self.payload)
        self.assertTrue((Path(self.temp.name)/'sessions'/(result['id']+'.json')).exists())
        self.lab.mode_change(False)
        with self.assertRaises(ValueError):self.lab.plan(self.payload)
        self.assertEqual(Lab(self.temp.name,'http://unused').mode,'OFF')
    def test_map_switch_rejected(self):
        self.lab.mode_change(True);self.meta['map_version']='76'
        with self.assertRaisesRegex(ValueError,'版本'):self.lab.plan(self.payload)
    def test_same_version_image_change_rejected(self):
        self.lab.mode_change(True)
        self.lab.get_nav=Mock(side_effect=lambda kind:dict(self.meta) if kind=='metadata' else raw_image(0))
        with self.assertRaisesRegex(ValueError,'内容'):self.lab.plan(self.payload)
    def test_no_traversal(self):
        with self.assertRaises(ValueError):self.lab.map('../../config')
        with self.assertRaises(ValueError):self.lab.task('../../config')
    def test_saved_task_survives_restart_and_download(self):
        self.lab.mode_change(True);task=self.lab.plan(self.payload)
        restarted=Lab(self.temp.name,'http://unused')
        self.assertEqual(restarted.task(task['id']),task)
        self.assertEqual(restarted.tasks()[0]['id'],task['id'])
        server=ThreadingHTTPServer(('127.0.0.1',0),handler(restarted));thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        try:
            with urlopen('http://127.0.0.1:'+str(server.server_port)+'/api/tasks/'+task['id']+'/download') as response:
                self.assertIn('attachment',response.headers['Content-Disposition'])
                self.assertEqual(json.load(response)['id'],task['id'])
        finally:server.shutdown();server.server_close();thread.join()
    def test_http_success_routes_return_result(self):
        server=ThreadingHTTPServer(('127.0.0.1',0),handler(self.lab));thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        base='http://127.0.0.1:'+str(server.server_port)
        try:
            for path,payload,key in [('/api/mode',{'enabled':True},'mode'),('/api/plan',self.payload,'id'),('/api/datasets/demo',{},'frames')]:
                request=Request(base+path,data=json.dumps(payload).encode(),headers={'Content-Type':'application/json','X-Lab-Token':self.lab.token})
                with urlopen(request) as response:self.assertIn(key,json.load(response))
        finally:server.shutdown();server.server_close();thread.join()
    def test_http_has_no_motion_endpoint_or_csrf_bypass(self):
        server=ThreadingHTTPServer(('127.0.0.1',0),handler(self.lab));thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        base='http://127.0.0.1:'+str(server.server_port)
        try:
            for path,headers,expected in [('/api/mode',{},403),('/api/navigation/goal',{'X-Lab-Token':self.lab.token},404)]:
                request=Request(base+path,data=b'{"enabled":true}',headers={'Content-Type':'application/json',**headers})
                with self.assertRaises(HTTPError) as caught:urlopen(request)
                self.assertEqual(caught.exception.code,expected)
            self.assertEqual(self.lab.mode,'OFF')
        finally:server.shutdown();server.server_close();thread.join()


if __name__=='__main__':unittest.main()
