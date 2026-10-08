import json,os,sys,tempfile,threading,unittest
from pathlib import Path
from unittest.mock import patch
from http.server import ThreadingHTTPServer
from urllib.request import Request,urlopen
from urllib.error import HTTPError
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cloud_server import Cloud,handler
from cloud_client import CloudClient
from server import Lab,handler as robot_handler

class CloudTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.cloud=Cloud(Path(self.tmp.name)/'cloud','t'*40,lambda key:self.cloud.update(key,state='READY'))
 def tearDown(self):self.tmp.cleanup()
 def test_sealed_jobs_and_path_escape(self):
  s=self.cloud.create(dict(demo=True,iterations=100));key=s['id']
  with self.assertRaises(ValueError):self.cloud.upload(key,'source','../x',0,b'a')
  self.cloud.upload(key,'source','a.json',0,b'{}')
  self.assertEqual(self.cloud.upload(key,'source','a.json',0,b'{}')['offset'],2)
  self.assertEqual((self.cloud.directory(key)/'source/a.json').read_bytes(),b'{}')
  with self.assertRaises(ValueError):self.cloud.upload(key,'source','a.json',0,b'[]')
  self.cloud.commit(key);self.cloud.queue.join()
  with self.assertRaises(ValueError):self.cloud.upload(key,'source','a.json',2,b'x')
  with self.assertRaises(ValueError):self.cloud.commit(key)
 def test_chunk_timeout_retry(self):
  with patch.dict(os.environ,ASTRA_CLOUD_URL='http://unused',ASTRA_CLOUD_TOKEN='t'*40):
   lab=Lab(Path(self.tmp.name)/'retry_robot','http://unused')
   key=self.cloud.create(dict(demo=True,iterations=100))['id'];calls=[]
   from io import BytesIO
   def lost_response(req,timeout):
    calls.append(req);result=self.cloud.upload(key,'source','a.json',0,req.data)
    if len(calls)==1:raise TimeoutError('response lost after write')
    return BytesIO(json.dumps(result).encode())
   with patch('cloud_client.urlopen',side_effect=lost_response),patch('cloud_client.time.sleep'):
    self.assertEqual(lab.cloud.request('/jobs/'+key+'/files/source/a.json',b'{}',0)['offset'],2)
   self.assertEqual(len(calls),2);self.assertEqual((self.cloud.directory(key)/'source/a.json').read_bytes(),b'{}')
 def test_light_robot_rejects_heavy_endpoints_and_remote_demo(self):
  http=ThreadingHTTPServer(('127.0.0.1',0),handler(self.cloud));thread=threading.Thread(target=http.serve_forever,daemon=True);thread.start()
  try:
   with patch.dict(os.environ,ASTRA_CLOUD_URL='http://127.0.0.1:'+str(http.server_port),ASTRA_CLOUD_TOKEN='t'*40):
    lab=Lab(Path(self.tmp.name)/'robot','http://unused');lab.mode_change(True)
    result=lab.cloud.submit(dict(demo=True,iterations=100))
    import time
    deadline=time.monotonic()+5
    while lab.cloud.active and time.monotonic()<deadline:time.sleep(.02)
    self.cloud.queue.join();rows=lab.cloud.jobs();self.assertEqual(rows[0]['state'],'READY');self.assertEqual(rows[0]['uploaded_bytes'],0)
    self.assertFalse(list(lab.data.rglob('*.ply')))
    rh=ThreadingHTTPServer(('127.0.0.1',0),robot_handler(lab));rt=threading.Thread(target=rh.serve_forever,daemon=True);rt.start()
    try:
     for endpoint in ['/api/trajectory','/api/packages','/api/reconstruction/start','/api/replays','/api/datasets/tracking-demo']:
      req=Request('http://127.0.0.1:'+str(rh.server_port)+endpoint,b'{}',{'Content-Type':'application/json','X-Lab-Token':lab.token})
      with self.assertRaises(HTTPError) as e:urlopen(req)
      self.assertEqual(e.exception.code,400)
    finally:rh.shutdown();rh.server_close();rt.join()
   with self.assertRaises(HTTPError) as e:urlopen(Request('http://127.0.0.1:'+str(http.server_port)+'/jobs',b'{}'))
   self.assertEqual(e.exception.code,403)
  finally:http.shutdown();http.server_close();thread.join()
