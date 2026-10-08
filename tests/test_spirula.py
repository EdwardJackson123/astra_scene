import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from plyfile import PlyData,PlyElement
from astra_scene.io import ROOT,write_json
from astra_scene.spirula import fuse,sh_basis,transform_ply
from astra_scene.roi import audit

def vertex(points,degree):
 names=['x','y','z']+['f_dc_'+str(i) for i in range(3)]+['opacity']+['scale_'+str(i) for i in range(3)]+['rot_'+str(i) for i in range(4)]+['f_rest_'+str(i) for i in range(3*((degree+1)**2-1))]
 v=np.zeros(len(points),dtype=[(n,'<f4') for n in names])
 for i,n in enumerate(['x','y','z']):v[n]=np.array(points)[:,i]
 v['rot_0']=1;v['opacity']=1
 return v
def save(path,v):PlyData([PlyElement.describe(v,'vertex')],text=False).write(str(path))
class SpirulaFusionTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory(dir=ROOT/'runs');self.root=Path(self.tmp.name)
  self.base=vertex([[0,0,1],[3,0,1],[10,0,1]],0);self.fresh=vertex([[0,0,1],[1.5,0,1],[3,0,1]],2)
  save(self.root/'base.ply',self.base);save(self.root/'fresh.ply',self.fresh)
  self.scope=dict(mode='add',bounds=[[-.5,-.5,.5],[3.5,.5,1.5]],regions=[dict(oriented_box=dict(center=[x,0,1],size=[1,1,1],quaternion_xyzw=[0,0,0,1])) for x in [0,3]])
  write_json(self.root/'config.json',dict(map_ply=str(self.root/'base.ply')))
  write_json(self.root/'manual_scope.json',self.scope);(self.root/'polish').mkdir()
  self.asset=dict(key='fixed',aligned_ply=str(self.root/'fresh.ply'),output_sh_degree=2,sh_selection='explicit')
 def tearDown(self):self.tmp.cleanup()
 def merge(self):
  with patch('astra_scene.spirula.asset',return_value=self.asset):
   return fuse(self.root,self.scope,self.root/'polish/game_candidate.ply',self.root/'polish/lineage.npz','fixed')
 def test_union_gap_and_lossless_original_SH_extension(self):
  d=self.merge();self.assertEqual(d['new_gaussians'],2)
  out=PlyData.read(str(self.root/'polish/game_candidate.ply'))['vertex'].data
  for n in self.base.dtype.names:self.assertEqual(out[n][:3].tobytes(),self.base[n].tobytes())
  self.assertTrue(np.all(out['f_rest_0'][:3]==0));audit(self.root)
 def test_sh_padding_tamper_is_rejected(self):
  self.merge();path=self.root/'polish/game_candidate.ply';v=PlyData.read(str(path))['vertex'].data.copy();v['f_rest_0'][2]=.1;save(path,v)
  with self.assertRaisesRegex(ValueError,'original_modified'):audit(self.root)
 def test_model_version_change_rejected(self):
  with patch('astra_scene.spirula.asset',return_value=self.asset):
   with self.assertRaisesRegex(ValueError,'已变化'):fuse(self.root,self.scope,self.root/'no.ply',expected_key='changed')
 def test_replace_only_removes_scope_originals(self):
  self.scope['mode']='replace';write_json(self.root/'manual_scope.json',self.scope)
  d=self.merge();self.assertEqual(d['removed_original'],2);audit(self.root)
  ids=np.load(self.root/'polish/lineage.npz')['original_ids'];self.assertEqual(ids.tolist(),[2,-1,-1])
 def test_same_degree_output_is_SH0(self):
  from astra_scene.spirula import ply_degree
  save(self.root/'fresh.ply',vertex([[0,0,1],[3,0,1]],0))
  self.asset.update(output_sh_degree=0,sh_selection='auto')
  self.merge();self.assertEqual(ply_degree(self.root/'polish/game_candidate.ply'),0);audit(self.root)
 def test_default_cannot_silently_raise_SH(self):
  self.asset['sh_selection']='auto'
  with self.assertRaisesRegex(ValueError,'默认 SH'):self.merge()
 def test_transform_preserves_directional_color(self):
  from scipy.spatial.transform import Rotation
  v=vertex([[.3,.1,.8]],2);v['f_dc_0']=.2
  for i in range(24):v['f_rest_'+str(i)]=i*.01
  save(self.root/'input.ply',v);R=Rotation.from_euler('z',37,degrees=True).as_matrix()
  out=transform_ply(self.root/'input.ply',self.root/'transformed.ply',2,R,np.array([1,2,3]))
  dirs=np.array([[1,0,0],[0,1,0],[0,0,1.]],float)
  def coeff(row):
   return np.vstack([np.array([row['f_dc_'+str(i)] for i in range(3)]),np.array([row['f_rest_'+str(i)] for i in range(24)]).reshape(3,8).T])
  np.testing.assert_allclose(sh_basis(dirs)@coeff(out[0]),sh_basis(dirs@R)@coeff(v[0]),atol=1e-7)
if __name__=='__main__':unittest.main()
