import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from geometry_boundary import GeometryBoundary,gaussian_extents

class BoundaryTests(unittest.TestCase):
    def test_disconnected_equipment_does_not_authorize_floor_between_it(self):
        x,y=np.meshgrid(np.linspace(-.1,.1,8),np.linspace(-.1,.1,8))
        a=np.c_[x.ravel(),y.ravel(),np.ones(x.size)]
        x,y=np.meshgrid(np.linspace(-.5,3.5,81),np.linspace(-.5,.5,21))
        physical=np.c_[x.ravel(),y.ravel(),np.zeros(x.size)]
        field=GeometryBoundary([a,a+[3,0,0]],physical,0)
        points=np.array([[0,0,0],[3,0,0],[1.5,0,0],[0,0,2.5]])
        self.assertEqual(field.floor_membership(points)[0].tolist(),[True,True,False,False])
        self.assertFalse(field.device_membership(np.array([[0,0,2.5]]),np.zeros((1,3)),'current')[0])

    def test_rotated_gaussian_extent_and_large_boundary_crossing(self):
        dtype=[('rot_'+str(i),'f4') for i in range(4)]+[('scale_'+str(i),'f4') for i in range(3)]
        v=np.zeros(1,dtype=dtype);v['rot_0']=v['rot_3']=2**-.5
        for i,scale in enumerate([.1,.01,.02]):v['scale_'+str(i)]=np.log(scale)
        np.testing.assert_allclose(gaussian_extents(v,1)[0],[.01,.1,.02],rtol=1e-5)

if __name__=='__main__':unittest.main()
