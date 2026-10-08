"""Observed connected assemblies and an independent, measured local floor field."""
import numpy as np


def gaussian_extents(vertices,sigma=2.):
    """Axis-aligned support of rotated Gaussian covariance, wxyz PLY rotation."""
    from scipy.spatial.transform import Rotation
    q=np.column_stack([vertices['rot_'+str(i)] for i in (1,2,3,0)]).astype(float)
    q/=np.maximum(np.linalg.norm(q,axis=1,keepdims=True),1e-12)
    R=Rotation.from_quat(q).as_matrix()
    scale=np.exp(np.column_stack([vertices['scale_'+str(i)] for i in range(3)]))
    return sigma*np.sqrt(np.sum(R*R*scale[:,None,:]**2,axis=2))


class GeometryBoundary:
    def __init__(self,assemblies,physical,floor,resolution=.04):
        from scipy.spatial import cKDTree
        from scipy.ndimage import distance_transform_edt
        import cv2
        arrays=[np.asarray(a) for a in assemblies if len(a)]
        if not arrays:raise ValueError('no completed assembly geometry')
        self.floor=floor;self.resolution=resolution;allpoints=np.concatenate(arrays)
        self.tree=cKDTree(allpoints)
        xy=allpoints[:,:2];self.origin=xy.min(0)-.4
        shape=np.ceil((xy.max(0)-self.origin+.4)/resolution).astype(int)[::-1]
        if np.prod(shape)>4_000_000:raise ValueError('assembly footprint unexpectedly large')
        field=np.zeros(shape,np.uint8)
        for a in arrays:
            uv=np.floor((a[:,:2]-self.origin)/resolution).astype('int32')
            # One hull per connected equipment assembly, never between objects.
            cv2.fillConvexPoly(field,cv2.convexHull(uv),1)
        radius=int(np.ceil(.18/resolution));field=cv2.dilate(field,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(radius*2+1,radius*2+1)))
        self.field=field.astype(bool);self.inner_distance=distance_transform_edt(self.field)*resolution
        physical=np.asarray(physical);ground=physical[np.abs(physical[:,2]-floor)<.12]
        if len(ground)<30:raise ValueError('insufficient measured floor support')
        self.ground=cKDTree(ground[:,:2])

    def floor_membership(self,points):
        uv=np.floor((points[:,:2]-self.origin)/self.resolution).astype(int)
        inside=(uv[:,0]>=0)&(uv[:,1]>=0)&(uv[:,0]<self.field.shape[1])&(uv[:,1]<self.field.shape[0])
        distance=np.zeros(len(points));ix=np.flatnonzero(inside)
        distance[ix]=self.inner_distance[uv[ix,1],uv[ix,0]]
        measured=self.ground.query(points[:,:2],workers=4)[0]<.18
        return (distance>0)&measured&(np.abs(points[:,2]-self.floor)<.18),distance

    def device_membership(self,points,extents,side):
        center=self.tree.query(points,workers=4)[0]<.14
        votes=center.astype(np.int8)
        for axis in range(3):
            for sign in (-1,1):
                probe=points.copy();probe[:,axis]+=sign*extents[:,axis]
                votes+=self.tree.query(probe,workers=4)[0]<.18
        if side=='old':return center|(votes>=3)
        # Retain thin rods, reject broad splats whose support crosses the device boundary.
        return center&(votes>=3)&(np.sort(extents,axis=1)[:,1]<.35)


def floor_seam_colors(old,old_xyz,new,new_xyz,selected,field,band=.25):
    """Only new, planar floor DC colors fade towards old floor at the boundary."""
    from scipy.spatial import cKDTree
    result=new[selected].copy();xyz=new_xyz[selected]
    member,distance=field.floor_membership(xyz);extent=gaussian_extents(result,1.)
    planar=member&(np.abs(xyz[:,2]-field.floor)<.08)&(extent[:,2]<.025)&(np.max(extent[:,:2],axis=1)>.015)
    edge=np.flatnonzero(planar&(distance<band))
    oldfloor=np.flatnonzero(np.abs(old_xyz[:,2]-field.floor)<.08)
    if not len(edge) or not len(oldfloor):return result,dict(changed=0,reason='no comparable floor seam')
    d,j=cKDTree(old_xyz[oldfloor,:2]).query(xyz[edge,:2],workers=4)
    edge=edge[d<.10];j=oldfloor[j[d<.10]]
    if len(edge)<30:return result,dict(changed=0,reason='insufficient colocated old/new floor samples')
    names=['f_dc_'+str(i) for i in range(3)]
    before=np.column_stack([result[n][edge] for n in names])*.28209479177387814+.5
    target=np.column_stack([old[n][j] for n in names])*.28209479177387814+.5
    good=(before.min(1)>.05)&(before.max(1)<.95)&(target.min(1)>.05)&(target.max(1)<.95)
    if good.sum()<30:return result,dict(changed=0,reason='insufficient reliable floor colors')
    gain=np.clip(np.median(target[good]/before[good],axis=0),.7,1.4)
    w=np.clip(1-distance[edge]/band,0,1);w=w*w*(3-2*w)
    after=np.clip(before*(1+w[:,None]*(gain-1)),0,1)
    for k,n in enumerate(names):result[n][edge]=(after[:,k]-.5)/.28209479177387814
    return result,dict(changed=len(edge),band_m=band,gain=gain.tolist(),new_floor_only=True,
                       geometry_modified=False,device_colors_modified=False,opacity_modified=False)
