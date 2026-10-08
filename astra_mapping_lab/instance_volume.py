"""One bounded convex volume per accepted instance; never a scene-wide hull."""
import numpy as np
from scipy.spatial import ConvexHull,QhullError


class InstanceVolumes:
    def __init__(self,instances,margin=.03):
        self.volumes=[];self.margin=margin
        for name,points in instances.items():
            p=np.unique(np.asarray(points,dtype=float),axis=0)
            if len(p)<4:continue
            try:hull=ConvexHull(p)
            except QhullError:
                # Give planar or linear supports a bounded physical thickness.
                offsets=np.vstack([np.eye(3),-np.eye(3)])*.015
                p=np.concatenate([p+v for v in offsets]);hull=ConvexHull(p)
            self.volumes.append((name,p.min(0),p.max(0),hull.equations))
        if not self.volumes:raise ValueError('No bounded accepted instances')

    def contains(self,points):
        result=np.zeros(len(points),bool)
        for name,lo,hi,equations in self.volumes:
            ix=np.flatnonzero(~result&((points>=lo-self.margin)&(points<=hi+self.margin)).all(1))
            for start in range(0,len(ix),10000):
                ids=ix[start:start+10000];inside=np.ones(len(ids),bool)
                for block in range(0,len(equations),32):
                    e=equations[block:block+32]
                    inside &= (points[ids]@e[:,:3].T+e[:,3]<=self.margin).all(1)
                result[ids[inside]]=True
        return result
