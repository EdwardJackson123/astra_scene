"""Conservative dependencies between overlapping detections, not object identity."""
import numpy as np
from scipy.spatial import cKDTree


def build_groups(regions,assemblies,proposals,reviews,masks,floor):
    ids=[r['id'] for r in regions];parents=list(range(len(ids)));edges=[]
    surfaces={key:np.asarray(assemblies[key])[np.asarray(assemblies[key])[:,2]>floor+.20] for key in ids}
    def find(i):
        while parents[i]!=i:parents[i]=parents[parents[i]];i=parents[i]
        return i
    for i,a in enumerate(ids):
        aa=surfaces[a]
        if len(aa)<20:continue
        for j in range(i):
            b=ids[j];bb=surfaces[b]
            if len(bb)<20:continue
            if np.any(aa.min(0)>bb.max(0)+.10) or np.any(bb.min(0)>aa.max(0)+.10):continue
            overlap=min(float((cKDTree(bb).query(aa)[0]<.08).mean()),float((cKDTree(aa).query(bb)[0]<.08).mean()))
            if overlap<.20:continue
            views_a={int(k.split('__')[1]) for k in masks if k.startswith(a+'__')}
            views_b={int(k.split('__')[1]) for k in masks if k.startswith(b+'__')}
            agreements=[]
            for view in sorted(views_a&views_b):
                ma=masks[a+'__'+str(view)];mb=masks[b+'__'+str(view)]
                intersection=int((ma&mb).sum());union=int((ma|mb).sum())
                if union and intersection/union>=.30:agreements.append(view)
            if len(agreements)<2:continue
            side_a=proposals[a]['evidence_side'];side_b=proposals[b]['evidence_side']
            if side_a==side_b:
                if proposals[a]['source_label']!=proposals[b]['source_label']:continue
                relation='overlapping_same_label_detections'
            else:
                if not any(reviews[k].get('change_type') in ('layout','move','remove','add') for k in (a,b)):continue
                relation='old_new_update_dependency'
            parents[find(i)]=find(j)
            edges.append(dict(a=a,b=b,relation=relation,geometry_overlap=overlap,mask_agreement_views=agreements))
    groups={}
    for i,key in enumerate(ids):groups.setdefault(find(i),[]).append(key)
    return dict(groups=[dict(id=f'event_{i:03d}',members=members,decision_unit='joint update; never independently restore stale members') for i,members in enumerate(groups.values())],
                edges=edges,semantic_source='existing DeepSeek review; no new calls',object_identity_verified=False,
                ground_excluded=True,bounds_authorize_deletion=False)


def removal_masks(clear,support,replacement,old_visible,second_axis):
    """Stable old surfaces only yield to repeatedly rendered current replacements."""
    old=(old_visible>=2)&(((clear>=4)&(support==0))|((replacement>=3)&(support>=2)&(clear==0)))
    new=(clear>=4)&(support==0)&(second_axis>.025)
    return old,new
