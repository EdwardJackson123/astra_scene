"""Continuous transition-band fusion with bounded local photometric refinement.

Core regions remain proposals, not instance ownership. A smooth spatial field
replaces their hard seam; original rows outside its support stay byte identical.
This experimental backend never publishes assets or changes capture geometry.
"""
import argparse, copy, json, fcntl, time
from pathlib import Path
import numpy as np
from .io import ROOT, output_path, write_json, sha256


def field(xyz, scope, band=.30):
    from .roi import box_geometry
    if not np.isfinite(band) or band <= 0:raise ValueError('positive_transition_band_required')
    distance=np.full(len(xyz),np.inf)
    for region in scope.get('regions',[scope]):
        center,size,rotation=box_geometry(region)
        q=np.maximum(np.abs((xyz-center)@rotation)-size/2,0)
        distance=np.minimum(distance,np.linalg.norm(q,axis=1))
    t=np.clip(distance/band,0,1)
    return 1-t*t*(3-2*t)


def context_scope(scope,band):
    from .roi import box_corners
    result=copy.deepcopy(scope)
    for r in result.get('regions',[result]):
        if r.get('oriented_box'):
            r['oriented_box']['size']=(np.asarray(r['oriented_box']['size'])+2*band).tolist()
            corners=box_corners(r);r['bounds']=[corners.min(0).tolist(),corners.max(0).tolist()]
        else:
            b=np.asarray(r['bounds']);r['bounds']=[(b[0]-band).tolist(),(b[1]+band).tolist()]
    if 'regions' in result:
        bounds=np.asarray([r['bounds'] for r in result['regions']])
        result['bounds']=[bounds[:,0].min(0).tolist(),bounds[:,1].max(0).tolist()]
    result.update(context_band_m=band,backend='context_fusion',automatic_scope=True,manual_scope=False)
    return result


def run(source, baseline, output, iterations=320, band=.30, width=768, depth_weight=.03):
    import torch
    from gsplat.rendering import rasterization
    from plyfile import PlyData,PlyElement
    from .postprocess import load
    from .spirula import asset,ply_degree
    from .training_views import training_views
    from .quality import target_mask
    from .roi import contains
    torch.manual_seed(12);np.random.seed(12);torch.set_num_threads(4)
    source=output_path(source);baseline=output_path(baseline);output=output_path(output)
    if iterations<1:raise ValueError('positive_iterations_required')
    output.mkdir(exist_ok=False)
    baseline_ply=baseline/'artifacts/scene.ply'
    if not baseline_ply.is_file():baseline_ply=baseline/'polish/game_candidate.ply'
    cfg,data,split,_=load(source);spec=asset(source,True)
    scope=json.loads((baseline/'manual_scope.json').read_text());expanded=context_scope(scope,band)
    if not split['validation_views'] or set(split['train_views']) & set(split['validation_views']+split['test_views']):
        raise ValueError('disjoint_validation_required')
    if ply_degree(spec['aligned_ply'])!=0 or ply_degree(cfg['map_ply'])!=0:
        raise ValueError('experimental_context_backend_currently_requires_SH0')
    write_json(output/'manual_scope.json',expanded)
    write_json(output/'inputs.json',dict(source=str(source),baseline=str(baseline),
        base_sha256=sha256(cfg['map_ply']),fresh_sha256=spec['ply_sha256'],
        baseline_sha256=sha256(baseline_ply),views_sha256=spec['views_sha256'],
        split=split,iterations=iterations,band_m=band,width=width,depth_weight=depth_weight,
        code_sha256=sha256(Path(__file__)),experimental=True))
    base=PlyData.read(cfg['map_ply'],mmap=True)['vertex'].data
    fresh=PlyData.read(spec['aligned_ply'],mmap=True)['vertex'].data
    xyz=lambda v:np.column_stack([v[n] for n in ['x','y','z']])
    old_w=field(xyz(base),scope,band);new_w=field(xyz(fresh),scope,band)
    old_ids=np.flatnonzero(old_w<1-1e-6);new_ids=np.flatnonzero(new_w>1e-4)
    initial=np.empty(len(old_ids)+len(new_ids),dtype=base.dtype)
    initial[:len(old_ids)]=base[old_ids]
    for name in base.dtype.names:initial[name][len(old_ids):]=fresh[name][new_ids]
    count=len(old_ids);weights=np.r_[1-old_w[old_ids],new_w[new_ids]]
    active=np.r_[old_w[old_ids]>0,np.ones(len(new_ids),bool)]
    alpha=1/(1+np.exp(-np.clip(initial['opacity'],-30,30)))
    alpha=np.clip(alpha*weights,1e-7,1-1e-7)
    initial['opacity'][active]=np.log(alpha[active]/(1-alpha[active]))
    active_ids=np.flatnonzero(active);frozen_ids=np.flatnonzero(~active)
    fresh_active=(active_ids>=count)
    views=training_views(source/'job',data,width)
    tensor=lambda a:torch.as_tensor(np.asarray(a).copy(),dtype=torch.float32,device='cuda')
    arrays=dict(means=xyz(initial),colors=np.column_stack([initial['f_dc_'+str(i)] for i in range(3)])*.28209479177387814+.5,
        opacity=initial['opacity'][:,None],scale=np.column_stack([initial['scale_'+str(i)] for i in range(3)]))
    fixed={k:tensor(v[frozen_ids]) for k,v in arrays.items()}
    params={k:torch.nn.Parameter(tensor(v[active_ids])) for k,v in arrays.items()}
    anchors={k:v.detach().clone() for k,v in params.items()}
    quats=tensor(np.column_stack([initial['rot_'+str(i)] for i in range(4)])[np.r_[frozen_ids,active_ids]])
    quats=quats/torch.linalg.vector_norm(quats,dim=1,keepdim=True).clamp_min(1e-8)
    fresh_mask=tensor(fresh_active[:,None]);start=time.monotonic()
    optimizer=torch.optim.Adam([dict(params=[params[k]],lr=lr) for k,lr in [('means',.00015),('colors',.003),('opacity',.008),('scale',.001)]])
    metadata={}
    for i in sorted(set(split['train_views']+split['validation_views'])):
        v=views[i];h,w=v['depth'].shape
        row=dict(depth=v['depth'],valid=v['confidence']&(v['depth']>.6)&(v['depth']<6),K=v['K'],pose=data['poses'][i],rgb=v['rgb'])
        mask=target_mask(row,expanded)
        metadata[i]=dict(K=tensor(v['K'])[None],view=tensor(np.linalg.inv(data['poses'][i]))[None],
                         truth=tensor(v['rgb']),depth=tensor(v['depth']),mask=torch.as_tensor(mask,device='cuda'),width=w,height=h)
    def render(i):
        m=metadata[i]
        p={k:torch.cat([fixed[k],v],0) for k,v in params.items()}
        rgb,alpha,_=rasterization(p['means'],quats,p['scale'].exp(),p['opacity'].sigmoid().flatten(),
              p['colors'].clamp(0,1),m['view'],m['K'],m['width'],m['height'],packed=True,render_mode='RGB+ED')
        return rgb[0,:,:,:3],rgb[0,:,:,3],alpha[0,:,:,0]
    def score():
        values=[]
        with torch.no_grad():
            for i in split['validation_views']:
                pred,dep,alpha=render(i);m=metadata[i];mask=m['mask']
                if int(mask.sum())<100:raise ValueError('insufficient_validation_target')
                tolerance=.08+.01*m['depth']
                depth_error=torch.relu((dep-m['depth']).abs()-tolerance)[mask].clamp(max=1).mean()
                values.append(dict(view=i,target_l1=float((pred-m['truth']).abs()[mask].mean()),full_l1=float((pred-m['truth']).abs().mean()),depth_excess_m=float(depth_error)))
        return float(np.mean([v['target_l1']+depth_weight*v['depth_excess_m'] for v in values])),values
    def snapshot():return {k:v.detach().cpu().clone() for k,v in params.items()}
    best_score,metrics=score();best=snapshot();selected=0;history=[dict(step=0,score=best_score,views=metrics)]
    print('CONTEXT_INIT',len(initial),'editable',len(active_ids),'validation',best_score,flush=True)
    # Persist the unoptimized field fusion to separate initialization from learning.
    PlyData([PlyElement.describe(initial,'vertex')],text=False).write(str(output/'initial.ply'))
    ids=np.array(split['train_views']);order=np.random.permutation(ids)
    for step in range(1,iterations+1):
        if (step-1)%len(ids)==0:order=np.random.permutation(ids)
        i=int(order[(step-1)%len(ids)]);m=metadata[i];pred,dep,alpha=render(i)
        mask=m['mask'];error=(pred-m['truth']).abs()
        loss=error[mask].mean() if mask.any() else error.mean()
        if (~mask).any():loss=loss+.5*error[~mask].mean()
        if mask.any() and depth_weight>0:
            tolerance=.08+.01*m['depth']
            depth_excess=torch.relu((dep-m['depth']).abs()-tolerance)[mask].clamp(max=1).mean()
            loss=loss+depth_weight*depth_excess+.005*torch.relu(.5-alpha[mask]).mean()
        loss=loss+.001*(params['opacity']-anchors['opacity']).square().mean()+.01*(params['scale']-anchors['scale']).square().mean()
        optimizer.zero_grad(set_to_none=True);loss.backward()
        # Existing Gaussian geometry remains fixed, even within the seam.
        params['means'].grad.mul_(fresh_mask);params['scale'].grad.mul_(fresh_mask)
        if step<40:params['means'].grad.zero_();params['scale'].grad.zero_()
        optimizer.step()
        with torch.no_grad():
            params['means'].copy_(torch.maximum(torch.minimum(params['means'],anchors['means']+.03*fresh_mask),anchors['means']-.03*fresh_mask))
            params['scale'].copy_(torch.maximum(torch.minimum(params['scale'],anchors['scale']+.25*fresh_mask),anchors['scale']-.25*fresh_mask))
            params['colors'].clamp_(0,1);params['opacity'].clamp_(-16,16)
        if step%40==0 or step==iterations:
            value,metrics=score();history.append(dict(step=step,score=value,views=metrics))
            if value<best_score:best_score=value;best=snapshot();selected=step
            write_json(output/'progress.json',dict(step=step,total=iterations,validation_l1=value,best=best_score,elapsed=time.monotonic()-start))
            print('CONTEXT_STEP',step,value,'best',best_score,flush=True)
    final=initial.copy()
    for j,n in enumerate(['x','y','z']):final[n][active_ids]=best['means'].numpy()[:,j]
    for j in range(3):
        final['f_dc_'+str(j)][active_ids]=(best['colors'].numpy()[:,j]-.5)/.28209479177387814
        final['scale_'+str(j)][active_ids]=best['scale'].numpy()[:,j]
    final['opacity'][active_ids]=best['opacity'].numpy().flatten()
    # Bound position updates to the declared context scope. Restore only centers
    # whose displacement crosses its boundary, avoiding post-hoc extra deletion.
    escaped=~contains(xyz(final[active_ids]),expanded)
    for n in ['x','y','z']:final[n][active_ids[escaped]]=initial[n][active_ids[escaped]]
    assert final[frozen_ids].tobytes()==base[old_ids[frozen_ids]].tobytes()
    PlyData([PlyElement.describe(final,'vertex')],text=False).write(str(output/'scene.ply'))
    np.savez_compressed(output/'lineage.npz',original_ids=np.r_[old_ids,np.full(len(new_ids),-1,np.int64)],owner=np.full(len(final),-1,np.int64))
    write_json(output/'report.json',dict(backend='context_fusion',selected_step=selected,validation_score=best_score,depth_weight=depth_weight,
        history=history,old_frozen_points=len(frozen_ids),old_transition_points=int((active_ids<count).sum()),
        fresh_points=len(new_ids),max_new_displacement_per_axis_m=.03,
        unchanged_outside_support=True,escaped_centers_restored=int(escaped.sum()),
        source_sha256=spec['ply_sha256'],candidate_sha256=sha256(output/'scene.ply'),
        deployable=False,test_views_used_for_gradients_or_checkpoint_selection=False,
        test_status='development_regression; followup experiments can be informed by prior regression findings',
        limitation='Experimental smooth context field, not instance identity; local photometric optimization does not prove geometry or navigation safety.'))
    print('CONTEXT_DONE',output,flush=True)


def verified_prepared(path, reconstruction, baseline_ply, source):
    """A reusable result must match the exact reconstruction and baseline bytes."""
    path=output_path(path)
    inputs=json.loads((path/'inputs.json').read_text())
    report=json.loads((path/'report.json').read_text())
    if (inputs['fresh_sha256']!=reconstruction['ply_sha256']
            or inputs['views_sha256']!=reconstruction['views_sha256']
            or inputs['base_sha256']!=reconstruction['base_sha256']
            or inputs['split']!=json.loads((Path(source)/'job/refit_report.json').read_text())
            or inputs['baseline_sha256']!=sha256(baseline_ply)
            or report['candidate_sha256']!=sha256(path/'scene.ply')):
        raise ValueError('context_prepared_source_mismatch')
    return path,report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--iterations',type=int,default=320);p.add_argument('--band',type=float,default=.30);p.add_argument('--width',type=int,default=768);p.add_argument('--depth-weight',type=float,default=.03)
    a=p.parse_args()
    with (ROOT/'runs/gpu.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        run(a.source,a.baseline,a.output,a.iterations,a.band,a.width,a.depth_weight)
