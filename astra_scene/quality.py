"""Structural acceptance against held-out observations, never a navigation certificate."""
import argparse,json
from pathlib import Path
import cv2,numpy as np
from .roi import contains
from .geometry import backproject
from .io import write_json

POLICY=dict(min_pixels=500,max_rgb_l1=.12,max_missing_fraction=.15,max_foreground_fraction=.10,
    depth_tolerance_m=.08,depth_relative_tolerance=.01)

def target_mask(row,scope):
    mask=np.zeros(row['depth'].shape,bool)
    y,x=np.nonzero(row['valid'])
    p,_=backproject(row)
    mask[y,x]=contains(p,scope)
    return mask

def structural_metrics(rgb,depth,alpha,row,scope,policy=None):
    policy={**POLICY,**(policy or {})}
    mask=target_mask(row,scope)
    count=int(mask.sum())
    if count<policy['min_pixels']:
        return dict(passed=False,pixels=count,reasons=['insufficient_observed_target_pixels'])
    truth=row['depth'];tol=policy['depth_tolerance_m']+policy['depth_relative_tolerance']*truth
    valid=np.isfinite(depth)&(depth>0)&np.isfinite(alpha)&(alpha>=.5)
    missing=(~valid)|(depth>truth+tol)
    front=valid&(depth<truth-tol)
    l1=float(np.abs(rgb-row['rgb'])[mask].mean())
    finite_rgb=bool(np.isfinite(rgb[mask]).all())
    missing_rate=float(missing[mask].mean());front_rate=float(front[mask].mean())
    reasons=[]
    if not finite_rgb:reasons.append('nonfinite_rgb')
    if l1>policy['max_rgb_l1']:reasons.append('target_absolute_rgb_error')
    if missing_rate>policy['max_missing_fraction']:reasons.append('missing_observed_surface')
    if front_rate>policy['max_foreground_fraction']:reasons.append('foreground_geometry_disagreement')
    result=dict(passed=not reasons,pixels=count,target_rgb_l1=l1 if finite_rgb else None,
        missing_surface_fraction=missing_rate,foreground_disagreement_fraction=front_rate,
        reasons=reasons,depth_residual_p90_m=float(np.percentile(np.nan_to_num(np.abs(depth[mask]-truth[mask]),nan=1e6,posinf=1e6),90)))
    if 'regions' in scope:
        result['regions']=[structural_metrics(rgb,depth,alpha,row,region,policy) for region in scope['regions']]
        result['passed'] &= all(r['passed'] for r in result['regions'])
        if not result['passed'] and not reasons:result['reasons']=['individual_region_failed']
    return result

def evaluate(run,candidates,out,width=1024):
    from .postprocess import load
    from .training_views import training_views
    from vps.refinement.gaussian_renderer import GsplatGaussianRenderer,GaussianCamera
    cfg,d,split,report=load(run)
    scope=json.loads((run/'manual_scope.json').read_text())
    views=training_views(run/'job',d,width)
    out.mkdir(parents=True,exist_ok=False)
    renderer=GsplatGaussianRenderer('cuda');metrics=[]
    ids=split['validation_views']+split['test_views']
    strips={i:[] for i in ids};baseline={}
    from .mask_consensus import edit_regression
    for label,path in candidates.items():
        for i in ids:
            v=views[i];h,w=v['depth'].shape;k=v['K']
            rendered=renderer.render(path,d['poses'][i],GaussianCamera(w,h,k[0,0],k[1,1],k[0,2],k[1,2]))
            rgb=rendered.rgb.astype(np.float32)/255
            row=dict(depth=v['depth'],valid=v['confidence']&(v['depth']>0),K=k,pose=d['poses'][i],rgb=v['rgb'])
            m=structural_metrics(rgb,rendered.depth,rendered.alpha,row,scope)
            if Path(path).resolve()==Path(cfg["map_ply"]).resolve():
                base_row={**row,"depth":rendered.depth,"valid":np.isfinite(rendered.depth)&(rendered.depth>0)&(rendered.alpha>.5)}
                baseline[i]=(rendered.rgb.copy(),target_mask(base_row,scope))
            else:
                if i not in baseline:
                    m["passed"]=False;m["reasons"].append("missing_original_render_for_regression")
                else:
                    base_rgb,base_mask=baseline[i]
                    edit=cv2.dilate((target_mask(row,scope)|base_mask).astype(np.uint8),np.ones((11,11),np.uint8)).astype(bool)
                    regression=edit_regression(base_rgb,rendered.rgb,np.uint8(np.clip(row["rgb"]*255,0,255)),edit,row["valid"])
                    m["edit_regression"]=regression;m["passed"] &= regression["passed"]
                    m["reasons"].extend(regression["reasons"])
            m.update(candidate=label,view=i,split='validation' if i in split['validation_views'] else 'test')
            metrics.append(m)
            mask=target_mask(row,scope);y,x=np.nonzero(mask)
            if len(y):
                y0=max(0,y.min()-20);y1=min(h,y.max()+21);x0=max(0,x.min()-20);x1=min(w,x.max()+21)
            else:
                # Keep insufficient-target metrics failed; show full view for diagnosis.
                y0,y1,x0,x1=0,h,0,w
            im=rendered.rgb[y0:y1,x0:x1].copy()
            cv2.putText(im,label,(6,18),cv2.FONT_HERSHEY_SIMPLEX,.5,(240,40,40),1,cv2.LINE_AA)
            strips[i].append(im)
            if label==list(candidates)[-1]:
                gt=np.uint8(v['rgb'][y0:y1,x0:x1]*255)
                cv2.putText(gt,'phone',(6,18),cv2.FONT_HERSHEY_SIMPLEX,.5,(240,40,40),1,cv2.LINE_AA)
                strips[i].append(gt)
            print(label,i,m,flush=True)
        # Do not accumulate every full 4.6M model on the GPU.
        renderer=GsplatGaussianRenderer('cuda')
        import torch
        torch.cuda.empty_cache()
    for i,ims in strips.items():
        cv2.imwrite(str(out/f'comparison_{i:03d}.jpg'),cv2.cvtColor(np.concatenate(ims,axis=1),cv2.COLOR_RGB2BGR))
    result=dict(policy=POLICY,policy_status='conservative_engineering_thresholds_not_field_calibrated',
        views=metrics,independent_revisit=False,depth_source='same_video_inferred_not_ground_truth',
        test_status='development_regression_views_reused_across_experiments',candidate_paths={k:str(v) for k,v in candidates.items()},
        passed={label:all(m['passed'] for m in metrics if m['candidate']==label and m['split']=='test') for label in candidates})
    write_json(out/'quality.json',result)
    (out/'index.html').write_text('<meta charset="utf-8"><title>局部重建对照</title><style>body{font:16px system-ui;margin:24px}img{max-width:100%}</style><h1>局部重建对照</h1><p>从左到右：'+', '.join(candidates)+'、手机实拍。对照帧未参与梯度训练；检查点由验证帧选取。这些测试帧用于开发回归，不是独立复访；推断深度只用于一致性检查。额外检查编辑区外渲染变化和目标改善幅度；背景稳定不能代替更新有效。</p>'+''.join(f'<h2>视角 {i}</h2><img src="comparison_{i:03d}.jpg">' for i in ids),encoding='utf-8')
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('out',type=Path)
    p.add_argument('--candidate',action='append',required=True)
    a=p.parse_args();evaluate(a.run,{v.split('=',1)[0]:Path(v.split('=',1)[1]) for v in a.candidate},a.out)
