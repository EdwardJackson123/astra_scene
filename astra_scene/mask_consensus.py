"""Conservative per-Gaussian multi-view mask decisions, independent of object labels."""
import numpy as np
def select_points(mask_votes,visible_votes,free_votes=None,surface_votes=None,minimum=3):
 mask_votes=np.asarray(mask_votes);visible_votes=np.asarray(visible_votes)
 selected=(mask_votes>=minimum)&(mask_votes/np.maximum(visible_votes,1)>=.8)
 if free_votes is not None:
  selected &= (np.asarray(free_votes)>=4)&(np.asarray(surface_votes)==0)
 return selected
def background_metrics(before,after,edit_mask):
 delta=np.abs(np.asarray(before,dtype=np.float32)-np.asarray(after,dtype=np.float32))/255
 values=delta.mean(-1)[~np.asarray(edit_mask,bool)]
 if len(values)<100:return dict(ready=False,pixels=int(len(values)),reason="insufficient_background")
 mean=float(values.mean());p95=float(np.quantile(values,.95))
 return dict(ready=True,pixels=int(len(values)),mean_l1=mean,p95_l1=p95,passed=mean<=.01 and p95<=.04)

def edit_regression(before,after,truth,edit_mask,valid=None):
 edit_mask=np.asarray(edit_mask,bool)
 valid=np.ones_like(edit_mask) if valid is None else np.asarray(valid,bool)
 bg=background_metrics(before,after,edit_mask|~valid)
 target=edit_mask&valid
 if int(target.sum())<100:
  return dict(passed=False,background=bg,reasons=["insufficient_target_pixels"])
 old_error=float(np.abs(np.asarray(before,float)-truth)[target].mean()/255)
 new_error=float(np.abs(np.asarray(after,float)-truth)[target].mean()/255)
 improvement=(old_error-new_error)/max(old_error,1e-8)
 effective=old_error<=.02 or improvement>=.05
 reasons=[]
 if not bg.get("passed",False):reasons.append("outside_edit_region_changed")
 if not effective:reasons.append("insufficient_target_improvement")
 return dict(passed=not reasons,background=bg,target_original_l1=old_error,target_candidate_l1=new_error,
             relative_improvement=improvement,reasons=reasons)

def mask_observation(visible,mask_values=None):
 visible=np.asarray(visible,bool)
 if mask_values is None:return np.zeros_like(visible),np.zeros_like(visible)
 return visible,visible&np.asarray(mask_values,bool)
