"""Evidence accumulation inspired by SuperMap; scores are not calibrated probabilities."""
import numpy as np
def existence_score(surface,free,occluded,prior=.9):
    surface=np.asarray(surface);free=np.asarray(free)
    # Occlusion is deliberately neutral. Cap correlated repeated observations.
    logodds=np.log(prior/(1-prior))+.85*np.minimum(surface,8)-.55*np.minimum(free,16)
    return 1/(1+np.exp(-np.clip(logodds,-6,6)))
def retire_surface(surface,free,score,owned):
    return np.asarray(owned,bool)&(np.asarray(free)>=4)&(np.asarray(surface)==0)&(np.asarray(score)<.15)
