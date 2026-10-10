"""Separate offline preview eligibility from alignment acceptance."""
import math
import numpy as np

POLICIES=('strict','preview')

def preview_allowed(report, policy):
    if policy!='preview':return False
    try:
        s=float(report['scale']);r=np.asarray(report['rotation'],float);t=np.asarray(report['translation'],float)
        return bool(.85<s<1.15 and math.isfinite(s) and r.shape==(3,3) and t.shape==(3,)
            and np.isfinite(r).all() and np.isfinite(t).all()
            and np.allclose(r.T@r,np.eye(3),atol=1e-4) and abs(np.linalg.det(r)-1)<1e-4
            and report['fit']['inliers']>=30 and report['holdout']['matches']>=30)
    except (KeyError,TypeError,ValueError):return False

def usable(report):
    return report.get('accepted') is True or (report.get('preview_only') is True and preview_allowed(report,'preview'))
