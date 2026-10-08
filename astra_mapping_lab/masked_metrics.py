"""Report full-image and static-region appearance separately for person removal."""
import numpy as np


def appearance(pred,gt,keep):
    import cv2
    a=np.asarray(pred,dtype=float)/255;b=np.asarray(gt,dtype=float)/255;keep=np.asarray(keep,bool)
    if a.shape!=b.shape or a.shape[:2]!=keep.shape or a.ndim!=3 or a.shape[2]!=3:raise ValueError('评估图像/遮罩形状不匹配')
    # A SSIM window must not straddle an ignored person boundary.
    valid=cv2.erode(keep.astype(np.uint8),np.ones((11,11),np.uint8),borderType=cv2.BORDER_CONSTANT,borderValue=0)>0
    if valid.sum()<100:raise ValueError('静态评估区域不足')
    ma=cv2.GaussianBlur(a,(11,11),1.5);mb=cv2.GaussianBlur(b,(11,11),1.5)
    va=cv2.GaussianBlur(a*a,(11,11),1.5)-ma*ma;vb=cv2.GaussianBlur(b*b,(11,11),1.5)-mb*mb
    cov=cv2.GaussianBlur(a*b,(11,11),1.5)-ma*mb
    ssim=((2*ma*mb+.01**2)*(2*cov+.03**2)/((ma*ma+mb*mb+.01**2)*(va+vb+.03**2)))
    def metrics(mask):
        diff=(a-b)[mask];return dict(rgb_l1=float(np.mean(abs(diff))),psnr=float(-10*np.log10(max(np.mean(diff**2),1e-12))),ssim=float(ssim[mask].mean()))
    return dict(full=metrics(np.ones(keep.shape,bool)),static=metrics(valid),masked_fraction=float(1-keep.mean()),static_eval_pixels=int(valid.sum()))
