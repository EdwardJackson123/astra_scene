"""Binary observation masks: white keeps observations, black ignores them."""
from pathlib import Path
import numpy as np
from PIL import Image
from replay import asset


def read_mask(path,shape):
    with Image.open(path) as image:
        if image.mode not in ('1','L'):raise ValueError('遮罩必须是单通道二值图')
        a=np.asarray(image.convert('L'))
    if a.shape!=tuple(shape):raise ValueError('遮罩与 RGB-D 尺寸不一致')
    if not np.isin(a,[0,255]).all():raise ValueError('遮罩必须使用 0/255，黑色忽略、白色保留')
    return a>0


def frame_mask(root,frame,shape):
    return read_mask(asset(root,frame['mask']),shape) if frame.get('mask') else np.ones(shape,bool)


def find_mask(folder,image_name):
    folder=Path(folder);name=Path(image_name)
    for candidate in (folder/(str(name)+'.png'),folder/name.with_suffix('.png')):
        if candidate.is_file():return candidate
    raise ValueError('缺少图像遮罩：'+image_name)
