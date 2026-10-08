"""Offline map geometry and candidate viewpoints. No robot control dependencies."""
import math
import numpy as np


def finite(value, name, low=None, high=None):
    if isinstance(value, bool):
        raise ValueError(f'{name} 必须是数字')
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise ValueError(f'{name} 必须是数字')
    if not math.isfinite(value) or (low is not None and value < low) or (high is not None and value > high):
        raise ValueError(f'{name} 超出范围')
    return value


def normalize_metadata(meta, width, height):
    if not isinstance(meta, dict):
        raise ValueError('地图元数据必须是对象')
    origin = meta.get('origin')
    if not isinstance(origin, (list, tuple)) or len(origin) != 3:
        raise ValueError('origin 必须包含 x、y、yaw 三个值')
    free = finite(meta.get('free_thresh', .196), 'free_thresh', 0, 1)
    occupied = finite(meta.get('occupied_thresh', .65), 'occupied_thresh', 0, 1)
    if free >= occupied:
        raise ValueError('free_thresh 必须小于 occupied_thresh')
    negate = meta.get('negate', 0)
    if negate not in (0, 1):
        raise ValueError('negate 必须为 0 或 1')
    mode = meta.get('mode', 'trinary')
    if mode != 'trinary':
        raise ValueError('首版只支持 trinary 栅格，暂不支持 scale/raw 模式')
    return dict(width=width, height=height, resolution=finite(meta.get('resolution'), 'resolution', .001, 5),
                origin=[finite(v, 'origin') for v in origin], frame_id=str(meta.get('frame_id', 'map'))[:80],
                free_thresh=free, occupied_thresh=occupied, negate=int(negate), mode='trinary')


def pixel_to_world(x, y, meta):
    # Continuous image-edge coordinates. Cell centres are (col+.5, row+.5).
    u, v = x * meta['resolution'], (meta['height'] - y) * meta['resolution']
    ox, oy, yaw = meta['origin']; c, s = math.cos(yaw), math.sin(yaw)
    return [ox + c*u - s*v, oy + s*u + c*v]


def world_to_pixel(x, y, meta):
    ox, oy, yaw = meta['origin']; c, s = math.cos(yaw), math.sin(yaw)
    dx, dy = x-ox, y-oy
    return [(c*dx+s*dy)/meta['resolution'], meta['height']-(-s*dx+c*dy)/meta['resolution']]


def rectangle(value, width, height):
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError('选区需要 [x0,y0,x1,y1]')
    x0,y0,x1,y1 = [finite(v, '选区坐标') for v in value]
    x0,x1 = sorted((x0,x1)); y0,y1 = sorted((y0,y1))
    if x0 < 0 or y0 < 0 or x1 > width or y1 > height or x1-x0 < 2 or y1-y0 < 2:
        raise ValueError('请在地图内框选至少 2×2 像素的区域')
    return [x0,y0,x1,y1]


def polygon(rect, meta):
    x0,y0,x1,y1 = rect
    return [pixel_to_world(x,y,meta) for x,y in [(x0,y0),(x1,y0),(x1,y1),(x0,y1)]]


def free_mask(image, meta):
    rgba = np.asarray(image.convert('RGBA'))
    gray = np.mean(rgba[:,:,:3].astype(np.float32), axis=2)/255
    occupancy = gray if meta['negate'] else 1-gray
    return (occupancy < meta['free_thresh']) & (rgba[:,:,3] == 255)


def clearance_mask(free, radius_cells):
    # Conservative square footprint; map exterior, occupied and unknown are blocked.
    r = int(radius_cells); h,w = free.shape
    if 2*r+1>min(h,w):return np.zeros_like(free,dtype=bool)
    blocked = np.pad((~free).astype(np.int32), ((r,r),(r,r)), constant_values=1)
    summed = np.pad(blocked, ((1,0),(1,0))).cumsum(0).cumsum(1)
    size = 2*r+1
    counts = summed[size:size+h,size:size+w]-summed[:h,size:size+w]-summed[size:size+h,:w]+summed[:h,:w]
    return counts == 0


def make_plan(image, meta, rect, margin_m=.8, clearance_m=.35):
    rect = rectangle(rect, meta['width'], meta['height'])
    margin_m = finite(margin_m, '上下文扩展', .1, 5)
    clearance_m = finite(clearance_m, '预览净空', .05, 2)
    x0,y0,x1,y1 = rect; res = meta['resolution']; pad = margin_m/res
    context = [max(0,x0-pad), max(0,y0-pad), min(meta['width'],x1+pad), min(meta['height'],y1+pad)]
    free = free_mask(image, meta); safe = clearance_mask(free, math.ceil(clearance_m/res))
    # Sample four sides outside ROI. These are independent proposals, never a route.
    points=[]; spacing=max(.6,2*clearance_m)/res
    distance=max(.5,clearance_m+.15)/res
    samples=[]
    for fraction in [.5,.2,.8,0,1]:
        x=x0+(x1-x0)*fraction; y=y0+(y1-y0)*fraction
        samples.extend([(x,y0-distance,x,y0),(x1+distance,y,x1,y),
                        (x,y1+distance,x,y1),(x0-distance,y,x0,y)])
    for x,y,tx,ty in samples:
        col,row=int(math.floor(x)),int(math.floor(y))
        if not (0<=col<meta['width'] and 0<=row<meta['height'] and safe[row,col]):continue
        px,py=col+.5,row+.5
        if any(math.hypot(px-p['pixel'][0],py-p['pixel'][1])<spacing for p in points):continue
        # Check static line segment up to, not inside, the requested region.
        count=max(2,int(math.hypot(tx-px,ty-py)*2))
        xs=np.linspace(px,tx,count,endpoint=False).astype(int).clip(0,meta['width']-1)
        ys=np.linspace(py,ty,count,endpoint=False).astype(int).clip(0,meta['height']-1)
        if not free[ys,xs].all():continue
        world=pixel_to_world(px,py,meta); target=pixel_to_world(tx,ty,meta)
        points.append(dict(id=f'view-{len(points)+1}',pixel=[px,py],target_pixel=[tx,ty],
                           world_xy=world,yaw=math.atan2(target[1]-world[1],target[0]-world[0]),
                           status='STATIC_CANDIDATE_ONLY',reachability='NOT_CHECKED'))
        if len(points)>=12:break
    roi=free[int(math.floor(y0)):int(math.ceil(y1)),int(math.floor(x0)):int(math.ceil(x1))]
    return dict(roi_pixels=rect,roi_world_polygon=polygon(rect,meta),context_pixels=context,
                context_world_polygon=polygon(context,meta),replacement_region=None,
                area_m2=(x1-x0)*(y1-y0)*res**2,free_fraction=float(roi.mean()),
                context_margin_m=margin_m,preview_clearance_m=clearance_m,viewpoints=points,
                route=None,execution_allowed=False,geometry_confirmed=False,
                warnings=['候选点只通过静态栅格净空筛选，未验证起点可达性、实时障碍或三维可见性。',
                          '预览净空是实验参数，尚未核实机器人 footprint。',
                          '框选范围不是实际替换范围；本阶段不会采集、重建或修改地图。']+
                         ([] if points else ['没有找到符合静态筛选条件的观察点，请调整选区或等待后续人工规划。']))
