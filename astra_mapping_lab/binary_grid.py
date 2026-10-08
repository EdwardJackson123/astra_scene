"""Two-class navigation export: every non-free cell is blocked."""
import numpy as np
from grid_common import classes

VERSION='binary_block_uncertain_v1'

def convert(grid,meta):
    _,free,unknown=classes(grid,meta)
    occupied_value=255 if meta.get('negate',0) else 0
    free_value=0 if meta.get('negate',0) else 255
    result=np.where(free,free_value,occupied_value).astype(np.uint8)
    occ,free2,unknown2=classes(result,meta)
    if unknown2.any() or not np.array_equal(free,free2):raise ValueError('invalid binary grid thresholds')
    return result,dict(version=VERSION,uncertain_to_blocked=int(unknown.sum()),free_cells=int(free2.sum()),blocked_cells=int(occ.sum()),unknown_cells=0)

def export(directory,report):
    from pathlib import Path
    import yaml
    from PIL import Image
    from recording import atomic
    d=Path(directory);meta=yaml.safe_load((d/'map.yaml').read_text())
    before=np.array(Image.open(d/'map.png').convert('L'));original=np.array(Image.open(d/'original_map.png').convert('L'))
    allowed=np.array(Image.open(d/'allowed_scope.png'))>0
    result,audit=convert(before,meta);reference,_=convert(original,meta)
    old_occ,_,_=classes(reference,meta);occ,free,_=classes(result,meta)
    report['pre_binary_counts']={k:report.get(k) for k in ('changed_cells','added_obstacles','cleared_obstacles','free_to_unknown','unknown_cells_in_scope')}
    report.update(changed_cells=int((result!=reference).sum()),added_obstacles=int((occ&~old_occ).sum()),cleared_obstacles=int((old_occ&free).sum()),
        occupied_to_unknown=0,unknown_to_free=0,free_to_unknown=0,unknown_cells_in_scope=0,
        outside_allowed_unchanged=bool(np.array_equal(result[~allowed],reference[~allowed])),binary_export=audit)
    audit['outside_uncertain_to_blocked']=int((classes(original,meta)[2]&~allowed).sum())
    audit['comparison_reference']='original map with the same unknown-to-blocked policy'
    Image.fromarray(result).save(d/'map.png');Image.fromarray(reference).save(d/'original_map.png')
    meta['image']='map.png';meta['mode']='trinary'
    for name in ('map.yaml','config.yaml'):(d/name).write_text(yaml.safe_dump(meta,sort_keys=False))
    atomic(d/'binary_export.json',audit)
    return report
