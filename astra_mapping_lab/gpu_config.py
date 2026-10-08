"""Explicit server-worker GPU selection (one worker per service)."""
import os
def gpu_index():
    value=int(os.environ.get('ASTRA_GPU_INDEX','1'))
    if value<0:raise ValueError('invalid GPU index')
    return value
