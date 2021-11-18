# import cucim
import cupy as cp
import numpy as np
import uarray as ua
from cucim.skimage import metrics as _cucim_metrics
from skimage._backend import scalar_or_array

from skimage import metrics


_implemented = {}


def _implements(skimage_func):
    """Decorator adds function to the dictionary of implemented functions"""
    def inner(func):
        _implemented[skimage_func] = func
        return func

    return inner


@_implements(metrics.structural_similarity)
def structural_similarity(im1, im2, *, win_size=None, gradient=False,
                          data_range=None, channel_axis=None,
                          multichannel=False, gaussian_weights=False,
                          full=False, **kwargs):
    if channel_axis is not None:
        if channel_axis == -1:
            multichannel=True
        else:
            return NotImplementedError(
                "TODO: add channel_axis support to cuCIM"
            )
    return _cucim_metrics.structural_similarity(
        im1, im2, win_size=win_size, gradient=gradient, data_range=data_range,
        multichannel=multichannel, gaussian_weights=gaussian_weights,
        full=full, **kwargs)
