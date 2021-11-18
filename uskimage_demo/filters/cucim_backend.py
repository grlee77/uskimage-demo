# import cucim
import cupy as cp
import numpy as np
import uarray as ua
from cucim.skimage import filters as _cucim_filters
from skimage._backend import scalar_or_array

from skimage import filters


_implemented = {}


def _implements(skimage_func):
    """Decorator adds function to the dictionary of implemented functions"""
    def inner(func):
        _implemented[skimage_func] = func
        return func

    return inner


@_implements(filters.gaussian)
def gaussian(image, sigma=1, output=None, mode='nearest', cval=0,
             preserve_range=False, truncate=4.0, *, channel_axis=None):
    multichannel = False
    if channel_axis is not None:
        if channel_axis == -1:
            multichannel=True
        else:
            return NotImplementedError(
                "TODO: add channel_axis support to cuCIM"
            )
    return _cucim_filters.gaussian(
        image, sigma=sigma, output=output, mode=mode, cval=cval,
        multichannel=multichannel, preserve_range=preserve_range,
        truncate=truncate)


@_implements(filters.difference_of_gaussians)
def difference_of_gaussians(image, low_sigma, high_sigma=None, *,
                            mode='nearest', cval=0, channel_axis=None,
                            truncate=4.0):
    multichannel = False
    if channel_axis is not None:
        if channel_axis == -1:
            multichannel=True
        else:
            return NotImplementedError(
                "TODO: add channel_axis support to cuCIM"
            )
    return _cucim_filters.difference_of_gaussians(
        image, low_sigma=low_sigma, high_sigma=high_sigma, mode=mode,
        cval=cval, multichannel=multichannel, truncate=truncate)


@_implements(filters.median)
def median(image, footprint=None, out=None, mode='nearest', cval=0.0,
           behavior='ndimage'):
    return _cucim_filters.median(
        image, footprint, out=out, mode=mode, cval=cval,
        behavior=behavior)
