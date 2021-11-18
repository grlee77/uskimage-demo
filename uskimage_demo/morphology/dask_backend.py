# import cucim
import math
import numbers

import numpy as np
import uarray as ua
import dask.array as da

from skimage import morphology
from skimage.util import apply_parallel
from skimage._backend import scalar_or_array
from skimage._shared.utils import _supported_float_type

# Backend support for skimage.filters

_implemented = {}


def _implements(skimage_func):
    """Decorator adds function to the dictionary of implemented functions"""
    def inner(func):
        _implemented[skimage_func] = func
        return func

    return inner


@_implements(morphology.binary_erosion)
def binary_erosion(image, footprint=None, out=None):

    image = image.astype(bool, copy=False)

    if out is not None:
        raise ValueError("out is unsupported")

    if footprint is None:
        depth = 1
    else:
        depth = [s // 2 for s in footprint.shape]

    # handled depth and sigma above, so set channel_axis to None
    return apply_parallel(
        morphology.binary_erosion,
        image,
        depth=depth,
        extra_keywords=dict(footprint=footprint,
                            out=out),
        dtype=bool,
        channel_axis=None,
        )
