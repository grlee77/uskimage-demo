from warnings import warn

import numpy as np
import uarray as ua

import skimage.filters as _skimage_filters
from skimage._shared.utils import convert_to_float, warn
from skimage._backend import scalar_or_array

from uskimage_demo.filters.diplib_backend import (
    _implemented as _filters_implemented
)
from uskimage_demo.morphology.diplib_backend import (
    _implemented as _morphology_implemented
)


# Backend support for skimage

__ua_domain__ = ['numpy.skimage.filters',
                 'numpy.skimage.morphology']
_implemented = {}
_implemented.update(_filters_implemented)
_implemented.update(_morphology_implemented)


@ua.wrap_single_convertor
def __ua_convert__(value, dispatch_type, coerce):
    if value is None:
        return None

    if dispatch_type is np.ndarray:
        if not coerce and not isinstance(value, np.ndarray):
            return NotImplemented
        return value

    if dispatch_type is np.dtype:
        return np.dtype(value)

    if dispatch_type is scalar_or_array:
        if np.isscalar(value):
            return value
        elif not coerce and not isinstance(value, np.ndarray):
            return NotImplemented
        return value

    return value


def __ua_function__(method, args, kwargs):
    fn = _implemented.get(method, None)
    if fn is None:
        return NotImplemented
    return fn(*args, **kwargs)
