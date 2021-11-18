
import cupy as cp
import numpy as np
import uarray as ua

from uskimage_demo.filters.cucim_backend import (
    _implemented as _filters_implemented,
)
from uskimage_demo.morphology.cucim_backend import (
    _implemented as _morphology_implemented,
)
from uskimage_demo.metrics.cucim_backend import (
    _implemented as _metrics_implemented,
)

# Backend support for skimage.filters

__ua_domain__ = ['numpy.skimage.filters',
                 'numpy.skimage.metrics',
                 'numpy.skimage.morphology']
_implemented = {}
_implemented.update(_filters_implemented)
_implemented.update(_metrics_implemented)
_implemented.update(_morphology_implemented)


@ua.wrap_single_convertor
def __ua_convert__(value, dispatch_type, coerce):
    if value is None:
        return None

    if dispatch_type is np.ndarray:
        if not isinstance(value, cp.ndarray):
            if not coerce:
                return NotImplemented
            else:
                return cp.asarray(value)
        return value

    if dispatch_type is np.dtype:
        return np.dtype(value)

    if dispatch_type is scalar_or_array:
        if np.isscalar(value):
            return value
        elif not isinstance(value, cp.ndarray):
            if not coerce:
                return NotImplemented
            else:
                return cp.asarray(value)
        return value

    return value


def __ua_function__(method, args, kwargs):
    fn = _implemented.get(method, None)
    if fn is None:
        return NotImplemented
    return fn(*args, **kwargs)
