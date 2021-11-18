from warnings import warn

import numpy as np
import uarray as ua

import skimage.filters as _skimage_filters
from skimage._shared.utils import convert_to_float, warn
from skimage._backend import scalar_or_array

from uskimage_demo.filters.cv2_backend import (
    _implemented as _filters_implemented
)
from uskimage_demo.morphology.cv2_backend import (
    _implemented as _morphology_implemented
)


# Backend support for skimage
__ua_domain__ = ['numpy.skimage.filters',
                 'numpy.skimage.morphology']
_implemented = {}
_implemented.update(_filters_implemented)
_implemented.update(_morphology_implemented)


def __ua_convert__(dispatchables, coerce):
    if coerce:
        try:
            replaced = [
                np.asarray(d.value) if d.coercible and d.type is np.ndarray
                else d.value for d in dispatchables]
        except TypeError:
            return NotImplemented
    else:
        replaced = [d.value for d in dispatchables]

    if not all(isinstance(r, np.ndarray) for r in replaced):
        return NotImplemented

    return replaced


def __ua_function__(method, args, kwargs):
    fn = _implemented.get(method, None)
    if fn is None:
        return NotImplemented
    return fn(*args, **kwargs)
