"""CuPy backend operating on NumPy arrays

__ua_domain__ and the _implemented dict is reused from cucim_backend.

__ua_convert__ differs in that the inputs are expected to be NumPy arrays which
will then be transferred to the GPU.

__ua_func__ differs in that it transfers outputs that are on the GPU back to
the host.
"""

import cupy as cp
import numpy as np
import uarray as ua

from .cucim_backend import __ua_domain__, _implemented


@ua.wrap_single_convertor
def __ua_convert__(value, dispatch_type, coerce):
    if value is None:
        return None

    if dispatch_type is np.ndarray:
        if not isinstance(value, np.ndarray) and not coerce:
            return NotImplemented
        return cp.asarray(value)

    if dispatch_type is np.dtype:
        return np.dtype(value)

    if dispatch_type is scalar_or_array:
        if np.isscalar(value):
            return value
        elif not isinstance(value, np.ndarray) and not coerce:
            return NotImplemented
        return cp.asarray(value)

    return value

def __ua_function__(method, args, kwargs):
    fn = _implemented.get(method, None)
    if fn is None:
        return NotImplemented

    out = fn(*args, **kwargs)

    # transfer device arrays back to the host
    if isinstance(out, cp.ndarray):
        cpu_out = cp.asnumpy(out)
    else:
        cpu_out = []
        for o in out:
            if isinstance(o, cp.ndarray):
                cpu_out.append(cp.asnumpy(o))
            else:
                cpu_out.append(o)
    # return CPU arrays
    return cpu_out
