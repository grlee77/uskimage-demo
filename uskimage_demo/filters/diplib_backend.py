from warnings import warn

import numpy as np
import uarray as ua

import skimage.filters as _skimage_filters
from skimage._shared.utils import convert_to_float, warn
from skimage._backend import scalar_or_array

try:
    import diplib as dip
    have_diplib = True
    ndi_mode_translation_dict = dict(
        constant='add zeros',
        nearest='zero order',
        mirror='mirror',
        wrap='periodic')

except ImportError:
    have_diplib = False
    ndi_mode_translation_dict = {}
    numpy_mode_translation_dict = {}

_implemented = {}


def _to_diplib_mode(mode, cval=0):
    """Convert from skimage mode name to the corresponding ndimage mode."""
    if mode not in ndi_mode_translation_dict:
        # warnings.warn(f"diplib does not support mode {mode}")
        return NotImplemented

    if mode == 'constant' and cval != 0.:
        # warnings.warn(f"diplib backend only supports cval=0 for 'constant' mode")
        return NotImplemented

    return ndi_mode_translation_dict[mode]


def _implements(skimage_func):
    """Decorator adds function to the dictionary of implemented functions"""
    def inner(func):
        _implemented[skimage_func] = func
        return func
    return inner


@_implements(_skimage_filters.gaussian)
def gaussian(image, sigma=1, output=None, mode='nearest', cval=0,
             preserve_range=False, truncate=4.0, *, channel_axis=None):

    diplib_mode = _to_diplib_mode(mode, cval)
    if diplib_mode == NotImplemented:
        return NotImplemented
    if not have_diplib:
        raise ImportError("PyDIP (DIPlib) is unavailable.")

    ndim_spatial = image.ndim if channel_axis is None else image.ndim - 1
    if np.isscalar(sigma):
        sigma = (sigma, ) * ndim_spatial
    elif len(sigma) != ndim_spatial:
        raise ValueError(
            "sigma must be a scalar or a sequence equal to the image.ndim "
            "(or image.ndim - 1 in the case of multichannel input)"
        )

    truncate = np.unique(truncate)
    if not len(truncate) == 1:
        # raise NotImplementedError("only scalar truncate is supported")
        return NotImplemented
    truncate = truncate[0]

    if channel_axis is not None:
        if channel_axis < 0:
            channel_axis += image.ndim
        if channel_axis != image.ndim - 1:
            image = np.moveaxis(image, source=channel_axis, destination=-1)
        sigma = sigma[:ndim_spatial]
    sigma = list(sigma)

    # special handling copied from skimage.filters.Gaussian
    if image.ndim == 3 and image.shape[-1] == 3 and channel_axis is None:
        msg = ("Images with dimensions (M, N, 3) are interpreted as 2D+RGB "
               "by default. Use `channel_axis=None` to interpret as "
               "3D image with last dimension of length 3.")
        warn(RuntimeWarning(msg))
        channel_axis = -1
    if any(s < 0 for s in sigma):
        raise ValueError("Sigma values less than zero are not valid")
    image = convert_to_float(image, preserve_range)
    # TODO: try removing this
    if output is None:
        output = np.empty_like(image)
    elif output.dtype not in [np.float32, np.float64]:
        raise ValueError(
            "Provided output data type must be np.float32 or np.float64."
        )

    # TODO: debug why multichannel case doesn't work currently (2d grayscale was okay)
    # TODO: debug why channel_axis input to this multimethod gets ignored?
    output[...] = dip.Gauss(
        image,
        sigmas=sigma[::-1],  # reversed?
        method='FIR',
        # derivativeOrder=[0] * ndim_spatial,
        boundaryCondition=[diplib_mode] * ndim_spatial,
        truncation=truncate)
    if channel_axis is not None and channel_axis != image.ndim - 1:
        output = np.moveaxis(output, source=-1, destination=channel_axis)
    return output

gaussian.__doc__ = _skimage_filters.gaussian.__doc__


# Note: do not have to define a multimethod for difference_of_gaussian
#       (it will reuse the gaussian multimethod already defined)


@_implements(_skimage_filters.median)
def median(image, footprint=None, out=None, mode='nearest', cval=0.0,
           behavior='ndimage'):

    diplib_mode = _to_diplib_mode(mode, cval)
    if diplib_mode == NotImplemented:
        return NotImplemented
    if not have_diplib:
        raise ImportError("PyDIP (DIPlib) is unavailable.")
    footprint = np.asarray(footprint, dtype=bool)
    if footprint.ndim > 2:
        # if np.all(footprint):
        #     # TODO: have to reverse order of elements in footprint.shape
        #     footprint = dip.Kernel(footprint.shape[::-1], 'rectangular')

        # have to specify , None in the call below to avoid any axis being
        # interpreted as a tensor_axis.
        footprint = dip.Image(footprint, None)

    # TODO: try removing this
    if out is not None:
        return NotImplemented

    if out is None:
        out = np.empty_like(image)
    elif out.dtype not in [np.float32, np.float64]:
        raise ValueError(
            "Provided output data type must be np.float32 or np.float64."
        )

    # TODO: debug why multichannel case doesn't work currently (2d grayscale was okay)
    # TODO: debug why channel_axis input to this multimethod gets ignored?
    out[...] = dip.MedianFilter(
        image,
        kernel=footprint,
        boundaryCondition=[diplib_mode] * image.ndim)
    return out


median.__doc__ = _skimage_filters.median.__doc__

