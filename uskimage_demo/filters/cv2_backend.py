# import cucim
from warnings import warn

import numpy as np

import skimage.filters as _skimage_filters
from skimage._shared.utils import (convert_to_float,
                                   deprecate_multichannel_kwarg, warn)

try:
    import cv2
    have_opencv = True
    numpy_mode_translation_dict = dict(
        constant=cv2.BORDER_CONSTANT,
        edge=cv2.BORDER_REPLICATE,
        symmetric=cv2.BORDER_REFLECT,
        reflect=cv2.BORDER_REFLECT_101,
        wrap=cv2.BORDER_WRAP,)

    ndi_mode_translation_dict = dict(
        constant=cv2.BORDER_CONSTANT,
        nearest=cv2.BORDER_REPLICATE,
        reflect=cv2.BORDER_REFLECT,
        mirror=cv2.BORDER_REFLECT_101,
        wrap=cv2.BORDER_WRAP,)

except ImportError:
    have_opencv = False
    ndi_mode_translation_dict = {}
    numpy_mode_translation_dict = {}


def _to_cv2_mode(mode, cval=0):
    """Convert from skimage mode name to the corresponding ndimage mode."""
    if mode not in ndi_mode_translation_dict:
        raise NotImplementedError(f"cv2 does not support mode {mode}")

    if mode == 'constant' and cval != 0.:
        raise NotImplementedError(
            "cv2 backend only supports cval=0 for 'constant' mode"
        )

    return ndi_mode_translation_dict[mode]


# Backend support for skimage.filters

__ua_domain__ = 'numpy.skimage.filters'
_implemented = {}


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

    # may need warnings or errors related to API changes here
    #if 'multichannel' in kwargs and not _skimage_1_0:
    #    warnings.warn('The \'multichannel\' argument is not supported for scikit-image >= 1.0')
    return fn(*args, **kwargs)


def _implements(skimage_func):
    """Decorator adds function to the dictionary of implemented functions"""
    def inner(func):
        _implemented[skimage_func] = func
        return func

    return inner


def _cv2_check_ndim(image, channel_axis=None):
    if image.ndim != (2 + int(channel_axis is not None)):
        return False


@_implements(_skimage_filters.gaussian)
@deprecate_multichannel_kwarg(multichannel_position=5)
def gaussian(image, sigma=1, output=None, mode='nearest', cval=0,
             multichannel=False, preserve_range=False, truncate=4.0, *,
             channel_axis=None):

    if not _cv2_check_ndim(image, channel_axis):
        # cv2 only supports 2D images (or 2D + channels)
        return NotImplemented

    opencv_mode = _to_cv2_mode(mode, cval)
    if not have_opencv:
        raise ImportError("OpenCV (cv2) is unavailable.")

    if mode == 'wrap':
        # mode='wrap' is unsupported by cv2.GaussianBlur
        return NotImplemented

    if np.isscalar(sigma):
        sigma = (sigma, sigma)
    elif len(sigma) != 2:
        raise ValueError(
            "sigma must be a scalar or a sequence of length image.ndim"
        )
    if np.isscalar(truncate):
        truncate = (truncate, truncate)
    elif len(truncate) != 2:
        raise ValueError(
            "truncate must be a scalar or a sequence of length image.ndim"
        )

    if channel_axis is not None:
        # For OpenCV the channels must be on the last axis
        image = np.moveaxis(image, source=channel_axis, destination=-1)

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
    if output is None:
        output = np.empty_like(image)
    elif output.dtype not in [np.float32, np.float64]:
        raise ValueError(
            "Provided output data type must be np.float32 or np.float64."
        )

    """
    According to the cv2.GaussianFilter docstring, the image can have any
    number of channels, which are processed independently. Depth must be
    one of CV_8U, CV_16U, CV_16S, CV_32F or CV_64F.
    """

    # determine ksize from sigma & truncate
    # the equation used is from scipy.ndimage.gaussian_filter1d
    wx = (2 * int(truncate[1] * sigma[1] + 0.5) + 1)
    wy = (2 * int(truncate[0] * sigma[0] + 0.5) + 1)

    cv2.GaussianBlur(
        image,
        dst=output,
        ksize=(wx, wy),
        sigmaX=sigma[1],
        sigmaY=sigma[0],
        borderType=opencv_mode)

    if channel_axis is not None:
        output = np.moveaxis(output, source=-1, destination=channel_axis)
    return output

gaussian.__doc__ = _skimage_filters.gaussian.__doc__
