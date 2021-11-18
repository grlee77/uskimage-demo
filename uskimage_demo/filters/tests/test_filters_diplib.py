import dask.array as da
import numpy as np
import pytest

from skimage import filters, set_backend
from skimage.morphology import footprints
from uskimage_demo import diplib_backend


modes = ['wrap', 'reflect', 'nearest', 'constant', 'mirror']
         # note:  'reflect' unsupported by diplib, falls back to skimage


@pytest.mark.parametrize('channel_axis', [None, 0, -1])
@pytest.mark.parametrize('mode', modes)
@pytest.mark.parametrize('ndim', [2, 3])
def test_gaussian(channel_axis, mode, ndim):
    if ndim == 2:
        image = np.random.randn(256, 64)
        sigma = (1.5, 1.0)
    elif ndim == 3:
        image = np.random.randn(64, 48, 32)
        sigma = (1.5, 1.0, 2)

    if channel_axis is not None:
        n_channels = 3
        image = np.stack((image,) * n_channels, axis=channel_axis)

    expected_output = filters.gaussian(
        image, sigma=sigma, mode=mode, channel_axis=channel_axis)

    only = mode != 'reflect'  # reflect unimplemented in diplib
    with set_backend(diplib_backend, coerce=False, only=only):
        out = filters.gaussian(image, mode=mode, sigma=sigma,
                               channel_axis=channel_axis)
    assert isinstance(out, np.ndarray)
    np.testing.assert_allclose(expected_output, out)


@pytest.mark.parametrize('mode', modes)
@pytest.mark.parametrize('ndim', [2, 3])
@pytest.mark.parametrize('footprint_type', ['rect', 'ball'])
def test_median(mode, ndim, footprint_type):
    if ndim == 2:
        image = np.random.randn(256, 64)
        if footprint_type == 'rect':
            footprint = np.ones((3, 5))
        elif footprint_type == 'ball':
            footprint = footprints.disk(2)
    elif ndim == 3:
        image = np.random.randn(64, 48, 32)
        if footprint_type == 'rect':
            footprint = np.ones((3, 5, 7))
        elif footprint_type == 'ball':
            footprint = footprints.ball(2)
    expected_output = filters.median(image, footprint, mode=mode)

    only = mode != 'reflect'  # reflect unimplemented in diplib
    with set_backend(diplib_backend, coerce=False, only=only):
        out = filters.median(image, footprint, mode=mode)
    assert isinstance(out, np.ndarray)
    np.testing.assert_allclose(expected_output, out)
