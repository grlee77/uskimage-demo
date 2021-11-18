import dask.array as da
import numpy as np
import pytest

from skimage import set_backend
from skimage import filters
from uskimage_demo import dask_backend

modes = ['wrap', 'reflect', 'nearest', 'mirror', 'constant']


@pytest.mark.parametrize('channel_axis', [None, 0, -1])
@pytest.mark.parametrize('coerce_input', [False, True])
@pytest.mark.parametrize('dask_input', [False, True])
@pytest.mark.parametrize('mode', modes)
def test_gaussian(channel_axis, coerce_input, dask_input, mode):
    image = np.random.randn(1024, 512)
    chunks = (256, 256)
    if channel_axis is not None:
        n_channels = 3
        image = np.stack((image,) * n_channels, axis=channel_axis)
        chunks = list(chunks)
        chunks.insert(channel_axis % image.ndim, n_channels)
        chunks = tuple(chunks)

    sigma = (1.5, 1.0)

    expected_output = filters.gaussian(
        image, sigma=sigma, mode=mode, channel_axis=channel_axis)

    if dask_input:
        image = da.asarray(image, chunks=chunks)

    only = (coerce_input == True or dask_input == True)
    with set_backend(dask_backend, coerce=coerce_input, only=only):
        out = filters.gaussian(image, mode=mode, sigma=sigma,
                               channel_axis=channel_axis)

    if dask_input or coerce_input:
        assert isinstance(out, da.Array)
        if dask_input:
            assert out.chunksize == chunks
        out = out.compute()
    assert isinstance(out, np.ndarray)
    np.testing.assert_allclose(expected_output, out)


@pytest.mark.parametrize('channel_axis', [None, 0, -1])
@pytest.mark.parametrize('coerce_input', [False, ])
@pytest.mark.parametrize('dask_input', [False, True])
@pytest.mark.parametrize('mode', ['reflect'])
def test_difference_of_gaussians(channel_axis, coerce_input, dask_input, mode):
    image = np.random.randn(1024, 512)
    chunks = (256, 256)
    if channel_axis is not None:
        n_channels = 3
        image = np.stack((image,) * n_channels, axis=channel_axis)
        chunks = list(chunks)
        chunks.insert(channel_axis % image.ndim, n_channels)
        chunks = tuple(chunks)

    low_sigma = 1.0
    high_sigma = 1.6

    expected_output = filters.difference_of_gaussians(
        image, low_sigma=low_sigma, high_sigma=high_sigma, mode=mode,
        channel_axis=channel_axis)

    if dask_input:
        image = da.asarray(image, chunks=chunks)

    only = (coerce_input == True or dask_input == True)
    with set_backend(dask_backend, coerce=coerce_input, only=only):
        out = filters.difference_of_gaussians(
            image, mode=mode, low_sigma=low_sigma, high_sigma=high_sigma,
            channel_axis=channel_axis)

    if dask_input or coerce_input:
        assert isinstance(out, da.Array)
        if dask_input:
            assert out.chunksize == chunks
        out = out.compute()
    assert isinstance(out, np.ndarray)
    np.testing.assert_allclose(expected_output, out)
