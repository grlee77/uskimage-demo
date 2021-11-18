import dask.array as da
import numpy as np
import pytest

from skimage import data, morphology, set_backend
from uskimage_demo import diplib_backend
from skimage.morphology import footprints

modes = ['wrap', 'reflect', 'nearest', 'constant', 'mirror']
         # note:  'reflect' unspported by diplib, falls back to skimage


@pytest.mark.parametrize('ndim', [2, 3])
@pytest.mark.parametrize('footprint_type', ['rect', 'ball'])
@pytest.mark.parametrize('morph_func', [morphology.binary_dilation,
                                        morphology.binary_erosion])
def test_binary_erosion_and_dilation(ndim, footprint_type, morph_func):
    if ndim == 2:
        image = data.binary_blobs(256, n_dim=2)[:, :64]
        if footprint_type == 'rect':
            footprint = np.ones((3, 5))
        elif footprint_type == 'ball':
            footprint = footprints.disk(2)
    elif ndim == 3:
        image = data.binary_blobs(64, n_dim=3)[:, :48, :32]
        if footprint_type == 'rect':
            footprint = np.ones((3, 5, 7))
        elif footprint_type == 'ball':
            footprint = footprints.ball(2)

    expected_output = morph_func(image, footprint)

    with set_backend(diplib_backend, coerce=False, only=True):
        out = morph_func(image, footprint)
    assert isinstance(out, np.ndarray)
    np.testing.assert_allclose(expected_output, out)
