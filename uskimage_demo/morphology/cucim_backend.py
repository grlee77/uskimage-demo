# import cucim
import cupy as cp
import numpy as np
import uarray as ua
from cucim.skimage import morphology as _cucim_morphology

from skimage import morphology

_implemented = {}


def _implements(skimage_func):
    """Decorator adds function to the dictionary of implemented functions"""
    def inner(func):
        _implemented[skimage_func] = func
        return func

    return inner


@_implements(morphology.binary_erosion)
def binary_erosion(image, footprint=None, out=None):
    return _cucim_morphology.binary_erosion(image, footprint, out=out)


@_implements(morphology.binary_dilation)
def binary_dilation(image, footprint=None, out=None):
    return _cucim_morphology.binary_dilation(image, footprint, out=out)

