#!/usr/bin/env python
import sys

from setuptools import setup, find_packages


setup(
    name="uskimage_demo",
    version="0.1",
    description="Example uarray-based scikit-image backends",
    author="Gregory Lee",
    author_email="grlee77@gmail.com",
    url="https://github.com/grlee77/skimage-backends-demo",
    packages=find_packages(),
    zip_safe=False,
)
