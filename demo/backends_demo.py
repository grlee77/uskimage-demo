import numpy as np

import skimage
from skimage import data, filters, img_as_float, metrics, morphology

from uskimage_demo import dask_numpy_backend
from uskimage_demo import diplib_backend

from _time_cpu import repeat


ndim = 2
if ndim == 3:
    # shape of images used for Gaussian filtering and structural_similarity
    shape = (256, 256, 128)

    # define footprints used for morphology and median filtering
    footprint = morphology.ball(3).astype(bool)
    footprint_sep = np.ones((7, 7, 7), dtype=bool)

    # define binary data for use in erosion demo
    blobs = data.binary_blobs(256, n_dim=ndim)
elif ndim == 2:
    # shape of images used for Gaussian filtering and structural_similarity
    shape = (4096, 2048)

    # define footprints used for morphology and median filtering
    footprint = morphology.disk(5).astype(bool)
    footprint_sep = np.ones((11, 11), dtype=bool)

    # define binary data for use in erosion demo
    blobs = data.binary_blobs(2048, n_dim=ndim)

float_dtype = np.float32
image = np.random.standard_normal(shape).astype(float_dtype)

# need a second image for use with structural_similarity
image2 = image + 0.02 * np.random.standard_normal(shape).astype(float_dtype)

print(f"Running with {ndim}D data:"
      f"\n\tFor gaussian, median and structural_similarity functions:"
      f"\n\t\timage.shape={shape}"
      f"\n\t\timage.dtype={image.dtype}"
      f"\n\tFor binary erosion:"
      f"\n\t\tbinary blobs shape={blobs.shape}"
      f"\n\tball footprint shape = {footprint.shape}"
      f"\n\trect footprint shape = {footprint_sep.shape}")

backend_list = ['skimage', 'dask_numpy']

have_diplib = False
try:
    import diplib
    have_diplib = True
    backend_list += ['diplib']
except ImportError:
    diplib = None
    pass

have_cucim = False
try:
    import cupy as cp
    import cucim
    from uskimage_demo import cucim_cpu_backend
    from uskimage_demo import cucim_backend
    from _time_gpu import repeat as repeat_gpu
    have_cucim = True
    backend_list += ['cucim_cpu', 'cucim']
except:
    pass


def time_functions(image, blobs, footprint, footprint_sep, image2, repeat, n_warmup=1):
    # When timing, quit running additional repetitions at either max_duration or
    # n_repeat, whichever happens first.
    max_duration = 2
    perf_kwargs = dict(n_warmup=n_warmup, n_repeat=10000, max_duration=max_duration)

    #########################
    # Time gaussian filtering
    #########################

    perf_gauss = repeat(filters.gaussian, args=(image, 3.5), **perf_kwargs)
    print(perf_gauss)

    #######################################
    # Time difference_of_gaussian filtering
    #######################################

    sigma_low = 1.5
    sigma_high = 3
    perf_dog = repeat(filters.difference_of_gaussians,
                      args=(image, sigma_low, sigma_high),
                      **perf_kwargs)
    print(perf_dog)

    #######################
    # Time median filtering
    #######################

    perf_median = repeat(filters.median, args=(image, footprint),
                         name='median (ball)', **perf_kwargs)
    print(perf_median)

    perf_median = repeat(filters.median, args=(image, footprint_sep),
                         name='median (rect)', **perf_kwargs)
    print(perf_median)

    # #####################
    # # Time binary erosion
    # #####################

    perf_binary_erosion = repeat(morphology.binary_erosion,
                                 args=(blobs, footprint),
                                 name='erosion (ball)',
                                 **perf_kwargs)
    print(perf_binary_erosion)


    perf_binary_erosion = repeat(morphology.binary_erosion,
                                 args=(blobs, footprint_sep),
                                 name='erosion (rect)',
                                 **perf_kwargs)
    print(perf_binary_erosion)


    # ############################
    # # Time structural similarity
    # ############################

    # Now an example of a function that is not currently have a multimethod
    # but does have an internal call to skimage.filters.gaussian which will
    # still be accelerated!

    perf_ssim = repeat(
        metrics.structural_similarity,
        args=(image, image2),
        kwargs=dict(sigma=1.5, gaussian_weights=True,
                    use_sample_covariance=False),
        name='structural sim.',
        **perf_kwargs
    )
    print(perf_ssim)


for backend_name in backend_list:
    if backend_name == 'dask_numpy':
        backend = dask_numpy_backend
        backend_kwargs = dict(only=True)
        print("\n\n** Timings with Dask backend enabled (CPU, scheduler='threads')**")
    elif backend_name == 'diplib':
        backend = diplib_backend
        # set only=False here since diplib backend doesn't include structural_similarity or difference_of_gaussian
        backend_kwargs = dict(only=False)
        print(f"\n\n** Timings with diplib backend enabled (threads={diplib.GetNumberOfThreads()})**")
    elif backend_name == 'cucim_cpu':
        backend = cucim_cpu_backend
        backend_kwargs = dict(only=True)
        print("\n\n** Timings with cucim_cpu backend **")
    elif backend_name == 'cucim':
        backend = cucim_backend
        backend_kwargs = dict(only=True)
        print("\n\n** Timings with cucim backend (no host/device transfer) **")
    elif backend_name == 'skimage':
        backend = 'skimage'
        backend_kwargs = dict(only=True)
        print("\n\n** Timings with default skimage backend**")
    else:
        raise ValueError(f"unkown backend: {backend_name}")

    with skimage.set_backend(backend, **backend_kwargs):
        n_warmup = 5 if 'cucim' in backend_name else 0
        if backend_name == 'cucim':
            time_functions(cp.asarray(image), cp.asarray(blobs),
                           cp.asarray(footprint), cp.asarray(footprint_sep),
                           cp.asarray(image2), repeat=repeat_gpu,
                           n_warmup=n_warmup)
        else:
            time_functions(image, blobs, footprint, footprint_sep, image2,
                           repeat=repeat, n_warmup=n_warmup)
