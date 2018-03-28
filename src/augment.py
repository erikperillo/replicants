"""
The MIT License (MIT)

Copyright (c) 2017, 2018 Erik Perillo <erik.perillo@gmail.com>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
IN THE SOFTWARE.
"""

from skimage import io
from skimage import transform as skt
from skimage import filters as skf
import random
import numpy as np

"""
Module for data augmentation.
"""


# filling strategy for when a transformation leaves blank pixel spaces
_FILL_MODE = "constant"


def _hwc_to_chw(x):
    """
    Converts image from ([...,] height, width, channels) to
    (channels, height, width).
    """
    if x.ndim < 3:
        return x
    return x.swapaxes(2, 1).swapaxes(1, 0)


def _chw_to_hwc(x):
    """
    Converts image from ([...,] channels, height, width) to
    (height, width, channels).
    """
    if x.ndim < 3:
        return x
    return x.swapaxes(0, 1).swapaxes(1, 2)


def _get_rng(rng):
    """
    If rng not in range format, converts it to range format.
    """
    if not isinstance(rng, (list, tuple)):
        rng = (rng, rng)
    return rng


def _rot90(x, reps=1):
    """
    Performs 90 degrees rotation 'reps' times.
    Assumes image with shape ([n_samples, n_channels,] height, width).
    """
    for __ in range(reps % 4):
        x = x.swapaxes(-2, -1)[..., ::-1]
    return x


def rot90(x, y, reps):
    """
    Wrapper for _rot90 for xy pair.
    reps is number of repetitions for 90 degrees rotation (clockwise).
    """
    reps = random.randint(*_get_rng(reps))
    return _rot90(x, reps), _rot90(y, reps)


def _hmirr(x):
    """
    Flips image horizontally.
    Assumes image with shape ([n_samples, n_channels,] height, width).
    """
    return x[..., ::-1]


def hmirr(x, y):
    """
    Wrapper for _hmirr for xy pair.
    """
    return _hmirr(x), _hmirr(y)


def _rotation(x, angle, **kwargs):
    """
    Rotates image in degrees in counter-clockwise direction.
    Assumes image in [0, 1] with shape
    ([n_samples, n_channels,] height, width).
    """
    x = _chw_to_hwc(x)
    x = skt.rotate(x, angle=angle, resize=False, mode=_FILL_MODE,
                   preserve_range=True, **kwargs).astype(x.dtype)
    x = _hwc_to_chw(x)
    return x


def rotation(x, y, angle, **kwargs):
    """
    Wrapper for _rotation for xy pair.
    angle is rotation angle in degrees (counter-clockwise).
    """
    angle = random.uniform(*_get_rng(angle))
    return _rotation(x, angle, **kwargs), _rotation(y, angle, **kwargs)


def _shear(x, angle):
    """
    Shears image.
    Assumes image in [0, 1] with shape
    ([n_samples, n_channels,] height, width).
    """
    at = skt.AffineTransform(shear=angle)
    x = _chw_to_hwc(x)
    x = skt.warp(x, at, mode=_FILL_MODE)
    x = _hwc_to_chw(x)
    return x


def shear(x, y, angle, **kwargs):
    """
    Wrapper for _shear for xy pair.
    angle is shear angle in radians (counter-clockwise).
    """
    angle = random.uniform(*_get_rng(angle))
    return _shear(x, angle, **kwargs), _shear(y, angle, **kwargs)


def _translation(x, transl):
    """
    Performs shift in image in dx, dy = transl (in pixels).
    Assumes image in [0, 1] with shape
    ([n_samples, n_channels,] height, width).
    """
    at = skt.AffineTransform(translation=transl)
    x = _chw_to_hwc(x)
    x = skt.warp(x, at, mode=_FILL_MODE)
    x = _hwc_to_chw(x)
    return x


def translation(x, y, transl):
    """
    Wrapper for _translation for xy pair.
    transl is range to translade in xy (a fraction ranging in [0, 1]).
    """
    transl_x = int(random.uniform(*_get_rng(transl))*x.shape[-1])
    transl_y = int(random.uniform(*_get_rng(transl))*x.shape[-2])
    transl = (transl_x, transl_y)
    return _translation(x, transl), _translation(y, transl)


def _add_noise(x, noise, clip=True):
    """
    Adds noise to image.
    Assumes image in [0, 1].
    """
    x = x + noise
    if clip:
        x = x.clip(0, 1)
    return x


def add_noise(x, y, noise, clip=True):
    """
    Wrapper for _add_noise for xy pair.
    """
    noise = np.random.uniform(*_get_rng(noise), size=x.shape[-2:])
    return _add_noise(x, noise, clip), y


def _mul_noise(x, noise, clip=True):
    """
    Multiplies image by a factor.
    Assumes image in [0, 1].
    """
    x = x*noise
    if clip:
        x = x.clip(0, 1)
    return x


def mul_noise(x, y, noise, clip=True):
    """
    Wrapper for _mul_noise for xy pair.
    """
    noise = random.uniform(*_get_rng(noise))
    return _mul_noise(x, noise, clip), y


def _blur(x, sigma):
    """
    Applies gaussian blur to image.
    Assumes image in [0, 1] with shape
    ([n_samples,] n_channels, height, width).
    """
    x = _chw_to_hwc(x)
    for i in range(x.shape[-1]):
        x[..., i] = skf.gaussian(x[..., i], sigma=sigma)
    x = _hwc_to_chw(x)
    return x


def blur(x, y, sigma):
    """
    Wrapper for _blur for xy pair.
    """
    sigma = random.uniform(*_get_rng(sigma))
    return _blur(x, sigma), y


def _identity(x):
    """
    Identity function.
    """
    return x


def identity(x, y):
    """
    Wrapper for _identity for xy pair.
    """
    return _identity(x), _identity(y)


def _unit_norm(x, minn, maxx, dtype="float32", epslon=1e-6):
    """
    Unit normalization.
    """
    x = ((x - minn)/max(maxx - minn, epslon)).astype(dtype)
    return x


def _unit_denorm(x, minn, maxx, dtype="float32"):
    """
    Unit de-normalization.
    """
    x = (x*(maxx - minn) + minn).astype(dtype)
    return x


# mapping of strings to methods
OPS_MAP = {
    "rot90": rot90,
    "rotation": rotation,
    "shear": shear,
    "translation": translation,
    "add_noise": add_noise,
    "mul_noise": mul_noise,
    "blur": blur,
    "identity": identity,
    "hmirr": hmirr,
}


def augment(xy, operations, copy_xy=False):
    """
    Performs data augmentation on x, y sample.

    op_seqs is a list of sequences of operations.
    Each sequence must be in format (op_name, op_kwargs, op_prob).
    Example of valid op_seqs:
    [
        ('hmirr', {}, 0.5),
        ('rot90', {'reps': 3}, 1.0)
        ('rot90', {'reps': (1, 2)}, 0.3)
    ]
    The arguments might be a range (e.g. (-10, 10) for angle will produce
    a uniformly random number between -10 and 10).

    Use copy_xy=True if you want to preserve original xy.
    """
    x, y = xy
    if copy_xy:
        x, y = x.copy(), y.copy()

    # pre-processing x, y for augmentation
    x_minn, x_maxx, x_dtype = x.min(), x.max(), x.dtype
    x = _unit_norm(x, x_minn, x_maxx, "float32")
    y_minn, y_maxx, y_dtype = y.min(), y.max(), y.dtype
    y = _unit_norm(y, y_minn, y_maxx, "float32")

    # randomly applying operations
    for op_name, op_kwargs, op_prob in operations:
        if random.uniform(0, 1) <= op_prob:
            x, y = OPS_MAP[op_name](x, y, **op_kwargs)

    # de-processing
    x = _unit_denorm(x, x_minn, x_maxx, x_dtype)
    y = _unit_denorm(y, y_minn, y_maxx, y_dtype)

    return x, y
