"""
The MIT License (MIT)

Copyright (c) 2017 Erik Perillo <erik.perillo@gmail.com>

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

"""
Module for data augmentation.
"""

from skimage import io
from skimage import transform as skt
from skimage import filters as skf
import numpy as np

def _get_rng(rng):
    if not isinstance(rng, (list, tuple)):
        rng = (rng, rng)
    return rng

def _rot90(arr, reps=1):
    """
    Performs 90 degrees rotation 'reps' times.
    Assumes image with shape ([n_samples, n_channels,] height, width).
    """
    for __ in range(reps%4):
        arr = arr.swapaxes(-2, -1)[..., ::-1]
    return arr

def rot90(x, y, reps=1):
    x, y = _rot90(x, reps), y if y is None else _rot90(y, reps)
    return x, y

def _hmirr(img):
    """
    Flips image horizontally.
    Assumes image with shape ([n_samples, n_channels,] height, width).
    """
    return img[..., ::-1]

def hmirr(x, y):
    x, y = _hmirr(x), y if y is None else _hmirr(y)
    return x, y

def some_of(x, y=None, ops=[]): 
    """
    Chooses one operation from ops.
    """
    op = np.random.choice(ops)
    x = op(x)
    if y is not None:
        y = op(y)
    return x, y

def _rotation(img, angle, **kwargs):
    """
    Rotates image in degrees in counter-clockwise direction.
    Assumes image in [0, 1] with shape ([n_samples, n_channels,] height, width).
    """
    img = img.swapaxes(0, 1).swapaxes(1, 2)
    img = skt.rotate(img, angle=angle, resize=False, mode="constant",
        preserve_range=True, **kwargs).astype(img.dtype)
    img = img.swapaxes(2, 1).swapaxes(1, 0)
    return img

def rotation(x, y, rng, **kwargs):
    angle = np.random.uniform(*rng)
    x = _rotation(x, angle, **kwargs)
    y = y if y is None else _rotation(y, angle, **kwargs)
    return x, y

def _shear(img, shear):
    """
    Shears image.
    Assumes image in [0, 1] with shape ([n_samples, n_channels,] height, width).
    """
    at = skt.AffineTransform(shear=shear)
    img = img.swapaxes(0, 1).swapaxes(1, 2)
    img = skt.warp(img, at)
    img = img.swapaxes(2, 1).swapaxes(1, 0)
    return img

def shear(x, y, rng, **kwargs):
    shear = np.random.uniform(*rng)
    x, y = _shear(x, shear), y if y is None else _shear(y, shear)
    return x, y

def _translation(img, transl):
    """
    Performs shift in image in dx, dy = transl.
    Assumes image in [0, 1] with shape ([n_samples, n_channels,] height, width).
    """
    at = skt.AffineTransform(translation=transl)
    img = img.swapaxes(0, 1).swapaxes(1, 2)
    img = skt.warp(img, at)
    img = img.swapaxes(2, 1).swapaxes(1, 0)
    return img

def translation(x, y, rng):
    h, w = x.shape[-2:]
    transl = (int(np.random.uniform(*rng)*w), int(np.random.uniform(*rng)*h))
    x, y = _translation(x, transl), y if y is None else _translation(y, transl)
    return x, y

def _add_noise(img, noise):
    """
    Adds noise to image.
    Assumes image in [0, 1].
    """
    img = img + noise
    return img

def add_noise(x, y, rng):
    noise = np.random.uniform(*rng, size=x.shape).astype("float32")
    x, y = _add_noise(x, noise), y
    return x, y

def _mul_noise(img, noise):
    """
    Multiplies image by a factor.
    Assumes image in [0, 1].
    """
    img = img*noise
    return img

def mul_noise(x, y, rng):
    noise = np.random.uniform(*rng)
    x, y = _mul_noise(x, noise), y
    return x, y

def _blur(img, sigma):
    """
    Applies gaussian blur to image.
    Assumes image in [0, 1] with shape ([n_samples, n_channels,] height, width).
    """
    img = img.swapaxes(0, 1).swapaxes(1, 2)
    for i in range(img.shape[-1]):
        img[..., i] = skf.gaussian(img[..., i], sigma=sigma)
    img = img.swapaxes(2, 1).swapaxes(1, 0)
    return img

def blur(x, y, rng=0.5):
    sigma = np.random.uniform(*rng)
    x, y = _blur(x, sigma), y
    return x, y

def identity(x, y):
    return x, y

def _unit_norm(img, minn, maxx, dtype="float32"):
    img = ((img - minn)/max(maxx - minn, 1)).astype(dtype)
    return img

def _unit_denorm(img, minn, maxx, dtype="float32"):
    img = (img*(maxx - minn) + minn).astype(dtype)
    return img

#mapping of strings to methods
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

def augment(xy, op_seqs, apply_on_y=False, add_iff_op=True):
    """
    Performs data augmentation on x, y sample.

    op_seqs is a list of sequences of operations.
    Each sequence must be in format (op_name, op_prob, op_kwargs).
    Example of valid op_seqs:
    [
        [
            ('identity', 1.0, {}),
        ],
        [
            ('hmirr', 1.0, {}),
            ('rot90', 1.0, {'reps': 3})
        ],
        [
            ('rotation', 0.5, {'rng': (-10, 10)}),
        ]
    ]
    ('identity' is necessary to keep the original image in the returned list.)

    add_iff_op: adds image to augm list only if some operation happened.
    """
    #list of augmented images
    augm = []

    #pre-processing x, y for augmentation
    x, y = xy
    x_minn, x_maxx, x_dtype = x.min(), x.max(), x.dtype
    x = _unit_norm(x, x_minn, x_maxx, "float32")
    if apply_on_y:
        y_minn, y_maxx, y_dtype = y.min(), y.max(), y.dtype
        y = _unit_norm(y, y_minn, y_maxx, "float32")

    #applying sequences
    for op_seq in op_seqs:
        _x, _y = x.copy(), y.copy() if apply_on_y else None

        some_op = False
        #applying sequence of operations
        for name, prob, kwargs in op_seq:
            op = OPS_MAP[name]
            if np.random.uniform(0.0, 1.0) <= prob:
                some_op = True
                _x, _y = op(_x, _y, **kwargs)

        #adding sample to augm list
        if some_op or not add_iff_op:
            _x = _unit_denorm(_x, x_minn, x_maxx, x_dtype)
            if apply_on_y:
                _y = _unit_denorm(_y, y_minn, y_maxx, y_dtype)
            augm.append((_x, _y if apply_on_y else y))

    return augm
