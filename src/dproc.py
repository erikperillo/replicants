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
Module for data processing.
"""

from skimage import io
from skimage import transform as transf
from skimage import color
from skimage import img_as_float

import numpy as np

import glob
import os

def _load_salicon(fp_or_fps):
    fn = os.path.basename(fp_or_fps)
    dn = os.path.dirname(os.path.dirname(fp_or_fps))
    x_fp = os.path.join(dn, "images", fn)
    y_fp = os.path.join(dn, "maps", fn)
    x = io.imread(x_fp)
    y = io.imread(y_fp)
    x = x.swapaxes(2, 1).swapaxes(1, 0)
    y = y.reshape((1, ) + y.shape)
    return x, y

def _load_judd(fp_or_fps):
    fn = os.path.basename(fp_or_fps)
    dn = os.path.dirname(os.path.dirname(fp_or_fps))
    x_fp = os.path.join(dn, "stimuli", fn)
    y_fp = glob.glob(os.path.join(dn, "fixmaps", fn.split(".")[0] + "*"))[0]
    x = io.imread(x_fp)
    y = io.imread(y_fp)
    if x.ndim < 3:
        x = np.dstack(3*(x, ))
    x = x.swapaxes(2, 1).swapaxes(1, 0)
    y = y.reshape((1, ) + y.shape)
    return x, y

def load(fp):
    return _load_judd(fp)

def _pre_proc(x, y=None):
    x = x.swapaxes(0, 1).swapaxes(1, 2)
    y = y.swapaxes(0, 1).swapaxes(1, 2)

    #converting images to float
    x = img_as_float(x)
    y = img_as_float(y)

    #reshaping to fix max input shape if necessary
    h, w = x.shape[:2]
    max_h, max_w = (480, 640)
    max_ratio = max(h/max_h, w/max_w)
    #if max_ratio > 1.0:
    #    x = transf.rescale(x, 1/max_ratio, mode="constant")#.astype("uint8")
    #    y = transf.rescale(y, 1/max_ratio, mode="constant")#.astype("uint8")
    x = transf.resize(x, (max_h, max_w), mode="constant")#.astype("uint8")
    y = transf.resize(y, (max_h, max_w), mode="constant")#.astype("uint8")

    #converting x colorspace to LAB
    x = color.rgb2lab(x)

    #preparing shapes to be divisable by 2^3
    h, w = x.shape[:2]
    x = x[h%8:, w%8:]
    y = y[h%8:, w%8:]
    #reshaping y
    y = transf.resize(y, (h//8, w//8), mode="constant")
    #normalizing y
    y = (y - y.min())/max(y.max() - y.min(), 1e-6)

    #normalizing each x channel
    for i in range(3):
        x[..., i] = (x[..., i] - x[..., i].mean())/x[..., i].std()

    x = x.swapaxes(2, 1).swapaxes(1, 0)
    y = y.swapaxes(2, 1).swapaxes(1, 0)

    return x, y

def pre_proc(batch_xy):
    for i, xy in enumerate(batch_xy):
        batch_xy[i] = _pre_proc(*xy)
    return batch_xy
