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

import util

def _unit_norm(x, eps=1e-6):
    return (x - x.min())/max(x.max() - x.min(), eps)

def _load_salicon(fp_or_fps):
    fn = os.path.basename(fp_or_fps)
    dn = os.path.dirname(os.path.dirname(fp_or_fps))
    x_fp = os.path.join(dn, "images", fn)
    y_fp = os.path.join(dn, "maps", fn)
    x = io.imread(x_fp)
    y = io.imread(y_fp)
    if x.ndim < 3:
        x = np.dstack(3*(x, ))
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

def train_load(fp):
    #return _load_judd(fp)
    return _load_salicon(fp)

def infer_load(fp):
    #return _load_judd(fp)
    return _load_judd(fp)

def _pre_proc(x, y=None, resize=False):
    x = x.swapaxes(0, 1).swapaxes(1, 2)
    #converting images to float
    x = img_as_float(x)

    if y is not None:
        y = y.swapaxes(0, 1).swapaxes(1, 2)
        y = img_as_float(y)

    #reshaping to fix max input shape if necessary
    h, w = x.shape[:2]
    max_h, max_w = (240, 320)
    max_ratio = max(h/max_h, w/max_w)
    if resize:
        x = transf.resize(x, (max_h, max_w), mode="constant")
        if y is not None:
            y = transf.resize(y, (max_h, max_w), mode="constant")
    elif max_ratio > 1.0:
        x = transf.rescale(x, 1/max_ratio, mode="constant")
        if y is not None:
            y = transf.rescale(y, 1/max_ratio, mode="constant")

    #converting x colorspace to LAB
    x = color.rgb2lab(x)

    #preparing shapes to be divisable by 2^3
    h, w = x.shape[:2]
    x = x[h%8:, w%8:]
    if y is not None:
        y = y[h%8:, w%8:]
        #reshaping y
        y = transf.resize(y, (h//8, w//8), mode="constant")
        #normalizing y
        y = _unit_norm(y)

    #normalizing each x channel
    for i in range(3):
        x[..., i] = (x[..., i] - x[..., i].mean())/x[..., i].std()

    x = x.swapaxes(2, 1).swapaxes(1, 0).astype("float32")
    if y is not None:
        y = y.swapaxes(2, 1).swapaxes(1, 0).astype("float32")

    if y is not None:
        return x, y
    return x

def train_pre_proc(batch_xy):
    for i, xy in enumerate(batch_xy):
        batch_xy[i] = _pre_proc(*xy, resize=True)
    return batch_xy

def infer_pre_proc(x):
    return _pre_proc(x, resize=False)

def infer_save_x(x, preds_dir, name):
    fp = util.uniq_path(preds_dir, name + "_x", ext=".png")
    x = np.moveaxis(x, 0, -1)
    print("x shp...", x.shape)
    io.imsave(fp, x.clip(0, 255).astype("uint8"))

def infer_save_y_pred(y_pred, preds_dir, name):
    fp = util.uniq_path(preds_dir, name + "_y-pred", ext=".png")
    y_pred = transf.rescale(y_pred, 8, mode="constant")
    y_pred = 255*_unit_norm(y_pred)
    io.imsave(fp, y_pred.clip(0, 255).astype("uint8"))

def infer_save_y_true(y_true, preds_dir, name):
    fp = util.uniq_path(preds_dir, name + "_y-true", ext=".png")
    y_true = 255*_unit_norm(y_true)
    y_true = y_true.reshape(y_true.shape[1:])
    io.imsave(fp, y_true.clip(0, 255).astype("uint8"))
