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

def _std_norm(x, eps=1e-6):
    return (x - x.mean())/max(x.std(), eps)

def _hwc_to_chw(img):
    if img.ndim < 3:
        return img
    return img.swapaxes(2, 1).swapaxes(1, 0)

def _chw_to_hwc(img):
    if img.ndim < 3:
        return img
    return img.swapaxes(0, 1).swapaxes(1, 2)

def _gray_to_rgb(img):
    """assumes img in shape channels, height, width"""
    if img.shape[0] == 3:
        return img
    return np.concatenate(3*(img, ), axis=0)

def _load(path):
    # loading image
    img = io.imread(path)
    # setting up three dimensions if needed
    if img.ndim < 3:
        img = img.reshape(img.shape + (1, ))
    # converting from height, width, channels to channels, height, width
    img = _hwc_to_chw(img)
    return img

def load_x(identifier, x_dir):
    #getting x path
    paths = glob.glob(os.path.join(x_dir, "{}.*".format(identifier)))
    assert len(paths) == 1
    path = paths[0]
    # loading x
    x = _load(path)
    # converting to rgb if needed
    if x.shape[0] == 1:
        x = _gray_to_rgb(x)
    return x

def load_y(identifier, y_dir):
    #getting y path
    paths = glob.glob(os.path.join(y_dir, "{}.*".format(identifier)))
    assert len(paths) == 1
    path = paths[0]
    # loading y
    y = _load(path)
    return y

def load_xy(identifier, x_dir, y_dir):
    x = load_x(identifier, x_dir)
    y = load_y(identifier, y_dir)
    return x, y

def pre_proc_x(x, shape):
    """
    assumes x in format channels, height, width
    """
    # converting to height, width, channels
    x = _chw_to_hwc(x)
    # converting image to float
    x = img_as_float(x)
    # reshaping to shape
    if x.shape[:2] != shape:
        x = transf.resize(x, shape, mode="constant")
    # converting x colorspace to LAB
    x = color.rgb2lab(x)
    # std-normalizing each x channel
    for i in range(3):
        x[..., i] = _std_norm(x[..., i])
    # converting back to channels, height, width
    x = _hwc_to_chw(x)
    return x

def pre_proc_y(y, shape):
    """
    assumes y in format channels, height, width
    """
    # converting to height, width, channels
    y = _chw_to_hwc(y)
    # converting image to float
    y = img_as_float(y)
    # reshaping to shape
    if y.shape[:2] != shape:
        y = transf.resize(y, shape, mode="constant")
    # unit-normalizing
    y = _unit_norm(y)
    # converting back to channels, height, width
    y = _hwc_to_chw(y)
    return y

def pre_proc_xy(xy, x_shape, y_shape):
    x, y = xy
    x = pre_proc_x(x, x_shape)
    y = pre_proc_y(y, y_shape)
    return x, y

def infer_save_x(x, preds_dir, name):
    return

def infer_save_y_pred(y_pred, preds_dir, name):
    return

def infer_save_y_true(y_true, preds_dir, name):
    return
