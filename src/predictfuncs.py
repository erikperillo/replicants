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

import numpy as np
import itertools

def stride_predict(x, shape, stride, pred_fn):
    if isinstance(shape, int):
        shape = (shape, shape)
    h_shape, w_shape = shape
    if isinstance(stride, int):
        stride = (stride, stride)
    h_stride, w_stride = stride
    height, width = x.shape[-2:]

    y_pred = np.zeros(shape=x.shape[-2:], dtype="float32")
    mult = np.zeros(shape=x.shape[-2:], dtype="int")
    pred_counter = 0

    for i, j in itertools.product(range(0, height-h_shape+1, h_stride),
        range(0, width-w_shape+1, w_stride)):
        _x = x[..., i:i+h_shape, j:j+w_shape].copy()
        _y_pred = pred_fn(_x)

        y_pred[i:i+h_shape, j:j+w_shape] += _y_pred
        mult[i:i+h_shape, j:j+w_shape] += 1

        pred_counter += 1

    mult[np.where(mult == 0)] = 1
    y_pred = y_pred/mult.astype("float32")
    #avg_pred_time = tot_pred_time/max(1, pred_counter)

    return y_pred
