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

from skimage import io
from skimage import transform as transf
from skimage import color
from skimage import img_as_float

import tensorflow as tf
import numpy as np
import os
import glob
import json

import util
import augment
import predictfuncs

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

_load = _load_judd

def __pre_proc(x, y=None):
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

def _pre_proc(batch_xy):
    for i, xy in enumerate(batch_xy):
        batch_xy[i] = __pre_proc(*xy)
    return batch_xy

#data augmentation default operation sequence to be applied in about every op
_def_augm_ops = [
    ("blur", 0.15, {"rng": (0.5, 1.0)}),
    ("translation", 0.15, {"rng": (-0.1, 0.1)}),
    ("rotation", 0.15, {"rng": (-35, 35)}),
    ("shear", 0.15, {"rng": (-0.1, 0.1)}),
    ("add_noise", 0.15, {"rng": (-0.1, 0.1)}),
    ("mul_noise", 0.15, {"rng": (0.9, 1.1)}),
]

"""
Sequences of operations for data augmentation.
Each sequence spans a new image and applies operations randomly as defined by
    their probabilities.
Each sequence must contain operations in format (op_name, op_prob, op_kwargs).
"""
_augment_op_seqs = [
    [
        ("identity", 1.0, {}),
    ] + _def_augm_ops,

    [
        ("hmirr", 1.0, {}),
    ] + _def_augm_ops,
]

def _augment(xy):
    return augment.augment(xy, _augment_op_seqs, apply_on_y=True)

import random
_fps = glob.glob("/home/erik/sal/judd_cat2000/stimuli/*")
random.seed(42)
random.shuffle(_fps)
#_train_fps = _fps[:12000]
#_test_fps = _fps[12000:13500]
#_val_fps = _fps[13500:]
#_train_fps = _fps[:2200]
_train_fps = _fps[:10]
_test_fps = _fps[2200:2600]
#_val_fps = _fps[2600:]
_val_fps = _fps[2600:2605]

#arguments for train routine
train = {
    #base directory where new directory with train data will be created
    "out_dir_basedir": "/home/erik/rand/traindata",

    #use tensorboard summaries
    "use_tensorboard": True,
    "tensorboard_port": 6006,

    #path to directory containing data needed by tensorflow's SavedModel
    #can be None
    "pre_trained_model_path": \
        None,
        #"/home/erik/rand/traindata/model-22/data/self/epoch-20_it-0",

    #learning rate of the model
    #"learning_rate": 3e-4,
    "learning_rate": 1e-5,

    #list with filepaths of train files
    "train_set_fps": _train_fps,

    #list with filepaths of validation files
    "val_set_fps": _val_fps,

    #number of epochs for training loop. can be None
    "n_epochs": 64,

    #logs metrics every log_every_its, can be None
    "log_every_its": 20,

    #computes metrics on validation set every val_every_its. can be None
    "val_every_its": 30,

    #number of times val set loss does not improve before early stopping.
    #can be None, in which case early stopping will never occur.
    "patience": 3,

    #save checkpoint with graph/weights every save_every_its besides epochs.
    #can be None
    "save_every_its": None,

    #verbosity
    "verbose": 2,

    #arguments to be provided by trloop.batch_gen function
    "batch_gen_kw": {
        #size of batch to be fed to model
        "batch_size": 2,

        #number of fetching threads for data loading/pre-processing/augmentation
        "n_threads": 14,

        #maximum number of samples to be loaded at a time.
        #the actual number may be slightly larger due to rounding.
        "max_n_samples": 8000,

        #the fetching threads loads a chunk of this size before augmentation
        #and pre-processing.
        #this spreads out the augmented versions of an image in the feeding line
        "fetch_thr_load_chunk_size": 10,

        #function to return tuple (x, y_true) given filepath
        "fetch_thr_load_fn": _load,

        #function to return list of tuples [(_x, _y), ...] given (x, y) tuple
        "fetch_thr_augment_fn": _augment,

        #function to return x (optionally (x, y))
        #given x (optionally y as second argument)
        "fetch_thr_pre_proc_fn": _pre_proc,

        #the maximum factor by which number of samples will be increased
        #due to data augmentation
        "max_augm_factor": len(_augment_op_seqs),
    },
}

def __predict(x, train_fn):
    x = _pre_proc(x)
    x = x.reshape((1, ) + x.shape)
    y_pred = train_fn(x)
    y_pred = y_pred.reshape(y_pred.shape[2:])
    return y_pred

def _predict(x, train_fn):
    #return __predict(x, train_fn)
    return predictfuncs.stride_predict(x, 224, 64,
        lambda x: __predict(x, train_fn))

def _predict_load(fp):
    xy = np.load(fp)
    return xy["x"]

def _save_x(x, preds_dir, name):
    rgbs = []
    for i in range(0, x.shape[0], 8):
        rgbs.append(np.dstack((x[i], x[i+1], x[i+2])))
    rgbs = np.hstack(rgbs)
    fp = util.uniq_filepath(preds_dir, name + "_x", ext=".png")
    io.imsave(fp, rgbs.clip(0, 255).astype("uint8"))

def _save_y_pred(y_pred, preds_dir, name):
    fp = util.uniq_filepath(preds_dir, name + "_y-pred", ext=".png")
    y_pred = 255*y_pred
    io.imsave(fp, y_pred.clip(0, 255).astype("uint8"))

def _save_y_true(y_true, preds_dir, name):
    fp = util.uniq_filepath(preds_dir, name + "_y-true", ext=".png")
    y_true = 255*y_true
    y_true = y_true.reshape(y_true.shape[1:])
    io.imsave(fp, y_true.clip(0, 255).astype("uint8"))

#arguments for predict routine
predict = {
    #random seed to be used, can be None
    "rand_seed": 42,

    #list of input filepaths containing x values (optionally (x, y_true) tuples)
    "input_fps": glob.glob(
        ""),
        #"/home/erik/data/ls8/{}/cuts/cuts_0/test/*.npz".format(_crop)),
        #"/home/erik/data/ls8/tensors/imgs/*.npz"),

    #whether or not shuffle list of input filepaths
    "shuffle_input_fps": True,

    #path to directory containing meta-graph and weights for model
    "model_path": \
        "/home/erik/rand/traindata/train-190/ckpts/final",

    #base dir where new preds directory will be created
    "preds_save_dir_basedir": "/home/erik/rand/preds",

    #load function that, given a filepath, returns x
    #(or (x, y_true) tuple if argument 'with_trues' is set to True)
    "load_fn": _load,

    #predict function
    #given x and pred_fn returned by model.MetaModel.get_pred_fn, returns y_pred
    "predict_fn": _predict,

    #if true, creates table.npz, containing x_fps, y_pred (and possibly y_true)
    "save_tables": True,

    #if true, tries to load true values with load_fn
    "with_trues": True,

    #maximum prediction data points to be stored, can be None
    "max_pred_points": 9999999,

    #maximum number of preds to save, can be None
    "max_n_preds_save": 30,

    #function to save x, given x, base_dir (always exists) and pattern 'name'
    "save_x_fn": _save_x,

    #function to save pred, given x, base_dir (always exists) and pattern 'name'
    "save_pred_fn": _save_y_pred,

    #function to save true, given x, base_dir (always exists) and pattern 'name'
    "save_true_fn": _save_y_true,
}
