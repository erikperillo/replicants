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

import random
import numpy as np
import os
import glob

import util
import augment
import dproc

### GENERAL CONFIGURATION
random.seed(42)
#paths
_train_output_dir = "/home/erik/new_ml/data/train/output"
_train_x_dir = "/home/erik/new_ml/data/train/input/x"
_train_y_dir = "/home/erik/new_ml/data/train/input/y"
#train/validation identifiers
_identifiers = [os.path.basename(p).split(".")[0]
    for p in glob.glob(os.path.join(_train_x_dir, "*"))]
random.shuffle(_identifiers)
_train_identifiers = _identifiers[:14000]
_val_identifiers = _identifiers[14000:]


### DATA AUGMENTATION CONFIGURATION ###
#data augmentation operations
_augment_ops = [
    ("hmirr", {}, 0.5),
    ("blur", {"sigma": (0.5, 1.0)}, 0.1),
    ("translation", {"transl": (-0.2, 0.2)}, 0.1),
    ("rotation", {"angle": (-35, 35)}, 0.2),
    ("shear", {"angle": (-0.2, 0.2)}, 0.1),
    ("add_noise", {"noise": (-0.1, 0.1)}, 0.1),
    ("mul_noise", {"noise": (0.9, 1.1)}, 0.1),
]
#data augmentation function
_augment_fn = lambda xy: augment.augment(
    xy=xy, operations=_augment_ops, copy_xy=False)


### DATA PROCESSING CONFIGURATION ###
_train_x_input_shape = (256, 320)
_train_y_input_shape = (256, 320)
_train_pre_proc_fn = lambda xy: dproc.pre_proc_xy(
    xy=xy, x_shape=_train_x_input_shape, y_shape=_train_y_input_shape)
_train_load_fn = lambda identifier: dproc.load_xy(
    identifier=identifier, x_dir=_train_x_dir, y_dir=_train_y_dir)

### TRAIN CONFIGURATION ###
train = {
    #base directory where new directory with train data will be created
    "out_dir_basedir": _train_output_dir,

    #use tensorboard summaries
    "use_tensorboard": True,
    "tensorboard_port": 6006,

    #path to directory containing data needed by tensorflow's SavedModel
    #can be None
    "pre_trained_model_path": \
        None,

    #learning rate of the model
    "learning_rate": 3e-4,

    #list with identifiers of train set
    "train_identifiers": _train_identifiers,

    #list with identifiers of validation set
    "val_identifiers": _val_identifiers,

    #number of epochs for training loop. can be None
    "n_epochs": 64,

    #logs metrics every log_every_its, can be None
    "log_every_its": 30,

    #computes metrics on validation set every val_every_its. can be None
    "val_every_its": None,

    #number of times val set loss does not improve before early stopping.
    #can be None, in which case early stopping will never occur.
    "patience": 4,

    #save checkpoint with graph/weights every save_every_its besides epochs.
    #can be None
    "save_every_its": None,

    #verbosity
    "verbose": 2,

    #arguments to be provided by trloop.batch_gen function
    "batch_gen_kw": {
        #size of batch to be fed to model
        "batch_size": 12,

        #number of fetching threads for data loading/pre-processing/augmentation
        "n_threads": 10,

        #maximum number of samples to be loaded at a time.
        #the actual number may be slightly larger due to rounding.
        "max_n_samples": 8000,

        #function to return tuple (x, y_true) given filepath
        "fetch_thr_load_fn": _train_load_fn,

        #function to return (possibly) augmented image
        "fetch_thr_augment_fn": _augment_fn,

        #function to return batches of x (optionally (x, y))
        #given batches of x (optionally y as second argument)
        "fetch_thr_pre_proc_fn": _train_pre_proc_fn,
    },
}

### INFER CONFIGURATION ###
infer = {
    #random seed to be used, can be None
    "rand_seed": 42,

    #list of input filepaths containing x values (optionally (x, y_true) tuples)
    "input_fps": [],

    #whether or not shuffle list of input filepaths
    "shuffle_input_fps": True,

    #path to directory containing meta-graph and weights for model
    "model_path": "",

    #base dir where new preds directory will be created
    "preds_save_dir_basedir": os.path.join(_train_output_dir, "inferences"),

    #if true, creates table.npz, containing x_fps, y_pred (and possibly y_true)
    "save_tables": not True,

    #if true, tries to load true values with load_fn
    "with_trues": not True,

    #maximum prediction data points to be stored, can be None
    "max_pred_points": 9999999,

    #maximum number of preds to save, can be None
    "max_n_preds_save": 999999,
}
