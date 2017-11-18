#!/usr/bin/env python3

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

import tensorflow as tf
import sys
import random
import shutil
import time
import itertools
import numpy as np
import os

import util
import model
import dproc
from config import predict as conf
import config

def predict(x, fn):
    x = _predict_pre_proc(x)
    x = x.reshape((1, ) + x.shape)
    y_pred = fn(x)
    y_pred = y_pred.reshape(y_pred.shape[2:])
    y_pred = transf.rescale(y_pred, 8, mode="constant")
    y_pred = unit_norm(y_pred)
    return y_pred

def load(fp):
    x = dproc.load(fp)[0]
    x = x.swapaxes(0, 1).swapaxes(1, 2)

    #reshaping to fix max input shape if necessary
    h, w = x.shape[:2]
    max_h, max_w = (480, 640)
    max_ratio = max(h/max_h, w/max_w)
    if max_ratio > 1.0:
        x = (255*transf.rescale(x, 1/max_ratio,
            mode="constant")).astype("uint8")
    return x

def save_x(x, preds_dir, name):
    fp = util.uniq_filepath(preds_dir, name + "_x", ext=".png")
    io.imsave(fp, x.clip(0, 255).astype("uint8"))

def save_y_pred(y_pred, preds_dir, name):
    fp = util.uniq_filepath(preds_dir, name + "_y-pred", ext=".png")
    y_pred = 255*y_pred
    io.imsave(fp, y_pred.clip(0, 255).astype("uint8"))

def save_y_true(y_true, preds_dir, name):
    fp = util.uniq_filepath(preds_dir, name + "_y-true", ext=".png")
    y_true = 255*y_true
    y_true = y_true.reshape(y_true.shape[1:])
    io.imsave(fp, y_true.clip(0, 255).astype("uint8"))

def mk_preds_dir(base_dir, pattern="train"):
    """
    Creates dir to store predictions.
    """
    #creating dir
    out_dir = util.uniq_filepath(base_dir, pattern)
    os.makedirs(out_dir)
    return out_dir

def main():
    if conf["rand_seed"] is not None:
        random.seed(conf["rand_seed"])

    if conf["input_fps"] is None:
        if len(sys.argv) < 2:
            print("usage: {} <filepath_or_dir_of_inputs>".format(sys.argv[0]))
            exit()
        else:
            input_fps = sys.argv[1]
    else:
        input_fps = conf["input_fps"]

    if isinstance(input_fps, str):
        input_fps = [input_fps]

    if conf["shuffle_input_fps"]:
        random.shuffle(input_fps)

    preds = None
    trues = None

    #creating base dir if needed
    if not os.path.isdir(conf["preds_save_dir_basedir"]):
        os.makedirs(conf["preds_save_dir_basedir"])
    #creating preds dir
    preds_dir = mk_preds_dir(conf["preds_save_dir_basedir"], "preds")
    #copying config file
    shutil.copy(config.__file__, preds_dir)

    #meta-model
    meta_model = model.MetaModel(config.model["build_graph_fn"])

    with tf.Session(graph=tf.Graph()) as sess:
        #loading model weights
        model.load(sess, conf["model_path"])
        meta_model.set_params_from_colls()

        #building functions
        _pred_fn = meta_model.get_pred_fn(sess)
        pred_fn = lambda x: predict(x, _pred_fn)

        indexes = None
        #iterating over images doing predictions
        for i, fp in enumerate(input_fps):
            print("on image '{}'".format(fp))

            if conf["with_trues"]:
                x, y_true = load(fp)
            else:
                x = load(fp)

            print("\tpredicting...")
            print("\tx shape:", x.shape)
            start_time = time.time()
            y_pred = pred_fn(x)
            pred_time = time.time() - start_time
            print("\tdone predicting. took {:.6f} seconds".format(pred_time))
            print("\ty_pred shape:", y_pred.shape)

            if conf["save_tables"]:
                #getting indexes
                if indexes is None:
                    pts_per_img = y_pred.size
                    if conf["max_pred_points"] is not None:
                        pts_per_img = min(
                            conf["max_pred_points"]//len(input_fps),
                            pts_per_img)
                    indexes = list(range(pts_per_img))

                if len(indexes) < y_pred.size:
                    random.shuffle(indexes)
                if preds is None:
                    preds = y_pred.flatten()[indexes]
                else:
                    preds = np.vstack((preds, y_pred.flatten()[indexes]))
                if conf["with_trues"]:
                    if trues is None:
                        trues = y_true.flatten()[indexes]
                    else:
                        trues = np.vstack((trues, y_true.flatten()[indexes]))

            if conf["max_n_preds_save"] is None or i < conf["max_n_preds_save"]:
                fn = os.path.basename(fp)
                name = fn.split(".")[0]
                ext = ("." + fn.split(".")[-1]) if "." in fn else ""

                #saving x
                if save_x is not None:
                    save_x(x, preds_dir, name)
                #saving prediction
                if save_y_pred is not None:
                    save_y_pred(y_pred, preds_dir, name)
                #saving ground-truth
                if save_true_fn is not None and conf["with_trues"]:
                    save_y_true(y_true, preds_dir, name)

        #saving predictions
        if conf["save_tables"] and preds is not None:
            fp = os.path.join(preds_dir, "table.npz")
            print("saving table to '{}'...".format(fp, flush=True))
            if conf["with_trues"]:
                np.savez(fp, y_pred=preds, y_true=trues, x_fp=input_fps)
            else:
                np.savez(fp, y_pred=preds, x_fp=input_fps)

        print("saved everything to", preds_dir)

if __name__ == "__main__":
    main()

