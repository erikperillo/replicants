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
from config import predict as conf
import config

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
        load_fn = conf["load_fn"]
        save_x_fn = conf["save_x_fn"]
        save_pred_fn = conf["save_pred_fn"]
        save_true_fn = conf["save_true_fn"]
        _pred_fn = meta_model.get_pred_fn(sess)
        pred_fn = lambda x: conf["predict_fn"](x, _pred_fn)

        indexes = None
        #iterating over images doing predictions
        for i, fp in enumerate(input_fps):
            print("on image '{}'".format(fp))

            if conf["with_trues"]:
                x, y_true = load_fn(fp)
            else:
                x = load_fn(fp)

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
                if save_x_fn is not None:
                    dir_path = os.path.join(preds_dir, "x")
                    if not os.path.isdir(dir_path):
                        os.makedirs(dir_path)
                    #save_x_fn(x, dir_path, name)
                    save_x_fn(x, preds_dir, name)
                #saving prediction
                if save_pred_fn is not None:
                    dir_path = os.path.join(preds_dir, "y_pred")
                    if not os.path.isdir(dir_path):
                        os.makedirs(dir_path)
                    #save_pred_fn(y_pred, dir_path, name)
                    save_pred_fn(y_pred, preds_dir, name)
                #saving ground-truth
                if save_true_fn is not None and conf["with_trues"]:
                    dir_path = os.path.join(preds_dir, "y_true")
                    if not os.path.isdir(dir_path):
                        os.makedirs(dir_path)
                    #save_true_fn(y_true, dir_path, name)
                    save_true_fn(y_true, preds_dir, name)

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

