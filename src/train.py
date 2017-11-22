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

import subprocess as sp
import tensorflow as tf
import os
import sys
import shutil
from collections import OrderedDict

import model
import trloop
import config as conf
import util

def populate_out_dir(out_dir):
    """
    Populates output dir with info files.
    """
    #info file
    with open(os.path.join(out_dir, "etc", "train-log", "info.txt"), "w") as f:
        print("date created (y-m-d):", util.date_str(), file=f)
        print("time created:", util.time_str(), file=f)
        print("git commit hash:", util.git_hash(), file=f)

    #saving train/val filepaths
    with open(os.path.join(out_dir, "input", "train.csv"), "w") as f:
        for fp in conf.train["train_set_fps"]:
            print(fp, file=f)

    with open(os.path.join(out_dir, "input", "val.csv"), "w") as f:
        for fp in conf.train["val_set_fps"]:
            print(fp, file=f)

def main():
    out_dir = util.mk_model_dir(conf.train["out_dir_basedir"])
    print("created out dir '{}', populating...".format(out_dir),
        flush=True, end=" ")
    populate_out_dir(out_dir)
    print("done.")

    #meta-model
    meta_model = model.MetaModel()

    #creating logging object
    log = util.Tee([sys.stdout,
        open(os.path.join(out_dir, "etc", "train-log", "train.log"), "w")])

    #building graph
    if conf.train["pre_trained_model_path"] is None:
        log.print("[info] building graph for the first time")
        graph = meta_model.build_graph()
    else:
        graph = tf.Graph()

    #tensorboard logging paths
    summ_dir = os.path.join(out_dir, "etc", "train-log", "summaries")
    if conf.train["use_tensorboard"]:
        #tensorboard summary writers
        train_writer = tf.summary.FileWriter(
            os.path.join(summ_dir, "train"), graph)
        val_writer = tf.summary.FileWriter(
            os.path.join(summ_dir, "val"), graph)
        #running tensorboard
        cmd = ["tensorboard", "--logdir={}".format(summ_dir),
            "--port={}".format(conf.train["tensorboard_port"])]
        log.print("[info] running '{}'".format(" ".join(cmd)))
        proc = sp.Popen(cmd, stdout=sp.PIPE, stderr=sp.PIPE)

    #training session
    with tf.Session(graph=graph) as sess:
        #if first time training, creates graph collections for model params
        #else, loads model weights and params from collections
        if conf.train["pre_trained_model_path"] is None:
            #sess.run(tf.global_variables_initializer())
            sess.run(
                tf.group(
                    tf.global_variables_initializer(),
                    tf.local_variables_initializer()))
            meta_model.mk_params_colls(graph=graph)
        else:
            log.print("[info] loading graph/weights from '{}'".format(
                conf.train["pre_trained_model_path"]))
            model.load(sess, conf.train["pre_trained_model_path"])
            meta_model.set_params_from_colls(graph=graph)

        #building functions
        #train function: cumputes loss
        _train_fn = meta_model.get_train_fn(sess)
        def train_fn(x, y_true):
            return _train_fn(x, y_true, {
                meta_model.params["learning_rate"]: conf.train["learning_rate"]
            })

        #test function: returns a dict with pairs metric_name: metric_value
        _test_fn = meta_model.get_test_fn(sess)
        def test_fn(x, y_true):
            metrics_values = _test_fn(x, y_true)
            return OrderedDict(zip(
                    meta_model.params["metrics"].keys(), metrics_values))

        #save model function: given epoch and iter number, saves checkpoint
        def save_model_fn(epoch=None, it=None, name=None):
            if name is None:
                path = os.path.join(out_dir, "self", "ckpts",
                    "epoch-{}_it-{}".format(epoch, it))
            else:
                path = os.path.join(out_dir, "self", "ckpts", "{}".format(name))
            model.save(sess, path, overwrite=True)
            print("    saved checkpoint to '{}'".format(path))

        #test
        if conf.train["use_tensorboard"]:
            _log_fn = meta_model.get_summary_fn(sess)
            def log_fn(x, y_true, its, train=True):
                summ = _log_fn(x, y_true)
                if train:
                    train_writer.add_summary(summ, its)
                    if its%10 == 0:
                        train_writer.flush()
                else:
                    val_writer.add_summary(summ, its)
                    if its%10 == 0:
                        val_writer.flush()
        else:
            log_fn = None

        #main train loop
        print("calling train loop")
        try:
            trloop.train_loop(
                train_set=conf.train["train_set_fps"],
                train_fn=train_fn,
                n_epochs=conf.train["n_epochs"],
                val_set=conf.train["val_set_fps"],
                val_fn=test_fn,
                val_every_its=conf.train["val_every_its"],
                patience=conf.train["patience"],
                log_every_its=conf.train["log_every_its"],
                log_fn=log_fn,
                save_model_fn=save_model_fn,
                save_every_its=conf.train["save_every_its"],
                verbose=conf.train["verbose"],
                print_fn=log.print,
                batch_gen_kw=conf.train["batch_gen_kw"]
            )
        except KeyboardInterrupt:
            print("Keyboard Interrupt event.")
        finally:
            #closing tensorboard writers
            if conf.train["use_tensorboard"]:
                train_writer.close()
                val_writer.close()

            #saving model on final state
            path = os.path.join(out_dir, "self", "ckpts", "final")
            print("saving checkpoint to '{}'...".format(path), flush=True)
            model.save(sess, path, overwrite=True)

    print("\ndone.", flush=True)

if __name__ == '__main__':
    main()
