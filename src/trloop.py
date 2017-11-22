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
Module for training models.
Contains routines for parallel data fetching/processing and training.
"""

import tensorflow as tf
import multiprocessing as mp
import numpy as np
import random
import time
import math
import queue
from collections import defaultdict

#true when it's time to kill fetch threads
_stop_fetch_threads = False

def _no_op(*args, **kwargs):
    pass

def _dummy_load(fp):
    return random.randint(0, 10), random.randint(0, 10)

def _dummy_augment(xy):
    return [xy]

def _dummy_pre_proc(xy):
    return xy

def _inf_gen():
    n = 0
    while True:
        yield n
        n += 1

def _pop_rand_elem(lst):
    if not lst:
        return None
    index = random.randint(0, len(lst)-1)
    lst[-1], lst[index] = lst[index], lst[-1]
    return lst.pop()

def _str_fmt_time(seconds):
    int_seconds = int(seconds)
    hours = int_seconds//3600
    minutes = (int_seconds%3600)//60
    seconds = int_seconds%60 + (seconds - int_seconds)
    return "{:02d}h:{:02d}m:{:.2f}s".format(hours, minutes, seconds)

def _str(obj):
    return str(obj) if obj is not None else "?"

def _str_fmt_dct(dct):
    """Assumes dict mapping str to float."""
    return " | ".join("{}: {:.4g}".format(k, v) for k, v in dct.items())

def fetch(fps, q, load_chunk_size, max_n_samples,
    load_fn, augment_fn, pre_proc_fn):
    """
    Thread to load, pre-process and augment data, putting it into queue q.
    """
    #list to store samples
    samples = []

    end = False
    while not end:
        if len(samples) < max_n_samples:
            #loading load_chunk_size files before putting into queue.
            #this is to better spread augmented samples
            for __ in range(load_chunk_size):
                if not fps:
                    end = True
                    break

                #getting filepath
                fp = fps.pop()
                #loading x and y
                xy = load_fn(fp)
                #augmentation
                augm_xy = augment_fn(xy)
                #pre-processing
                augm_xy = pre_proc_fn(augm_xy)
                #putting to samples list
                samples.extend(augm_xy)

        #putting random sample into queue
        if samples:
            q.put(_pop_rand_elem(samples))

    #putting remaining samples to queue
    while samples:
        q.put(_pop_rand_elem(samples))

def batch_gen(
        filepaths,
        batch_size=1,
        n_threads=1,
        max_n_samples=None,
        fetch_thr_load_chunk_size=1,
        fetch_thr_load_fn=_dummy_load,
        fetch_thr_augment_fn=_dummy_augment,
        fetch_thr_pre_proc_fn=_dummy_pre_proc,
        max_augm_factor=1,
        shuffle_fps=True):

    """
    Main thread for generation of batches.
    Spans other threads for data loading and processing, gathering the
        samples into batches and yielding them.
    """
    if max_n_samples is None:
        max_n_samples = len(filepaths)*max_augm_factor

    #shuffling filepaths
    if shuffle_fps:
        random.shuffle(filepaths)

    #threads and thread queues
    threads = []
    qs = []
    #maximum number of samples per thread
    max_n_samples_per_thr = math.ceil(
        max_n_samples/(n_threads*fetch_thr_load_chunk_size*max_augm_factor))
    #number of filepaths per thread
    n_fps_per_thr = math.ceil(len(filepaths)/n_threads)

    #initializing thread objects
    for i in range(n_threads):
        #getting a slice of filepaths for thread
        thr_fps = filepaths[i*n_fps_per_thr:(i+1)*n_fps_per_thr]
        #queue in which fetch thread will put its samples
        thr_q = mp.Queue(maxsize=1)
        #process object
        thr = mp.Process(
            target=fetch,
            args=(thr_fps, thr_q,
                fetch_thr_load_chunk_size, max_n_samples_per_thr,
                fetch_thr_load_fn, fetch_thr_augment_fn, fetch_thr_pre_proc_fn))

        threads.append(thr)
        qs.append(thr_q)

    #starting threads
    for thr in threads:
        thr.daemon = True
        thr.start()

    #true iff all threads are finished
    all_done = False
    #indexes for threads
    thr_ids = list(range(n_threads))
    #batch to be yielded
    batch = []

    while not all_done and not _stop_fetch_threads:
        all_done = True
        #shuffling indexes to fetch from threads in random order
        random.shuffle(thr_ids)

        for i in thr_ids:
            if threads[i].is_alive():
                all_done = False

            #trying to get sample from thread
            try:
                xy = qs[i].get(block=False)
            except queue.Empty:
                continue

            #if reached batch size, yields batch
            batch.append(xy)
            if len(batch) == batch_size:
                batch_x = np.stack([b[0] for b in batch], axis=0)
                batch_y = np.stack([b[1] for b in batch], axis=0)
                yield batch_x, batch_y
                batch = []

    #joining processes
    for thr in threads:
        thr.terminate()

def _val_set_loop(val_set, val_fn, val_batch_gen_kw, print_fn=_no_op):
    """
    Computes metrics over entire validation dataset.
    """
    #estimating number of batches for validation phase
    n_batches = len(val_set)//val_batch_gen_kw["batch_size"]
    #validation samples generator
    val_gen = batch_gen(val_set, **val_batch_gen_kw)
    #sum for each metric
    metrics_sum = defaultdict(float)

    #running validation set
    for i, (bx, by) in enumerate(val_gen):
        metrics = val_fn(bx, by)

        for k, v in metrics.items():
            metrics_sum[k] += v

        print_fn("[batch {}/~{}] {}".format(
            i+1, n_batches, _str_fmt_dct(metrics)), 16*" ", end="\r")

    print_fn("    processed {} batches".format(i+1), 16*" ")

    #calculating mean values
    metrics_mean = {k: v/max(i+1, 1) for k, v in metrics_sum.items()}
    return metrics_mean

def _val_step(
    val_set, val_fn, val_batch_gen_kw,
    patience, best_loss, no_loss_improv_count,
    print_fn=_no_op):

    #calculating metrics over validation dataset
    metrics_mean = _val_set_loop(val_set, val_fn, val_batch_gen_kw, print_fn)
    print_fn("    mean metrics:", _str_fmt_dct(metrics_mean))

    stop = False
    #if patience is set, check for early stopping condition
    if patience is not None:
        #loss must be in metrics_mean dict
        if not "loss" in metrics_mean:
            raise ValueError(
                "val_fn must return a dict with element 'loss'!")

        loss = metrics_mean["loss"]
        #stopping criteria
        if best_loss is None or loss < best_loss:
            best_loss = loss
            no_loss_improv_count = 0
        else:
            no_loss_improv_count += 1
            if no_loss_improv_count > patience:
                stop = True

    return stop, best_loss, no_loss_improv_count

def train_loop(
    train_set, train_fn,
    n_epochs=10,
    val_set=None, val_fn=None, val_every_its=None, patience=None,
    log_every_its=None, log_fn=_no_op,
    save_model_fn=_no_op, save_every_its=None,
    verbose=2, print_fn=print,
    batch_gen_kw={}):
    """
    General Training loop.
    """
    #it is set to True to stop fetch threads
    global _stop_fetch_threads

    #info/warning functions
    info = print_fn if verbose >= 2 else _no_op
    warn = print_fn if verbose >= 1 else _no_op
    print_ = print if verbose >= 2 else _no_op

    #checking if using validation
    validation = val_set is not None and val_fn is not None
    #setting up validation batches_gen_kwargs
    val_batch_gen_kw = dict(batch_gen_kw)
    val_batch_gen_kw["max_augm_factor"] = 1
    val_batch_gen_kw["fetch_thr_augment_fn"] = _dummy_augment

    #batch generator for validation set
    if log_every_its is not None:
        val_gen = batch_gen(val_set, **val_batch_gen_kw)

    #total train iterations
    its = 0

    #parameters for early stopping to be used if patience is set
    best_val_loss = None
    no_val_loss_improv_count = 0

    #initial start time
    start_time = time.time()

    #main loop
    print_("starting training loop...")
    for epoch in _inf_gen():
        #checking stopping
        if n_epochs is not None and epoch >= n_epochs:
            warn("\nWARNING: maximum number of epochs reached")
            _stop_fetch_threads = True
            return

        info("{}/{} epochs (time so far: {})".format(
            epoch, _str(n_epochs), _str_fmt_time(time.time() - start_time)))

        #dictionary in format metric: value
        loss_sum = 0
        #estimating number of batches for train phase
        n_batches = len(train_set)//batch_gen_kw["batch_size"]
        n_batches *= batch_gen_kw["max_augm_factor"]

        #main train loop
        for i, (bx, by) in enumerate(batch_gen(train_set, **batch_gen_kw)):
            its += 1
            
            #model update
            loss = train_fn(bx, by)
            loss_sum += loss
            print_("[epoch {}, batch {}/~{}] loss: {:.4g}".format(
                epoch, i+1, n_batches, loss), 16*" ", end="\r")

            #logging every #step
            if log_every_its is not None and its%log_every_its == 0:
                try:
                    val_bx, val_by = next(val_gen)
                except StopIteration:
                    val_gen = batch_gen(val_set, **val_batch_gen_kw)
                    val_bx, val_by = next(val_gen)

                #uncomment this to print statistics on console
                #metrics = val_fn(bx, by)
                #print_("[batch {}] metrics (train): {}".format(
                #    i+1, _str_fmt_dct(metrics)), 16*" ")
                #metrics = val_fn(val_bx, val_by)
                #print_("[batch {}] metrics (val): {}".format(
                #    i+1, _str_fmt_dct(metrics)), 16*" ")
                log_fn(bx, by, its, train=True)
                log_fn(val_bx, val_by, its, train=False)

            #validation every #step if required
            if val_every_its is not None and its%val_every_its == 0:
                info("\n----------\nvalidation on it #{}".format(its), 16*" ")
                #old best validation loss
                best_val_loss_ = best_val_loss
                early_stopping, best_val_loss, no_val_loss_improv_count =\
                    _val_step(val_set, val_fn, val_batch_gen_kw,
                        patience, best_val_loss, no_val_loss_improv_count,
                        info)
                info("----------")

                if best_val_loss_ is None or best_val_loss < best_val_loss_:
                    save_model_fn(name="best")

                if early_stopping:
                    info("WARNING: early stopping condition met")
                    _stop_fetch_threads = True
                    return

            #saving model every #step if required
            if save_every_its is not None and its%save_every_its == 0:
                save_model_fn(epoch, i+1)

        #saving model after epoch
        save_model_fn(epoch+1, 0)

        info("\n----------\nend of epoch #{}".format(epoch))
        #printing train loop metrics
        loss_mean = loss_sum/max(i+1, 1)
        info("train set:", 32*" ")
        info("    processed {} batches".format(i+1), 16*" ")
        info("    mean loss: {}".format(loss_mean), flush=True)

        if not validation:
            info("----------")
            continue

        #validation step after epoch
        info("val set:")
        best_val_loss_ = best_val_loss
        early_stopping, best_val_loss, no_val_loss_improv_count =\
            _val_step(val_set, val_fn, val_batch_gen_kw,
                patience, best_val_loss, no_val_loss_improv_count,
                info)
        info("----------")

        if early_stopping and val_every_its is None:
            info("WARNING: early stopping condition met")
            _stop_fetch_threads = True
            return

        if best_val_loss_ is None or best_val_loss < best_val_loss_:
            save_model_fn(name="best")
