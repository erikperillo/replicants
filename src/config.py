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

"""
Simple U-Net implementation in TensorFlow
Objective: detect vehicles
y = f(X)
X: image (640, 960, 3)
y: mask (640, 960, 1)
   - binary image
   - background is masked 0
   - vehicle is masked 255
Loss function: maximize IOU
    (intersection of prediction & grount truth)
    -------------------------------
    (union of prediction & ground truth)
Notes:
    In the paper, the pixel-wise softmax was used.
    But, I used the IOU because the datasets I used are
    not labeled for segmentations
Original Paper:
    https://arxiv.org/abs/1505.04597
"""
import time
import os
import pandas as pd
import tensorflow as tf

def conv_conv_conv_pool(input_, n_filters, training, name, pool=True, activation=tf.nn.relu):
    """{Conv -> BN -> RELU}x2 -> {Pool, optional}
    Args:
        input_ (4-D Tensor): (batch_size, H, W, C)
        n_filters (list): number of filters [int, int]
        training (1-D Tensor): Boolean Tensor
        name (str): name postfix
        pool (bool): If True, MaxPool2D
        activation: Activaion functions
    Returns:
        net: output of the Convolution operations
        pool (optional): output of the max pooling operations
    """
    net = input_

    with tf.variable_scope("layer{}".format(name)):
        for i, F in enumerate(n_filters):
            net = tf.layers.conv2d(net, F, (3, 3), activation=None, padding='same', name="conv_{}".format(i + 1))
            net = tf.layers.batch_normalization(net, training=training, name="bn_{}".format(i + 1))
            net = activation(net, name="relu{}_{}".format(name, i + 1))

        if pool is False:
            return net

        pool = tf.layers.max_pooling2d(net, (2, 2), strides=(2, 2), name="pool_{}".format(name))

        return net, pool



def conv_conv_conv_pool(input_, n_filters, training, name, pool=True, first_layer=False, activation=tf.nn.relu):
    """{Conv -> BN -> RELU}x2 -> {Pool, optional}
    Args:
        input_ (4-D Tensor): (batch_size, H, W, C)
        n_filters (list): number of filters [int, int]
        training (1-D Tensor): Boolean Tensor
        name (str): name postfix
        pool (bool): If True, MaxPool2D
        activation: Activaion functions
    Returns:
        net: output of the Convolution operations
        pool (optional): output of the max pooling operations
    """
    x = input_

    n1, n2, n3 = n_filters

    i = 0
    with tf.variable_scope("layer{}".format(name)):
        conv1 = tf.layers.conv2d(x, n1, (3, 3), activation=None, padding='same', name="conv_{}".format(i + 1))
        if not first_layer:
            bn1 = tf.layers.batch_normalization(conv1, training=training, name="bn_{}".format(i + 1))
        else:
            bn1 = conv1
        relu1 = activation(bn1, name="relu{}_{}".format(name, i + 1))

        i += 1
        conv2 = tf.layers.conv2d(relu1, n2, (3, 3), activation=None, padding='same', name="conv_{}".format(i + 1))
        bn2 = tf.layers.batch_normalization(conv2, training=training, name="bn_{}".format(i + 1))
        relu2 = activation(bn2, name="relu{}_{}".format(name, i + 1))

        i += 1
        conv3 = tf.layers.conv2d(relu2, n3, (3, 3), activation=None, padding='same', name="conv_{}".format(i + 1))
        bn3 = tf.layers.batch_normalization(conv3, training=training, name="bn_{}".format(i + 1))
        relu3 = activation(bn3, name="relu{}_{}".format(name, i + 1))
        if pool:
            relu3 = tf.layers.max_pooling2d(relu3, (2, 2), strides=(2, 2), name="pool_{}".format(name))

    return relu1, relu2, relu3

def conv_conv_pool(input_, n_filters, training, name, activation=tf.nn.relu):
    """{Conv -> BN -> RELU}x2 -> {Pool, optional}
    Args:
        input_ (4-D Tensor): (batch_size, H, W, C)
        n_filters (list): number of filters [int, int]
        training (1-D Tensor): Boolean Tensor
        name (str): name postfix
        pool (bool): If True, MaxPool2D
        activation: Activaion functions
    Returns:
        net: output of the Convolution operations
        pool (optional): output of the max pooling operations
    """
    x = input_

    n1, n2 = n_filters

    i = 0
    with tf.variable_scope("layer{}".format(name)):
        conv2 = tf.layers.conv2d(x, n1, (3, 3), activation=None, padding='same', name="conv_{}".format(i + 1))
        bn2 = tf.layers.batch_normalization(conv2, training=training, name="bn_{}".format(i + 1))
        relu2 = activation(bn2, name="relu{}_{}".format(name, i + 1))

        i += 1
        conv3 = tf.layers.conv2d(relu2, n2, (3, 3), activation=None, padding='same', name="conv_{}".format(i + 1))
        bn3 = tf.layers.batch_normalization(conv3, training=training, name="bn_{}".format(i + 1))
        relu3 = activation(bn3, name="relu{}_{}".format(name, i + 1))

    return relu2, relu3


def upsample_concat(inputA, input_B, name):
    """Upsample `inputA` and concat with `input_B`
    Args:
        input_A (4-D Tensor): (N, H, W, C)
        input_B (4-D Tensor): (N, 2*H, 2*H, C2)
        name (str): name of the concat operation
    Returns:
        output (4-D Tensor): (N, 2*H, 2*W, C + C2)
    """
    upsample = upsampling_2D(inputA, size=(2, 2), name=name)

    return tf.concat([upsample, input_B], axis=-1, name="concat_{}".format(name))


def upsampling_2D(tensor, name, size=(2, 2)):
    """Upsample/Rescale `tensor` by size
    Args:
        tensor (4-D Tensor): (N, H, W, C)
        name (str): name of upsampling operations
        size (tuple, optional): (height_multiplier, width_multiplier)
            (default: (2, 2))
    Returns:
        output (4-D Tensor): (N, h_multiplier * H, w_multiplier * W, C)
    """
    H, W, _ = tensor.get_shape().as_list()[1:]

    H_multi, W_multi = size
    target_H = H * H_multi
    target_W = W * W_multi

    return tf.image.resize_nearest_neighbor(tensor, (target_H, target_W), name="upsample_{}".format(name))


def make_unet(X, training):
    """Build a U-Net architecture
    Args:
        X (4-D Tensor): (N, H, W, C)
        training (1-D Tensor): Boolean Tensor is required for batchnormalization layers
    Returns:
        output (4-D Tensor): (N, H, W, C)
            Same shape as the `input` tensor
    Notes:
        U-Net: Convolutional Networks for Biomedical Image Segmentation
        https://arxiv.org/abs/1505.04597
    """
    net = X

    convrelu1, bnconvrelu1, pool1 = conv_conv_conv_pool(net, [32, 32, 32], training=training, first_layer=True, name=1)
    print(convrelu1, pool1)
    bnconvrelu21, bnconvrelu22, pool2 = conv_conv_conv_pool(pool1, [48, 48, 48], training=training, name=2)
    print(bnconvrelu21, pool2)
    bnconvrelu31, bnconvrelu32, pool3 = conv_conv_conv_pool(pool2, [64, 64, 64], training=training, name=3)
    print(bnconvrelu31, pool3)
    bnconvrelu41, bnconvrelu42, pool4 = conv_conv_conv_pool(pool3, [96, 96, 96], training=training, name=4)
    print(bnconvrelu41, pool4)
    bnconvrelu51, bnconvrelu52, pool5 = conv_conv_conv_pool(pool4, [112, 112, 112], training=training, name=5)
    print(bnconvrelu51, pool5)
    bnconvrelu61, bnconvrelu62, pool6 = conv_conv_conv_pool(pool5, [128, 128, 128], training=training, name=6, pool=False)
    print(bnconvrelu61, pool6)

    up7 = upsample_concat(pool6, bnconvrelu52, name=7)
    bnconvrelu71, bnconvrelu72 = conv_conv_pool(up7, [128, 112], training=training, name=7)
    up8 = upsample_concat(bnconvrelu72, bnconvrelu42, name=8)
    bnconvrelu81, bnconvrelu82 = conv_conv_pool(up8, [112, 96], training=training, name=8)
    up9 = upsample_concat(bnconvrelu82, bnconvrelu32, name=9)
    bnconvrelu91, bnconvrelu92 = conv_conv_pool(up9, [96, 64], training=training, name=9)
    up10 = upsample_concat(bnconvrelu92, bnconvrelu22, name=10)
    bnconvrelu101, bnconvrelu102 = conv_conv_pool(up10, [64, 48], training=training, name=10)
    up11 = upsample_concat(bnconvrelu102, bnconvrelu1, name=11)
    bnconvrelu111, bnconvrelu112 = conv_conv_pool(up11, [48, 32], training=training, name=11)

    return tf.layers.conv2d(bnconvrelu112, 1, (1, 1), name='final', activation=None, padding='same')

def _bn_conv(*args, **kwargs):
    def layer(net):
        #net = keras.layers.BatchNormalization(momentum=0.9)(net)
        #net = keras.layers.Conv2D(*args, **kwargs)(net)
        return net
    return layer

def _bn_deconv(*args, **kwargs):
    def layer(net):
        #net = keras.layers.BatchNormalization(momentum=0.9)(net)
        #net = keras.layers.Conv2DTranspose(*args, **kwargs)(net)
        return net
    return layer

def conv(net, **kwargs):
    return tf.layers.conv2d(net, **kwargs)

def unit_norm(x, epslon=1e-12):
    """assumes tensor of shape (n_samples, height, width)."""
    minn = tf.reshape(
        tf.reduce_min(x, reduction_indices=[1, 2, 3]),
        [-1, 1, 1, 1])
    maxx = tf.reshape(
        tf.reduce_max(x, reduction_indices=[1, 2, 3]),
        [-1, 1, 1, 1])
    norm = (x - minn)/tf.maximum(maxx - minn, epslon)
    return norm

def inception(
        net, pool_red, conv1x1, conv3x3_red, conv3x3, conv5x5_red, conv5x5):

    pool_layer = tf.layers.max_pooling2d(net,
        pool_size=(2, 2), strides=(1, 1), padding="same")
    pool_layer = conv(pool_layer, filters=pool_red,
        kernel_size=(1, 1), activation=tf.nn.relu, padding="same")

    conv1x1_layer = conv(net, filters=conv1x1,
        kernel_size=(1, 1), activation=tf.nn.relu, padding="same")

    conv3x3_layer = conv(net, filters=conv3x3_red,
        kernel_size=(1, 1), activation=tf.nn.relu, padding="same")
    conv3x3_layer = conv(conv3x3_layer, filters=conv3x3,
        kernel_size=(3, 3), activation=tf.nn.relu, padding="same")

    conv5x5_layer = conv(net, filters=conv5x5_red,
        kernel_size=(1, 1), activation=tf.nn.relu, padding="same")
    conv5x5_layer = conv(conv5x5_layer, filters=conv5x5,
        kernel_size=(5, 5), activation=tf.nn.relu, padding="same")

    concat_layer = tf.concat(
        [pool_layer, conv1x1_layer, conv3x3_layer, conv5x5_layer], axis=-1)

    return concat_layer

def _build_graph():
    params = {}
    #placeholders
    params["x"] = tf.placeholder("float32", shape=(None, 3, None, None),
        name="x")
    params["y_true"] = tf.placeholder("float32", shape=(None, 1, None, None),
        name="y_true")

    #transposing
    x = tf.transpose(params["x"], [0, 2, 3, 1])
    y_true = tf.transpose(params["y_true"], [0, 2, 3, 1])

    #learning phase
    params["learning_phase"] = tf.placeholder("bool")

    #building net
    net = x

    #first layer
    net = conv(net,
        filters=48, kernel_size=(3, 3), activation=tf.nn.relu, padding="same")
    net = tf.layers.max_pooling2d(net,
        pool_size=(2, 2), strides=(2, 2))

    #second layer
    net = conv(net,
        filters=64, kernel_size=(3, 3), activation=tf.nn.relu, padding="same")
    net = conv(net,
        filters=96, kernel_size=(3, 3), activation=tf.nn.relu, padding="same")
    net = tf.layers.max_pooling2d(net,
        pool_size=(2, 2), strides=(2, 2))

    #third layer
    net = conv(net,
        filters=128, kernel_size=(3, 3), activation=tf.nn.relu, padding="same")
    net = conv(net,
        filters=128, kernel_size=(3, 3), activation=tf.nn.relu, padding="same")
    net = conv(net,
        filters=144, kernel_size=(3, 3), activation=tf.nn.relu, padding="same")
    net = conv(net,
        filters=144, kernel_size=(3, 3), activation=tf.nn.relu, padding="same")
    net = tf.layers.max_pooling2d(net,
        pool_size=(2, 2), strides=(2, 2))

    #fourth layer
    net = inception(net, 96, 128, 96, 192, 48, 96)
    net = inception(net, 64, 128, 80, 160, 24, 48)
    net = inception(net, 64, 128, 80, 160, 24, 48)
    net = inception(net, 64, 128, 96, 192, 28, 56)
    net = inception(net, 64, 128, 96, 192, 28, 56)
    net = inception(net, 64, 128, 112, 224, 32, 64)
    net = inception(net, 64, 128, 112, 224, 32, 64)
    net = inception(net, 112, 160, 128, 256, 40, 80)

    #last layer
    net = conv(net,
        filters=1, kernel_size=(1, 1), activation=tf.nn.relu, padding="same")
    #net = unit_norm(net)
    y_pred = net

    #counting number of params
    print("n. params: {}".format(
        np.sum(
            [np.product(
                [xi.value for xi in x.get_shape()])\
            for x in tf.global_variables()])))

    params["y_pred"] = tf.transpose(y_pred, [0, 3, 1, 2], name="y_pred")

    #cost function
    #_loss = tf.contrib.metrics.streaming_pearson_correlation(y_pred, y_true)
    _loss = tf.losses.mean_squared_error(labels=y_true, predictions=y_pred)
    params["loss"] = tf.reduce_mean(_loss, name="loss")

    params["learning_rate"] = tf.placeholder("float32")

    #update step
    extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(extra_update_ops):
        params["update"] = tf.train.AdamOptimizer(params["learning_rate"],
            name="update").minimize(params["loss"])

    #calculating metrics
    #cc = tf.contrib.metrics.streaming_pearson_correlation(y_pred, y_true)

    #metrics
    params["metrics"] = {
        "loss": params["loss"],
        #"cc": cc,
    }

    return params

#arguments for load
model = {
    #this function builds the graph of the model.
    #it receives an optional (tensorflow) pre-graph argument or create a new.
    #it must return a dictionary mapping all the model parameters defined
    #in model.MetaModel.PARAMS_KEYS.
    "build_graph_fn": _build_graph,
}

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
