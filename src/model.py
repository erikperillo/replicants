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
Module for representing Tensorflow/keras meta-models.
"""

import tensorflow as tf
import numpy as np
import shutil
import os

def get_param_from_coll(coll_key, index=0, max_coll_size=1, graph=None):
    """
    Gets index'th parameter from collection 'coll_key' in graph.
    """
    graph = tf.get_default_graph() if graph is None else graph
    coll = graph.get_collection(coll_key)
    assert max_coll_size is None or len(coll) <= max_coll_size
    obj = coll[index]
    return obj

def save(sess, save_dir, overwrite=False, **builder_kwargs):
    """
    Saves metagraph and weights of current session into directory save_dir.
    """
    if not "tags" in builder_kwargs:
        builder_kwargs["tags"] = []
    if os.path.isdir(save_dir):
        if overwrite:
            shutil.rmtree(save_dir)
        else:
            raise Exception("'{}' exists".format(save_dir))
    builder = tf.saved_model.builder.SavedModelBuilder(save_dir)
    builder.add_meta_graph_and_variables(sess, **builder_kwargs)
    builder.save()

def load(sess, save_dir, **builder_kwargs):
    """
    Loads metagraph and weights to current session from data in save_dir.
    """
    if not "tags" in builder_kwargs:
        builder_kwargs["tags"] = []
    tf.saved_model.loader.load(sess, export_dir=save_dir, **builder_kwargs)

def _var_summaries(var):
    """
    Attaches a lot of summaries to a Tensor (for TensorBoard visualization).
    """
    with tf.name_scope("summaries"):
        mean = tf.reduce_mean(var)
        tf.summary.scalar("mean", mean)
        with tf.name_scope("stddev"):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar("stddev", stddev)
        tf.summary.scalar("max", tf.reduce_max(var))
        tf.summary.scalar("min", tf.reduce_min(var))
        tf.summary.histogram("histogram", var)

def conv(net, *args, **kwargs):
    with tf.name_scope("conv"):
        net = tf.layers.conv2d(net, *args, **kwargs)
    return net

def upconv(net, *args, **kwargs):
    with tf.name_scope("upconv"):
        net = tf.layers.conv2d_transpose(net, *args, **kwargs)
    return net

def max_pool(net, *args, **kwargs):
    with tf.name_scope("max_pool"):
        net = tf.layers.max_pooling2d(net, *args, **kwargs)
    return net

def batch_norm(net, *args, **kwargs):
    with tf.name_scope("batch_norm"):
        net = tf.layers.batch_normalization(net, *args, **kwargs)
    return net

def concat(net_a, net_b):
    with tf.name_scope("concat"):
        net = tf.concat([net_a, net_b], axis=-1)
    return net

def batch_norm_conv(net, training, *args, **kwargs):
    with tf.name_scope("batch_norm-conv"):
        net = batch_norm(net, training=training)
        net = conv(net, *args, **kwargs)
    return net

def batch_norm_upconv(net, training, *args, **kwargs):
    with tf.name_scope("batch_norm-upconv"):
        net = batch_norm(net, training=training)
        net = upconv(net, *args, **kwargs)
    return net

def unet_input_block(net, n_filters, training):
    with tf.name_scope("input_block"):
        #first layer without batch norm
        net = conv(net, filters=n_filters[0],
            kernel_size=(3, 3), activation=tf.nn.relu, padding="same")
        #second layer, saving output to be returned
        net = batch_norm_conv(net, training, filters=n_filters[1],
            kernel_size=(3, 3), activation=tf.nn.relu, padding="same")
        middle_block = net
        #third layer
        net = batch_norm_conv(net, training, filters=n_filters[2],
            kernel_size=(3, 3), activation=tf.nn.relu, padding="same")
        #downsampling with maxpool
        net = max_pool(net, pool_size=(2, 2), strides=(2, 2))
    return net, middle_block

def unet_block_downsample(net, n_filters, training):
    with tf.name_scope("block_downsample"):
        #first layer
        net = batch_norm_conv(net, training, filters=n_filters[0],
            kernel_size=(3, 3), activation=tf.nn.relu, padding="same")
        #second layer, saving output to be returned
        net = batch_norm_conv(net, training, filters=n_filters[1],
            kernel_size=(3, 3), activation=tf.nn.relu, padding="same")
        middle_block = net
        #third layer
        net = batch_norm_conv(net, training, filters=n_filters[2],
            kernel_size=(3, 3), activation=tf.nn.relu, padding="same")
        #downsampling with maxpool
        net = max_pool(net, pool_size=(2, 2), strides=(2, 2))
    return net, middle_block

def unet_block_upsample(net, n_filters, training):
    with tf.name_scope("block_upsample"):
        #first layer
        net = batch_norm_conv(net, training, filters=n_filters[0],
            kernel_size=(3, 3), activation=tf.nn.relu, padding="same")
        #second layer
        net = batch_norm_conv(net, training, filters=n_filters[1],
            kernel_size=(3, 3), activation=tf.nn.relu, padding="same")
        #third layer, transposed convolution
        net = batch_norm_upconv(net, training, filters=n_filters[2],
            kernel_size=(3, 3), strides=(2, 2), activation=tf.nn.relu,
            padding="same")
    return net

def unet_output_block(net, n_filters, training, last_activation=None):
    with tf.name_scope("output_block"):
        #first layer
        net = batch_norm_conv(net, training, filters=n_filters[0],
            kernel_size=(3, 3), activation=tf.nn.relu, padding="same")
        #second layer
        net = batch_norm_conv(net, training, filters=n_filters[1],
            kernel_size=(3, 3), activation=tf.nn.relu, padding="same")
        #third layer, linear combination with maybe sigmoid
        net = conv(net, filters=1,
            kernel_size=(1, 1), activation=last_activation, padding="same")
    return net

def unet(net, training, name="unet"):
    with tf.name_scope(name):
        #downsampling blocks
        #input
        net, middle_block_1 = unet_input_block(net,
            [32, 32, 32], training)
        net, middle_block_2 = unet_block_downsample(net,
            [48, 48, 48], training)
        net, middle_block_3 = unet_block_downsample(net,
            [64, 64, 64], training)
        net, middle_block_4 = unet_block_downsample(net,
            [96, 96, 96], training)
        net, middle_block_5 = unet_block_downsample(net,
            [128, 128, 128], training)

        #upsampling blocks
        net = concat(unet_block_upsample(net, [180, 180, 180], training),
            middle_block_5)
        net = concat(unet_block_upsample(net, [128, 128, 128], training),
            middle_block_4)
        net = concat(unet_block_upsample(net, [96, 96, 96], training),
            middle_block_3)
        net = concat(unet_block_upsample(net, [64, 64, 64], training),
            middle_block_2)
        net = concat(unet_block_upsample(net, [48, 48, 48], training),
            middle_block_1)

        #output
        net = unet_output_block(net, [32, 32], training, tf.nn.relu)

    return net

def _build_graph():
    """
    Builds graph.
    This function must return a dictionary with values
        for all keys in MetaModel.PARAMS_KEYS.
    """
    params = {}
    #placeholders
    params["x"] = tf.placeholder("float32",
        shape=(None, 3, None, None), name="x")
    params["y_true"] = tf.placeholder("float32",
        shape=(None, 1, None, None), name="y_true")

    #transposing
    x = tf.transpose(params["x"], [0, 2, 3, 1])
    y_true = tf.transpose(params["y_true"], [0, 2, 3, 1])

    #learning phase
    params["learning_phase"] = tf.placeholder("bool")

    #net = unit_norm(net)
    y_pred = unet(x, training=params["learning_phase"])

    #counting number of params
    print("n. params: {}".format(
        np.sum(
            [np.product(
                [xi.value for xi in x.get_shape()])\
            for x in tf.global_variables()])))

    params["y_pred"] = tf.transpose(y_pred, [0, 3, 1, 2], name="y_pred")

    #cost function
    _loss = tf.losses.mean_squared_error(labels=y_true, predictions=y_pred)
    params["loss"] = tf.reduce_mean(_loss, name="loss")

    params["learning_rate"] = tf.placeholder("float32")

    #update step
    extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(extra_update_ops):
        params["update"] = tf.train.AdamOptimizer(params["learning_rate"],
            name="update").minimize(params["loss"])

    #metrics
    params["metrics"] = {
        "loss": params["loss"],
    }

    return params

class MetaModel:
    """
    This class contains meta-information about arbitrary models.
    It builds the tensorflow graph and saves important data for later loading.
    """
    NAME = "model"

    PARAMS_KEYS = {
        #input placeholder tensor
        "x",
        #prediction placeholder tensor
        "y_pred",
        #placeholder for true value of y
        "y_true",
        #loss symbolic function
        "loss",
        #update symbolic function
        "update",
        #variable that is True/1 if training phase and 0 otherwise
        "learning_phase",
        #learning rate
        "learning_rate",
        #dictionary in format metric_name: metric_symbolic_expression
        "metrics",
    }

    @staticmethod
    def get_coll_key_for_param(param_name):
        return "/".join([MetaModel.NAME, "params", param_name])

    def __init__(self):
        """
        Initialization of metamodel.
        Every value except for metrics is a tensorflow op/tensor of the graph:
            x: input tensor.
            y_pred: output tensor, prediction
            y_true: placeholder for true data to be used for comparisons.
            loss: loss function to be optimized during training.
            update: op used to update parameters during training.
            learning_phase: tensor that is 1 iff it's training phase.
            metrics: a dict in format metric_name: metric_tensor.
        """
        self.params = {}

    def build_graph(self, pre_graph=tf.Graph()):
        """
        Calls build_graph_fn and returns built graph.
        """
        graph = tf.get_default_graph() if pre_graph is None else pre_graph
        with graph.as_default():
            self.params = _build_graph()
        assert set(self.params.keys()) == MetaModel.PARAMS_KEYS
        return graph

    def _mk_summaries(self):
        #metrics summaries
        with tf.name_scope(self.get_coll_key_for_param("metrics")):
            for k, v in self.params["metrics"].items():
                if k == "loss":
                    continue
                tf.summary.scalar(k, v)

        #loss summary
        with tf.name_scope(self.get_coll_key_for_param("loss")):
            tf.summary.scalar("loss", self.params["loss"])

        #image summary
        with tf.name_scope(self.get_coll_key_for_param("img")):
            tf.summary.image("true", tf.transpose(
                    self.params["y_true"], [0, 2, 3, 1]), max_outputs=4)
            tf.summary.image("pred", tf.transpose(
                    self.params["y_pred"], [0, 2, 3, 1]), max_outputs=4)
            tf.summary.image("x", tf.transpose(
                    self.params["x"], [0, 2, 3, 1]), max_outputs=4)

        #other params summaries
        for k in {"x", "y_pred", "y_true"}:
            var = self.params[k]
            with tf.name_scope(self.get_coll_key_for_param(k)):
                _var_summaries(var)

    def mk_params_colls(self, graph=None):
        """
        Makes parameters collections in the given graph for model parameters.
        Each parameter is stored in a different collection.
        Parameters can then be further retrieved for retraining/prediction.
        """
        graph = tf.get_default_graph() if graph is None else graph
        coll_keys = set(graph.get_all_collection_keys())

        #creating param collections
        for k, p in self.params.items():
            if k == "metrics":
                continue
            coll_key = self.get_coll_key_for_param(k)
            assert coll_key not in coll_keys
            tf.add_to_collection(coll_key, p)

        #creating metrics collections
        for k, p in self.params["metrics"].items():
            coll_key = self.get_coll_key_for_param("metrics/{}".format(k))
            assert coll_key not in coll_keys
            tf.add_to_collection(coll_key, p)

        self._mk_summaries()

    def set_params_from_colls(self, graph=None):
        """
        Sets tensors/ops from collections contained in graph, saved using
            the method 'mk_params_colls'.
        """
        graph = tf.get_default_graph() if graph is None else graph

        #getting default parameters
        for k in MetaModel.PARAMS_KEYS - {"metrics"}:
            coll_key = self.get_coll_key_for_param(k)
            self.params[k] = get_param_from_coll(coll_key, graph=graph)

        #getting metrics
        self.params["metrics"] = {}
        metrics_coll_key_pattern = self.get_coll_key_for_param("metrics")
        for k in graph.get_all_collection_keys():
            if k.startswith(metrics_coll_key_pattern):
                _k = k.split(metrics_coll_key_pattern)[1][1:]
                self.params["metrics"][_k] = get_param_from_coll(k, graph=graph)

        #self._mk_summaries()

    def get_train_fn(self, sess):
        """
        Gets train function.
        """
        def train_fn(x, y_true, extra_feed_dict={}):
            feed_dict = {
                self.params["x"]: x,
                self.params["y_true"]: y_true,
                self.params["learning_phase"]: True,
            }
            feed_dict.update(extra_feed_dict)
            __, loss = sess.run([self.params["update"], self.params["loss"]],
                feed_dict=feed_dict)
            return loss

        return train_fn

    def get_test_fn(self, sess):
        """
        Gets test function.
        """
        def test_fn(x, y_true, extra_feed_dict={}):
            feed_dict = {
                self.params["x"]: x,
                self.params["y_true"]: y_true,
                self.params["learning_phase"]: False,
            }
            feed_dict.update(extra_feed_dict)
            metrics = sess.run(list(self.params["metrics"].values()),
                feed_dict=feed_dict)
            return metrics

        return test_fn

    def get_pred_fn(self, sess):
        """
        Gets prediction function.
        """
        def pred_fn(x, extra_feed_dict={}):
            feed_dict={
                self.params["x"]: x,
                self.params["learning_phase"]: False,
            }
            feed_dict.update(extra_feed_dict)
            pred = sess.run(self.params["y_pred"], feed_dict=feed_dict)
            return pred

        return pred_fn

    def get_summary_fn(self, sess):
        merged = tf.summary.merge_all()
        def summary_fn(x, y_true, extra_feed_dict={}):
            feed_dict={
                self.params["x"]: x,
                self.params["y_true"]: y_true,
                self.params["learning_phase"]: False,
            }
            feed_dict.update(extra_feed_dict)
            summ, __ = sess.run(
                [merged, list(self.params["metrics"].values())],
                feed_dict=feed_dict)
            return summ

        return summary_fn
