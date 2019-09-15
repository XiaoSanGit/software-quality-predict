import numpy as np
import tensorflow as tf

# class batch_norm(object):
#     """Code modification of http://stackoverflow.com/a/33950177"""
#     def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
#         with tf.variable_scope(name):
#             self.epsilon = epsilon
#             self.momentum = momentum
#
#             self.ema = tf.train.ExponentialMovingAverage(decay=self.momentum)
#             self.name = name
#
#     def __call__(self, x, train=True):
#         shape = x.get_shape().as_list()
#
#         if train:
#             with tf.variable_scope(self.name) as scope:
#                 self.beta = tf.get_variable("beta", [shape[-1]],
#                                     initializer=tf.constant_initializer(0.))
#                 self.gamma = tf.get_variable("gamma", [shape[-1]],
#                                     initializer=tf.random_normal_initializer(1., 0.02))
#
#                 batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
#                 ema_apply_op = self.ema.apply([batch_mean, batch_var])
#                 self.ema_mean, self.ema_var = self.ema.average(batch_mean), self.ema.average(batch_var)
#
#                 with tf.control_dependencies([ema_apply_op]):
#                     mean, var = tf.identity(batch_mean), tf.identity(batch_var)
#         else:
#             mean, var = self.ema_mean, self.ema_var
#
#         normed = tf.nn.batch_norm_with_global_normalization(
#                 x, mean, var, self.beta, self.gamma, self.epsilon, scale_after_normalization=True)
#
#         return normed

def make_cpu_variables(name, shape, initializer=tf.contrib.layers.variance_scaling_initializer(), trainable=True):
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, initializer=initializer, trainable=trainable)
    return var

def batch_normal(x, is_train, name, activation_fn=None):
    """
    Function for batch normalization

    Input:
    --- x: input, 4-D Tensor, with shape [bsize, height, width, channel]
    --- is_train: Is training or not
    --- name: Layer name
    --- activation_fn: Activation function
    Output:
    --- outputs: Output of batch normalization
    """
    with tf.name_scope(name), tf.variable_scope(name):
        outputs = tf.contrib.layers.batch_norm(x,
                                               decay=0.999,
                                               scale=True,
                                               activation_fn=activation_fn,
                                               is_training=is_train)
        return outputs

# common conv1d from residual attention network.
def conv1(x,
         k,
         c_o,
         s,
         name,
         relu,
         group=1,
         bias_term=False,
         padding="SAME",
         trainable=True):
    """
    Function for convolutional layer

    Input:
    --- x: Layer input, 3-D Tensor, with shape [bsize, width, channals] ->modules, features
    --- k: Width of kernels
    --- c_o: Amount of kernels, channels out
    --- s: Stride
    --- name: Layer name
    --- relu: Do relu or not
    --- group: Amount of groups
    --- bias_term: Add bias or not
    --- padding: Padding method, SAME or VALID
    --- trainable: Whether the parameters in this layer are trainable
    Output:
    --- outputs: Output of the convolutional layer
    """
    with tf.name_scope(name), tf.variable_scope(name):
        # Get the input channel
        c_i = x.get_shape()[-1]/group
        # Create the weights, with shape [k_h, k_w, c_i, c_o]
        weights = make_cpu_variables("weights", [k, c_i, c_o], trainable=trainable)
        # Create a function for convolution calculation
        def conv1d(i, w):
            return tf.nn.conv1d(i, w, s, padding)
        # If we don't need to divide this convolutional layer
        if group == 1:
            outputs = conv1d(x, weights)
        # If we need to divide this convolutional layer
        else:
            # Split the input and weights
            group_inputs = tf.split(x, group, 3, name="split_inputs")
            group_weights = tf.split(weights, group, 3, name="split_weights")
            group_outputs = [conv1d(i, w) for i, w in zip(group_inputs, group_weights)]
            # Concatenate the groups
            outputs = tf.concat(group_outputs, 3)
        if bias_term:
            # Create the biases, with shape [c_o]
            biases = make_cpu_variables("biases", [c_o], trainable=trainable)
            # Add the biases
            outputs = tf.nn.bias_add(outputs, biases)
        if relu:
            # Nonlinear process
            outputs = tf.nn.relu(outputs)
        # Return layer's output
        return outputs

def residual_unit(x, ci, co, name,k=3, stride=1,is_train=True):
    """
    Implementation of Residual Unit
    Input:
    --- x: Unit input, 3-D Tensor, with shape [bsize,  width, channel]
    --- k : kernel_size
    --- ci: Input channels
    --- co: Output channels
    --- name: Unit name
    --- stride: Convolution stride
    Output:
    --- outputs: Unit output
    """
    with tf.name_scope(name), tf.variable_scope(name):
        # Batch Normalization
        bn_1 = batch_normal(x, is_train, "bn_1", tf.nn.relu)
        # 1x1 Convolution degrade dims.
        conv_1 = conv1(bn_1, 1, co/4, 1, "conv_1", relu=False)
        # Batch Normalization
        bn_2 = batch_normal(conv_1, is_train, "bn_2", tf.nn.relu)
        # 3x3 Convolution
        conv_2 = conv1(bn_2, k,  co/4,  stride, "conv_2", relu=False)
        # Batch Normalization
        bn_3 = batch_normal(conv_2, is_train, "bn_3", tf.nn.relu)
        # 1x1 Convolution
        conv_3 = conv1(bn_3, 1, co,  1, "conv_3", relu=False)
        # Skip connection
        if co != ci or stride > 1:
            skip = conv1(bn_1, 1,  co,  stride, "conv_skip", relu=False)
        else:
            skip = x
        outputs = tf.add(conv_3, skip, name="fuse")
        return outputs


def relu(x, name):
    """
    Function for relu layer

    Input:
    --- x: Layer input, 4-D Tensor, with shape [bsize, height, width, channel]
    --- name: Layer name
    Output:
    --- outputs: Output of the relu layer
    """

    with tf.name_scope(name):
        outputs = tf.nn.relu(x)
        # Return layer's output
        return outputs

# standard convolution layer
def conv2d(x, inputFeatures, outputFeatures, name):
    with tf.variable_scope(name):
        w = tf.get_variable("w",[5,5,inputFeatures, outputFeatures], initializer=tf.truncated_normal_initializer(stddev=0.02))
        b = tf.get_variable("b",[outputFeatures], initializer=tf.constant_initializer(0.0))
        conv = tf.nn.conv2d(x, w, strides=[1,2,2,1], padding="SAME") + b
        return conv

def conv_transpose(x, outputShape, name):
    with tf.variable_scope(name):
        # h, w, out, in
        w = tf.get_variable("w",[5,5, outputShape[-1], x.get_shape()[-1]], initializer=tf.truncated_normal_initializer(stddev=0.02))
        b = tf.get_variable("b",[outputShape[-1]], initializer=tf.constant_initializer(0.0))
        convt = tf.nn.conv2d_transpose(x, w, output_shape=outputShape, strides=[1,2,2,1])
        return convt

def deconv2d(input_, output_shape,
             k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
             name="deconv2d"):
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [k_h, k_h, output_shape[-1], input_.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev))

        deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape, strides=[1, d_h, d_w, 1])

        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

        return deconv

# leaky reLu unit
def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)

# fully-conected layer
def dense(x, inputFeatures, outputFeatures, scope=None, with_w=False):
    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [inputFeatures, outputFeatures], tf.float32, tf.random_normal_initializer(stddev=0.02))
        bias = tf.get_variable("bias", [outputFeatures], initializer=tf.constant_initializer(0.0))
        if with_w:
            return tf.matmul(x, matrix) + bias, matrix, bias
        else:
            return tf.matmul(x, matrix) + bias


def collect_scalar_summaries(scalar_list):
    summaries = []
    for s in scalar_list:
        if hasattr(s, 'name'):
            summaries.append(tf.summary.scalar(s.name.split('/')[-1], s))
    return summaries

def collect_hist_summaries(hist_list):
    summaries = []
    for name, v in hist_list:
        summaries.append(tf.summary.histogram(name, v))
    return summaries