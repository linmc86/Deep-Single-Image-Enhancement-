import tensorflow as tf
from utils.configs import *


def nethighlayer_rgb(input):
    numc = 32
    with tf.variable_scope(name_or_scope="high_level"):
        network = bn(conv_relu(net=input, in_c=3, out_c=numc, padding='SAME', name='conv1'), name='c1_bn')
        network = bn(conv_relu(net=network, in_c=numc, out_c=numc, padding='SAME', name='conv2'), name='c2_bn')
        network = bn(conv_relu(net=network, in_c=numc, out_c=numc, padding='SAME', name='conv3'), name='c3_bn')
        network = bn(conv_relu(net=network, in_c=numc, out_c=numc, padding='SAME', name='conv4'), name='c4_bn')
        network = bn(conv_relu(net=network, in_c=numc, out_c=numc, padding='SAME', name='conv5'), name='c5_bn')
        network = conv_relu(net=network, in_c=numc, out_c=3, padding='SAME', name='conv6')
        output = input + network
    return output


def netbotlayer_rgb(input):
    numc = 32
    with tf.variable_scope(name_or_scope="bot_level"):
        network = bn(conv_relu(net=input, in_c=3, out_c=numc, padding='SAME', name='conv1'), name='c1_bn')
        network = bn(conv_relu(net=network, in_c=numc, out_c=numc, padding='SAME', name='conv2'), name='c2_bn')
        network = bn(conv_relu(net=network, in_c=numc, out_c=numc, padding='SAME', name='conv3'), name='c3_bn')
        network = bn(conv_relu(net=network, in_c=numc, out_c=numc, padding='SAME', name='conv4'), name='c4_bn')
        network = conv_relu(net=network, in_c=numc, out_c=3, padding='SAME', name='conv5')
        output = input + network
    return output


def nethighlayer_gray(input):
    numc = 32
    with tf.variable_scope(name_or_scope="high_level"):
        network = bn(conv_relu(net=input, in_c=1, out_c=numc, padding='SAME', name='conv1'), name='c1_bn')
        network = bn(conv_relu(net=network, in_c=numc, out_c=numc, padding='SAME', name='conv2'), name='c2_bn')
        network = bn(conv_relu(net=network, in_c=numc, out_c=numc, padding='SAME', name='conv3'), name='c3_bn')
        network = bn(conv_relu(net=network, in_c=numc, out_c=numc, padding='SAME', name='conv4'), name='c4_bn')
        network = bn(conv_relu(net=network, in_c=numc, out_c=numc, padding='SAME', name='conv5'), name='c5_bn')
        network = conv1x1(network, 1, name='conv6_1x1', padding='SAME')
        output = input + network
    return output


def netbotlayer_gray(input):
    numc = 32
    with tf.variable_scope(name_or_scope="bot_level"):
        network = bn(conv_relu(net=input, in_c=1, out_c=numc, padding='SAME', name='conv1'), name='c1_bn')
        network = bn(conv_relu(net=network, in_c=numc, out_c=numc, padding='SAME', name='conv2'), name='c2_bn')
        network = bn(conv_relu(net=network, in_c=numc, out_c=numc, padding='SAME', name='conv3'), name='c3_bn')
        network = bn(conv_relu(net=network, in_c=numc, out_c=numc, padding='SAME', name='conv4'), name='c4_bn')
        network = conv1x1(network, 1, name='conv5_1x1', padding='SAME')
        output = input + network
    return output


def netbotlayer_gray_lev_3(input):
    numc = 64
    with tf.variable_scope(name_or_scope="bot_level"):
        network = bn(conv_relu(net=input, in_c=1, out_c=numc, w_size=5, padding='SAME', name='conv1'), name='c1_bn')
        network = bn(conv_relu(net=network, in_c=numc, out_c=numc, w_size=5, padding='SAME', name='conv2'), name='c2_bn')
        network = bn(conv_relu(net=network, in_c=numc, out_c=numc, w_size=5, padding='SAME', name='conv3'), name='c3_bn')
        network = bn(conv_relu(net=network, in_c=numc, out_c=numc, w_size=1, padding='SAME', name='conv4'), name='c4_bn')
        network = conv1x1(network, 1, name='conv5_1x1', padding='SAME')
        output = input + network
    return output


def nethightest_gray(input):
    numc = 32
    with tf.variable_scope(name_or_scope="high_level"):
        network = bn(conv_relu(net=input, in_c=1, out_c=numc, padding='SAME', name='conv1'), name='c1_bn')
        network = bn(conv_relu(net=network, in_c=numc, out_c=numc, padding='SAME', name='conv2'), name='c2_bn')
        network = bn(conv_relu(net=network, in_c=numc, out_c=numc, padding='SAME', name='conv3'), name='c3_bn')
        network = bn(conv_relu(net=network, in_c=numc, out_c=numc, padding='SAME', name='conv4'), name='c4_bn')
        network = bn(conv_relu(net=network, in_c=numc, out_c=1, padding='SAME', name='conv5'), name='c5_bn')
        output = input + network
    return output


def netbottest_gray(input):
    numc = 32
    with tf.variable_scope(name_or_scope="bot_level"):
        network = bn(conv_relu(net=input, in_c=1, out_c=numc, padding='SAME', name='conv1'), name='c1_bn')
        network = bn(conv_relu(net=network, in_c=numc, out_c=numc, padding='SAME', name='conv2'), name='c2_bn')
        network = bn(conv_relu(net=network, in_c=numc, out_c=numc, padding='SAME', name='conv3'), name='c3_bn')
        network = bn(conv_relu(net=network, in_c=numc, out_c=numc, padding='SAME', name='conv4'), name='c4_bn')
        network = bn(conv_relu(net=network, in_c=numc, out_c=1, padding='SAME', name='conv5'), name='c5_bn')
        output = input + network
    return output


def bn(inputs, epsilon=0.01, name='batch_norm'):
    inputs_shape = inputs.get_shape()
    mean, variance = tf.nn.moments(inputs, range(len(inputs_shape.as_list()) - 1))

    output = tf.nn.batch_normalization(inputs, mean, variance, None, None, variance_epsilon=epsilon, name=name)
    return output


def weight_variable(shape, name=None, stddev=0.1):
    initial = tf.truncated_normal(shape, stddev=stddev)
    return tf.Variable(initial, name=name)


def bias_variable(shape, name=None):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)


def conv_relu(net, in_c, out_c, name, w_size=3, strides=1, padding='VALID'):
    w = weight_variable([w_size, w_size, in_c, out_c], name=name)
    b = bias_variable([out_c], name=name)
    network = tf.nn.conv2d(input=net,
                           filter=w,
                           padding=padding,
                           strides=[1, strides, strides, 1],
                           name="{}_conv".format(name),
                           )
    network = tf.nn.leaky_relu(network + b, name="{}_relu".format(name))
    return network


def conv1x1(net, numfilters, name, padding='VALID'):
    return tf.layers.conv2d(net,
                            filters=numfilters,
                            strides=(1, 1),
                            kernel_size=(1, 1),
                            name="{}_conv1x1".format(name), padding=padding)
