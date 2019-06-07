# coding: utf-8

from __future__ import division, print_function

import tensorflow.contrib.slim as slim
import tensorflow as tf

def conv2d(inputs, filters, kernel_size, strides=1):  # stride>1时padding，valid卷积实现same
    def _fixed_padding(inputs, kernel_size):
        pad_total = kernel_size - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg

        padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                                        [pad_beg, pad_end], [0, 0]], mode='CONSTANT')
        return padded_inputs

    if strides > 1:
        inputs = _fixed_padding(inputs, kernel_size)
    inputs = slim.conv2d(inputs, filters, kernel_size, stride=strides,
                         padding=('SAME' if strides == 1 else 'VALID'))
    return inputs


def darknet53_body(inputs):
    def res_block(inputs, filters):  # same卷积 先1x1降channel再3x3升回channel，再残差连接
        shortcut = inputs
        net = conv2d(inputs, filters * 1, 1)
        net = conv2d(net, filters * 2, 3)

        net = net + shortcut

        return net

    # first two conv2d layers
    net = conv2d(inputs, 32, 3, strides=1)  # same:416*416*32
    net = conv2d(net, 64, 3, strides=2)  # padding_valid:208*208*64

    # res_block * 1
    net = res_block(net, 32)  # 208*208*64->same:208*208*32->same:208*208*64

    net = conv2d(net, 128, 3, strides=2)  # padding_valid:104*104*128

    # res_block * 2
    for i in range(2):
        net = res_block(net, 64)  # 104*104*128

    net = conv2d(net, 256, 3, strides=2)  # padding_valid:52*52*256

    # res_block * 8
    for i in range(8):
        net = res_block(net, 128)  # 52*52*256

    route_1 = net
    net = conv2d(net, 512, 3, strides=2)  # padding_valid:26*26*512

    # res_block * 8
    for i in range(8):
        net = res_block(net, 256)  # 26*26*512

    route_2 = net
    net = conv2d(net, 1024, 3, strides=2)  # padding_valid:13*13*1024

    # res_block * 4
    for i in range(4):
        net = res_block(net, 512)  # 13*13*1024
    route_3 = net

    return route_1, route_2, route_3


def yolo_block(inputs, filters):  # 1x1->3x3->1x1->3x3->1x1(route)------->3x3(net)
    net = conv2d(inputs, filters * 1, 1)
    net = conv2d(net, filters * 2, 3)
    net = conv2d(net, filters * 1, 1)
    net = conv2d(net, filters * 2, 3)
    net = conv2d(net, filters * 1, 1)
    route = net
    net = conv2d(net, filters * 2, 3)
    return route, net


def upsample_layer(inputs, out_shape):  # 最近邻采样resize_image
    new_height, new_width = out_shape[1], out_shape[2]
    # NOTE: here height is the first
    inputs = tf.image.resize_nearest_neighbor(inputs, (new_height, new_width), align_corners=True, name='upsampled')
    return inputs


