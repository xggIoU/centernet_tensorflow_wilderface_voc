import tensorflow as tf
import tensorflow.contrib.slim as slim

def shuffle_unit(x, groups):
    with tf.variable_scope('shuffle_unit'):
        n, h, w, c = x.get_shape().as_list()
        x = tf.reshape(x, shape=tf.convert_to_tensor([tf.shape(x)[0], h, w, groups, c // groups]))
        x = tf.transpose(x, tf.convert_to_tensor([0, 1, 2, 4, 3]))
        x = tf.reshape(x, shape=tf.convert_to_tensor([tf.shape(x)[0], h, w, c]))
    return x


def conv_bn_relu(x, out_channel, kernel_size, stride=1, dilation=1):
    with tf.variable_scope(None, 'conv_bn_relu'):
        x = slim.conv2d(x, out_channel, kernel_size, stride, rate=dilation,
                        biases_initializer=None, activation_fn=None)
        x = slim.batch_norm(x, activation_fn=tf.nn.relu, fused=False)
    return x


def conv_bn(x, out_channel, kernel_size, stride=1, dilation=1):
    with tf.variable_scope(None, 'conv_bn'):
        x = slim.conv2d(x, out_channel, kernel_size, stride, rate=dilation,
                        biases_initializer=None, activation_fn=None)
        x = slim.batch_norm(x, activation_fn=None, fused=False)
    return x


def depthwise_conv_bn(x, kernel_size, stride=1, dilation=1):
    with tf.variable_scope(None, 'depthwise_conv_bn'):
        x = slim.separable_conv2d(x, None, kernel_size, depth_multiplier=1, stride=stride,
                                  rate=dilation, activation_fn=None, biases_initializer=None)
        x = slim.batch_norm(x, activation_fn=None, fused=False)
    return x


def resolve_shape(x):
    with tf.variable_scope(None, 'resolve_shape'):
        n, h, w, c = x.get_shape().as_list()
        if h is None or w is None:
            kernel_size = tf.convert_to_tensor([tf.shape(x)[1], tf.shape(x)[2]])
        else:
            kernel_size = [h, w]
    return kernel_size

def global_avg_pool2D(x):
    with tf.variable_scope(None, 'global_pool2D'):
        kernel_size = resolve_shape(x)
        x = slim.avg_pool2d(x, kernel_size, stride=1)
        x.set_shape([None, 1, 1, None])
    return x


def se_unit(x, bottleneck=2):
    with tf.variable_scope(None, 'SE_module'):
        n, h, w, c = x.get_shape().as_list()

        kernel_size = resolve_shape(x)
        x_pool = slim.avg_pool2d(x, kernel_size, stride=1)
        x_pool = tf.reshape(x_pool, shape=[-1, c])
        fc = slim.fully_connected(x_pool, bottleneck, activation_fn=tf.nn.relu,
                                  biases_initializer=None)
        fc = slim.fully_connected(fc, c, activation_fn=tf.nn.sigmoid,
                                  biases_initializer=None)
        if n is None:
            channel_w = tf.reshape(fc, shape=tf.convert_to_tensor([tf.shape(x)[0], 1, 1, c]))
        else:
            channel_w = tf.reshape(fc, shape=[n, 1, 1, c])

        x = tf.multiply(x, channel_w)
    return x

def sa_unit(x):
    with tf.variable_scope(None, 'SA_module'):
        shape=x.get_shape().as_list()
        y=slim.conv2d(x,shape[-1],kernel_size=1,stride=1,biases_initializer=None,activation_fn=None)
        y=slim.batch_norm(y,activation_fn=None, fused=False)
        y=tf.nn.sigmoid(y)
        x=tf.multiply(x,y)
        return x

def deconv_bn_relu(inputs, filters, kernel_size=3, strides=2):
    with tf.variable_scope(None, 'deconv_bn_relu'):
        output=slim.conv2d_transpose(inputs,filters,kernel_size=kernel_size,stride=strides,biases_initializer=None,activation_fn=None)
        output = slim.batch_norm(output, activation_fn=tf.nn.relu, fused=False)
        return output

def upsample_layer(inputs, out_shape):  # 最近邻采样resize_image
    new_height, new_width = out_shape[1], out_shape[2]
    # NOTE: here height is the first
    inputs = tf.image.resize_nearest_neighbor(inputs, (new_height, new_width), align_corners=True, name='upsampled')
    return inputs

def shufflenet_v2_block(x, out_channel, kernel_size, stride=1, dilation=1, shuffle_group=2):
    with tf.variable_scope(None, 'shuffle_v2_block'):
        if stride == 1:
            top, bottom = tf.split(x, num_or_size_splits=2, axis=3)

            half_channel = out_channel // 2

            top = conv_bn_relu(top, half_channel, 1)
            top = depthwise_conv_bn(top, kernel_size, stride, dilation)
            top = conv_bn_relu(top, half_channel, 1)

            out = tf.concat([top, bottom], axis=3)
            out = shuffle_unit(out, shuffle_group)

        else:
            half_channel = out_channel // 2
            b0 = conv_bn_relu(x, half_channel, 1)
            b0 = depthwise_conv_bn(b0, kernel_size, stride, dilation)
            b0 = conv_bn_relu(b0, half_channel, 1)

            b1 = depthwise_conv_bn(x, kernel_size, stride, dilation)
            b1 = conv_bn_relu(b1, half_channel, 1)

            out = tf.concat([b0, b1], axis=3)
            out = shuffle_unit(out, shuffle_group)
        return out


