import tensorflow as tf
import tensorflow.contrib.slim as slim

def shuffle_unit(x, groups):
    with tf.variable_scope('shuffle_unit'):
        n, h, w, c = x.get_shape().as_list()
        x = tf.reshape(x, shape=tf.convert_to_tensor([tf.shape(x)[0], h, w, groups, c // groups]))
        x = tf.transpose(x, tf.convert_to_tensor([0, 1, 2, 4, 3]))
        x = tf.reshape(x, shape=tf.convert_to_tensor([tf.shape(x)[0], h, w, c]))
    return x

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

def conv_bn_relu(x, out_channel, kernel_size, strides=1, dilation=1):
    with tf.variable_scope(None, 'conv_bn_relu'):
        x = slim.conv2d(x, out_channel, kernel_size, strides, rate=dilation,
                        biases_initializer=None, activation_fn=None)
        x = slim.batch_norm(x, activation_fn=tf.nn.relu, fused=False)
    return x

def conv_bn_sigmoid(x, out_channel, kernel_size, strides=1, dilation=1):
    with tf.variable_scope(None, 'conv_bn_sigmoid'):
        x = slim.conv2d(x, out_channel, kernel_size, strides, rate=dilation,
                        biases_initializer=None, activation_fn=None)
        x = slim.batch_norm(x, activation_fn=tf.nn.sigmoid, fused=False)
    return x

def conv_bn_leakyrelu(x, out_channel, kernel_size, strides=1, dilation=1):
    with tf.variable_scope(None, 'conv_bn_leakyrelu'):
        x = slim.conv2d(x, out_channel, kernel_size, strides, rate=dilation,
                        biases_initializer=None, activation_fn=None)
        x = slim.batch_norm(x, activation_fn=tf.nn.leaky_relu, fused=False)
    return x

def bn_leakyrelu_conv(x, out_channel, kernel_size, strides=1, dilation=1):
    with tf.variable_scope(None, 'bn_leakyrelu_conv'):
        x = slim.batch_norm(x, activation_fn=tf.nn.leaky_relu, fused=False)
        x = slim.conv2d(x, out_channel, kernel_size, strides, rate=dilation,
                        biases_initializer=None, activation_fn=None)
    return x

def bn_relu_conv(x, out_channel, kernel_size, strides=1, dilation=1):
    with tf.variable_scope(None, 'bn_relu_conv'):
        x = slim.batch_norm(x, activation_fn=tf.nn.relu, fused=False)
        x = slim.conv2d(x, out_channel, kernel_size, strides, rate=dilation,
                        biases_initializer=None, activation_fn=None)
    return x

def dropout(x,p=0.7):
    x=slim.dropout(x,keep_prob=p)
    return x

def conv_bn(x, out_channel, kernel_size, strides=1, dilation=1):
    with tf.variable_scope(None, 'conv_bn'):
        x = slim.conv2d(x, out_channel, kernel_size, strides, rate=dilation,
                        biases_initializer=None, activation_fn=None)
        x = slim.batch_norm(x, activation_fn=None, fused=False)
    return x

def bn_leakyrelu(x):
    with tf.variable_scope(None, 'bn_leakyrelu'):
        x = slim.batch_norm(x, activation_fn=tf.nn.leaky_relu, fused=False)
    return x

def bn_relu(x):
    with tf.variable_scope(None, 'bn_relu'):
        x = slim.batch_norm(x, activation_fn=tf.nn.relu, fused=False)
    return x

def conv_relu(x, out_channel, kernel_size, strides=1, dilation=1):
    with tf.variable_scope(None, 'conv_relu'):
        x = slim.conv2d(x, out_channel, kernel_size, strides, rate=dilation, activation_fn=tf.nn.relu)
    return x

def conv_leakyrelu(x, out_channel, kernel_size, strides=1, dilation=1):
    with tf.variable_scope(None, 'conv_leakyrelu'):
        x = slim.conv2d(x, out_channel, kernel_size, strides, rate=dilation, activation_fn=tf.nn.leaky_relu)
    return x

def conv(x, out_channel, kernel_size, strides=1, dilation=1):
    x = slim.conv2d(x, out_channel, kernel_size, strides, rate=dilation,activation_fn=None)
    return x

def maxpool(x,kernel_size,strides=2,padding='same'):
    x=slim.max_pool2d(x,kernel_size,strides,padding=padding)
    return x

def avgpool(x,kernel_size,strides=2,padding='same'):
    x=slim.avg_pool2d(x,kernel_size,strides,padding=padding)
    return x

def get_static_or_dynamic_shape(tensor):
  """
  Returns a list containing static or dynamic values for the dimensions.
  """
  static_tensor_shape = tensor.shape.as_list()
  dynamic_tensor_shape = tf.shape(tensor)
  combined_shape = []
  for index, dim in enumerate(static_tensor_shape):
    if dim is not None:
      combined_shape.append(dim)
    else:
      combined_shape.append(dynamic_tensor_shape[index])
  return combined_shape

def depthwise_conv_bn(x, kernel_size, strides=1, dilation=1):
    with tf.variable_scope(None, 'depthwise_conv_bn'):
        x = slim.separable_conv2d(x, None, kernel_size, depth_multiplier=1, stride=strides,
                                  rate=dilation, activation_fn=None, biases_initializer=None)
        x = slim.batch_norm(x, activation_fn=None, fused=False)
    return x

def depth_bn_point_bn(x,kernel_size,point_filters,strides=1,dilation=1):
    with tf.variable_scope(None, 'depth_bn_point_bn'):
        x = slim.separable_conv2d(x, None, kernel_size, depth_multiplier=1, stride=strides,
                                  rate=dilation, activation_fn=None, biases_initializer=None)
        x = slim.batch_norm(x, activation_fn=None, fused=False)
        x = slim.conv2d(x, point_filters, 1, 1, rate=1,
                        biases_initializer=None, activation_fn=None)
        x = slim.batch_norm(x, activation_fn=None, fused=False)
        return x

def depth_bn_point_bn_relu(x,kernel_size,point_filters,strides=1,dilation=1):
    with tf.variable_scope(None, 'depth_bn_point_bn_relu'):
        x = slim.separable_conv2d(x, None, kernel_size, depth_multiplier=1, stride=strides,
                                  rate=dilation, activation_fn=None, biases_initializer=None)
        x = slim.batch_norm(x, activation_fn=None, fused=False)
        x = slim.conv2d(x, point_filters, 1, 1, rate=1,
                        biases_initializer=None, activation_fn=None)
        x = slim.batch_norm(x, activation_fn=tf.nn.relu, fused=False)
        return x

def depth_bn_relu_point_bn_relu(x,kernel_size,point_filters,strides=1,dilation=1):
    with tf.variable_scope(None, 'depth_bn_relu_point_bn_relu'):
        x = slim.separable_conv2d(x, None, kernel_size, depth_multiplier=1, stride=strides,
                                  rate=dilation, activation_fn=None, biases_initializer=None)
        x = slim.batch_norm(x, activation_fn=tf.nn.relu, fused=False)
        x = slim.conv2d(x, point_filters, 1, 1, rate=1,
                        biases_initializer=None, activation_fn=None)
        x = slim.batch_norm(x, activation_fn=tf.nn.relu, fused=False)
        return x

def global_avg_pool2D(x):
    with tf.variable_scope(None, 'global_pool2D'):
        # n,h,w,c=x.get_shape().as_list
        x = slim.avg_pool2d(x, (x.shape[1],x.shape[2]), stride=1)
    return x

def deconv_bn_relu(inputs, filters, kernel_size=4, strides=2):
    """Paramaters for Deconvolution were chosen to avoid artifacts, following
        link https://distill.pub/2016/deconv-checkerboard/
        """
    with tf.variable_scope(None, 'deconv_bn_relu'):
        output=slim.conv2d_transpose(inputs,filters,kernel_size=kernel_size,stride=strides,biases_initializer=None,activation_fn=None)
        output = slim.batch_norm(output, activation_fn=tf.nn.relu, fused=False)
        return output

def deconv_bn(inputs, filters, kernel_size=4, strides=2):
    with tf.variable_scope(None, 'deconv_bn'):
        output=slim.conv2d_transpose(inputs,filters,kernel_size=kernel_size,stride=strides,biases_initializer=None,activation_fn=None)
        output = slim.batch_norm(output, activation_fn=None, fused=False)
        return output

def upsample_layer(inputs, out_shape):

    new_height, new_width = out_shape[0], out_shape[1]
    inputs = tf.image.resize_bilinear(inputs, (new_height, new_width), align_corners=True, name='upsampled')
    # inputs = tf.image.resize_nearest_neighbor(inputs, (new_height, new_width), align_corners=True, name='upsampled')
    return inputs

def se_unit(x):
    with tf.variable_scope(None, 'se_module'):
        n, h, w, c = x.get_shape().as_list()
        x_pool = slim.avg_pool2d(x, (h,w), stride=1)
        x_pool = tf.reshape(x_pool, shape=[-1, c])
        fc = slim.fully_connected(x_pool, c//8, activation_fn=tf.nn.relu,
                                  biases_initializer=None)
        fc = slim.fully_connected(fc, c, activation_fn=tf.nn.sigmoid,
                                  biases_initializer=None)
        channel_w = tf.reshape(fc, shape=[tf.shape(x)[0], 1, 1, c])
        x = tf.multiply(x, channel_w)

        return x

def se_conv_unit(x):
    with tf.variable_scope(None, 'se_conv_unit'):
        shape=x.get_shape().as_list()
        y = slim.avg_pool2d(x, (shape[1],shape[2]), stride=1)
        y=slim.conv2d(y, shape[-1], 1, 1,activation_fn=None)
        y = slim.batch_norm(y, activation_fn=tf.nn.sigmoid, fused=False)
        x = tf.multiply(x, y)
    return x

def sa_conv_unit(x):
    with tf.variable_scope(None, 'sa_conv_unit'):
        shape=x.get_shape().as_list()
        y=slim.conv2d(x,shape[-1],kernel_size=1,stride=1,biases_initializer=None,activation_fn=None)
        y=slim.batch_norm(y,activation_fn=None, fused=False)
        y=tf.nn.sigmoid(y)
        x=tf.multiply(x,y)
        return x



def atrous_spatial_pyramid_pooling(x):
    """空洞空间金字塔池化
    """
    with tf.variable_scope('ASSP_layers'):

        feature_map_size = tf.shape(x)

        image_level_features = tf.reduce_mean(x, [1, 2], keep_dims=True)
        image_level_features =conv_bn(image_level_features, 256,1, 1)
        image_level_features = upsample_layer(image_level_features, (feature_map_size[1],feature_map_size[2]))

        at_pool1x1   = conv_bn(x, 256,1,1,1)
        at_pool3x3_1 = conv_bn(x, 256,3,1,dilation=6)
        at_pool3x3_2 = conv_bn(x, 256,3,1,dilation=12)
        at_pool3x3_3 = conv_bn(x, 256,3,1,dilation=18)

        net = tf.concat((image_level_features, at_pool1x1, at_pool3x3_1, at_pool3x3_2, at_pool3x3_3), axis=3)

        net = conv_bn(net,256,1,1)

        return net