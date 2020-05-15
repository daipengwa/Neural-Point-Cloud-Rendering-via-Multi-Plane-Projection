from __future__ import division
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

def lrelu(x):
    return tf.maximum(x * 0.2, x)


def upsample_and_concat_2D(x1, x2, output_channels):

    upsample = tf.image.resize_bilinear(x1, size=[tf.shape(x2)[1], tf.shape(x2)[2]])
    deconv_output = tf.concat([upsample, x2], axis=3)
    deconv_output.set_shape([None, None, None, output_channels * 2])
    return deconv_output


def upsample_and_concat_3D(x1, x2, output_channels, in_channels, only_depth_dims=False):

    pool_size = 2
    if only_depth_dims:
        deconv_filter = tf.Variable(tf.truncated_normal([pool_size, 1, 1, output_channels, in_channels], stddev=0.02))
        deconv = tf.nn.conv3d_transpose(x1, deconv_filter, tf.shape(x2), strides=[1, pool_size, 1, 1, 1])
        deconv_out = tf.concat([deconv, x2], axis=4)
    else:
        deconv_filter = tf.Variable(tf.truncated_normal([pool_size, pool_size, pool_size, output_channels, in_channels], stddev=0.02))
        deconv = tf.nn.conv3d_transpose(x1, deconv_filter, tf.shape(x2), strides=[1, pool_size, pool_size, pool_size, 1])
        deconv_out = tf.concat([deconv, x2], axis=4)

    return deconv_out


def calculate_num_parameter(var_lists):

    total_num = 0
    for var in var_lists:
        shape = var.get_shape()
        var_para = 1

        for dims in shape:
            var_para *= int(dims)

        total_num += var_para

    return total_num


def neural_render(input, reuse=False, use_dilation=True):
    if reuse:
        tf.get_variable_scope().reuse_variables()

    conv0 = tf.layers.conv3d(inputs=input, kernel_size=[1, 1, 1], filters=16, strides=1, activation=lrelu, padding='same', name='conv0')

    conv1 = tf.layers.conv3d(inputs=conv0, kernel_size=[3, 3, 3], filters=32, strides=1, activation=lrelu, padding='same', name='conv1')
    pool1 = tf.layers.max_pooling3d(inputs=conv1, pool_size=[2, 2, 2], strides=2, padding='same', name='pool1')

    conv2 = tf.layers.conv3d(inputs=pool1, kernel_size=[3, 3, 3], filters=32, strides=1, activation=lrelu, padding='same', name='conv2')
    pool2 = tf.layers.max_pooling3d(inputs=conv2, pool_size=[2, 1, 1], strides=[2, 1, 1], padding='same', name='pool2')

    conv3 = tf.layers.conv3d(inputs=pool2, kernel_size=[3, 3, 3], filters=64, strides=1, activation=lrelu, padding='same', name='conv3')
    pool3 = tf.layers.max_pooling3d(inputs=conv3, pool_size=[2, 2, 2], strides=2, padding='same', name='pool3')

    conv4 = tf.layers.conv3d(inputs=pool3, kernel_size=[3, 3, 3], filters=64, strides=1, activation=lrelu, padding='same', name='conv4')
    pool4 = tf.layers.max_pooling3d(inputs=conv4, pool_size=[2, 1, 1], strides=[2, 1, 1], padding='same', name='pool4')

    conv5 = tf.layers.conv3d(inputs=pool4, kernel_size=[3, 3, 3], filters=128, strides=1, activation=lrelu, padding='same', name='conv5')
    pool5 = tf.layers.max_pooling3d(inputs=conv5, pool_size=[2, 2, 2], strides=2, padding='same', name='pool5')

    if use_dilation:
        conv6 = tf.layers.conv3d(inputs=pool5, kernel_size=[1, 3, 3], filters=128, dilation_rate=[1, 2, 2], strides=1, activation=lrelu, padding='same', name='conv6')
    else:
        conv6 = tf.layers.conv3d(inputs=pool5, kernel_size=[1, 3, 3], filters=128, dilation_rate=[1, 1, 1], strides=1, activation=lrelu, padding='same', name='conv6')

    up1 = upsample_and_concat_3D(conv6, conv5, output_channels=128, in_channels=128, only_depth_dims=False)
    conv7 = tf.layers.conv3d(inputs=up1, kernel_size=[3, 3, 3], filters=128, strides=1, activation=lrelu, padding='same', name='conv7')

    up2 = upsample_and_concat_3D(conv7, conv4, output_channels=64, in_channels=128, only_depth_dims=True)
    conv8 = tf.layers.conv3d(inputs=up2, kernel_size=[3, 3, 3], filters=64, strides=1, activation=lrelu, padding='same', name='conv8')

    up3 = upsample_and_concat_3D(conv8, conv3, output_channels=64, in_channels=64, only_depth_dims=False)
    conv9 = tf.layers.conv3d(inputs=up3, kernel_size=[3, 3, 3], filters=64, strides=1, activation=lrelu, padding='same', name='conv9')

    up4 = upsample_and_concat_3D(conv9, conv2, output_channels=32, in_channels=64, only_depth_dims=True)
    conv10 = tf.layers.conv3d(inputs=up4, kernel_size=[3, 3, 3], filters=32, strides=1, activation=lrelu, padding='same', name='conv10')

    up5 = upsample_and_concat_3D(conv10, conv1, output_channels=32, in_channels=32, only_depth_dims=False)
    conv11 = tf.layers.conv3d(inputs=up5, kernel_size=[3, 3, 3], filters=32, strides=1, activation=lrelu, padding='same', name='conv11')

    conv12 = tf.layers.conv3d(inputs=conv11, kernel_size=[1, 1, 1], filters=3, strides=1, activation=None, padding='same', name='conv12')
    conv12_1 = tf.layers.conv3d(inputs=conv11, kernel_size=[1, 1, 1], filters=1, strides=1, activation=None, padding='same', name='conv12_1')

    weight = tf.nn.softmax(conv12_1, dim=1)

    final = tf.reduce_sum(conv12*weight, axis=1)

    return conv12, weight, final




if __name__ == '__main__':

    input = tf.placeholder(dtype=tf.float32, shape=[1, 32, 480, 640, 11])
    mask = tf.placeholder(dtype=tf.float32, shape=[1, 32, 480, 640, 1])

    images, weights, out = neural_render(input=input, reuse=False, use_dilation=True)

    var_lists = tf.trainable_variables()
    num_parameter = calculate_num_parameter(var_lists)
















