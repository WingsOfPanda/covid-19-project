#
# Author:
#    wingsofpanda--熊猫之翼
# Function:
#   denseNet - zoo
# Date:
#    June 30th, 2019
# Version:
#    1.1.0
#


import tensorflow as tf
from tensorflow.python.training.moving_averages import assign_moving_average
slim = tf.contrib.slim
from tensorflow import layers
from tensorflow import reduce_sum
from assests.basic_blocks import conv_operation_modularitied_3d
import numpy as np
import args



def bottleneck_layer(x, filters, name_blk='', reuse=tf.AUTO_REUSE):

    x = conv_operation_modularitied_3d(x, filters, [1,1,1], tasks=[3,1,2], stride=1, activation='relu', normalizer='GN', reuse=reuse, scope=name_blk+'conv_size1_op')
    x = conv_operation_modularitied_3d(x, filters, [3,3,3], tasks=[3,1,2], stride=1, activation='relu', normalizer='GN', reuse=reuse, scope=name_blk+'conv_size3_op')

    return x


def dense_block(x, num_depths=2, filters=args.growth_rate, activation = 'relu', reuse=tf.AUTO_REUSE, scope=''):

    for i in range(num_depths):
        _x = bottleneck_layer(x, filters, name_blk='{}_layer/{}'.format(scope,i), reuse=reuse)
        _x = tf.concat([_x, x], axis=-1)
        x = _x
    return x

def transition_block(x, name_blk = '',  reuse=tf.AUTO_REUSE):

    tensorshape = x.get_shape().as_list()
    nb_channel = tensorshape[4]
    x = conv_operation_modularitied_3d(x, nb_channel, [1,1,1], tasks=[3,1,2], stride=1, activation='relu', normalizer='GN', reuse=reuse, scope=name_blk+'transition_size1_op')
    x = slim.avg_pool3d(x, [2,2,2], stride=[2,2,2], padding = 'SAME', scope=name_blk+'transition_avg_pool3d')
    return x

def global_avg_pool_with_fixed_stride(x, scope='', stride=7, reuse=tf.AUTO_REUSE, all_reduce=True):
    if all_reduce:
        x = tf.reduce_mean(x, axis=[1,2,3])  # GlobalAveragePulling3D in a cheap way
    else: 
        x = slim.avg_pool3d(x, stride, stride=stride, padding='SAME', scope=scope+'stride_{}_global_pool3d'.format(stride))

    return x


def dense_net_121(input, reuse=tf.AUTO_REUSE):

    _input = tf.identity(input, name='input')

    x = conv_operation_modularitied_3d(_input, 32, [7,7,7], tasks=[3,1,2], stride=[2,2,1], activation='relu', normalizer='GN', reuse=reuse, scope='all_start')
    x = slim.max_pool3d(x, [3,3,3], stride=[2,2,2], padding = 'SAME', scope='all_start_maxpool')

    x = dense_block(x, num_depths=6, scope='dense_blk/1', reuse=reuse)
    x = transition_block(x, name_blk = 'trans_blk/1', reuse=reuse)

    x = dense_block(x, num_depths=12, scope='dense_blk/2', reuse=reuse)
    x = transition_block(x, name_blk = 'trans_blk/2', reuse=reuse)


    x = dense_block(x, num_depths=24, scope='dense_blk/3', reuse=reuse)
    x = transition_block(x, name_blk = 'trans_blk/3', reuse=reuse)

    x = dense_block(x, num_depths=16, scope='dense_blk/4', reuse=reuse)
    x = transition_block(x, name_blk = 'trans_blk/4', reuse=reuse)

    x = global_avg_pool_with_fixed_stride(x, scope='global_pool_2end', stride=7, reuse=reuse, all_reduce=True)

    x = slim.flatten(x)
    # print('flatten done with size ', x.shape)
    pred = slim.fully_connected(x, args.nb_classes, activation_fn=None, scope='fully_connected', reuse=reuse)
    pred_softmax = tf.nn.softmax(pred)
    # print('final output size ', output.shape)
    output = tf.identity(pred, name='output')
    output_softmax = tf.identity(pred_softmax, name='output_softmax')

    return output, output_softmax


def dense_net_264(input, reuse=tf.AUTO_REUSE):
    
    _input = tf.identity(input, name='input')

    x = conv_operation_modularitied_3d(_input, 32, [7,7,7], tasks=[3,1,2], stride=[2,2,1], activation='relu', normalizer='GN', reuse=reuse, scope='all_start')
    x = slim.max_pool3d(x, [3,3,3], stride=[2,2,2], padding = 'SAME', scope='all_start_maxpool')

    x = dense_block(x, num_depths=6, scope='dense_blk/1', reuse=reuse)
    x = transition_block(x, name_blk = 'trans_blk/1', reuse=reuse)

    x = dense_block(x, num_depths=12, scope='dense_blk/2', reuse=reuse)
    x = transition_block(x, name_blk = 'trans_blk/2', reuse=reuse)


    x = dense_block(x, num_depths=64, scope='dense_blk/3', reuse=reuse)
    x = transition_block(x, name_blk = 'trans_blk/3', reuse=reuse)

    x = dense_block(x, num_depths=48, scope='dense_blk/4', reuse=reuse)
    x = transition_block(x, name_blk = 'trans_blk/4', reuse=reuse)

    x = global_avg_pool_with_fixed_stride(x, scope='global_pool_2end', stride=7, reuse=reuse, all_reduce=True)

    x = slim.flatten(x)
    # print('flatten done with size ', x.shape)
    pred = slim.fully_connected(x, args.nb_classes, activation_fn=None, scope='fully_connected', reuse=reuse)
    pred_softmax = tf.nn.softmax(pred)
    # print('final output size ', output.shape)
    output = tf.identity(pred, name='output')
    output_softmax = tf.identity(pred_softmax, name='output_softmax')

    return output, output_softmax