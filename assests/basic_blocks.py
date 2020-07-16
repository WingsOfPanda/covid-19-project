

#Author:
#    wingsofpanda -- 熊猫之翼
#Function:
#   collectiong of loss functions that popularly used
#Date:
#    June 11th, 2019
#Version:
#    1.0.0

import json
import os
import tensorflow as tf
import numpy as np
from tensorflow import layers
from tensorflow.python.ops import array_ops
import args
from math import ceil
from assests.normalizer import GN_auto_channel3d as GN_auto_channel
slim = tf.contrib.slim

'''
    basic conv operations. including: normalization, activation, and conv
    '''


def conv_operation_modularitied_3d(x, filters, kernel_size, tasks=[1,2,3,4], stride=1, activation='relu', normalizer='GN', reuse=False, scope=''):
    '''
    tasks order:
        task 1: normalization
        task 2: activation
        taks 3: conv with filters and kernel_size
        taks 4: conv transpose with filters and kernel_size
    '''
    for task in tasks:
        if task == 1:
            if normalizer == 'GN':
                x = GN_auto_channel(x, reuse=reuse, name='{}/GN'.format(scope))
            elif normalizer == 'IN':
                x = slim.instance_norm(x, reuse=reuse, scope='{}/IN'.format(scope))
            elif normalizer == 'FRN':
                # print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  we are using FRN >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
                x = FRN_norm2d(x, nameScope='{}/FRN'.format(scope), reuse=reuse, eps=1e-6)

        elif task == 2:
            if activation == 'relu':
                x = tf.nn.relu(x)
            elif activation == 'softmax':
                x = tf.nn.softmax(x)
            elif activation == 'sigmoid':
                x = tf.nn.sigmoid(x)
        elif task == 3:
            x = slim.conv3d(inputs=x, num_outputs=filters, kernel_size=kernel_size, stride=stride, padding='SAME', activation_fn=None, normalizer_fn=None, weights_regularizer=slim.l2_regularizer(0.0005), reuse=reuse, scope=scope+'conv3d')
        elif task == 4:
            x = slim.conv3d_transpose(inputs=x, num_outputs=filters, kernel_size=kernel_size, stride=stride, padding='SAME', activation_fn=None, normalizer_fn=None, reuse=reuse, scope=scope+'conv3d_transpose')
    return x

'''
    ResNet structure
    '''

def ResBlockCore(x, filter_chain, kernal_chain, reuse=False, scope=''):
    layer = 0
    for filter, kernal in zip(filter_chain, kernal_chain):
        if layer == len(filter_chain) - 1:
            layer += 1
            continue
        x = CNN_operation_modularitied_2d(x, filter, kernal, tasks=[3, 1, 2], reuse=reuse, scope='{}/layer{}/conv_mod'.format(scope, layer))
        layer += 1

    x = CNN_operation_modularitied_2d(x, filter_chain[-1], kernal_chain[-1], tasks=[3], reuse=reuse, scope='{}/layer{}/conv_mod'.format(scope, layer))

    return x

def ResNetBottleNeck(x, filters=16, reuse=False, scope=''):

    tensorshape = x.get_shape().as_list()
    nb_channel = tensorshape[3]

    _x = ResBlockCore(x, [int(nb_channel/2), int(nb_channel/2), nb_channel], [1, 3, 1], reuse=reuse, scope='{}/RBC'.format(scope))
    x = tf.add(x, _x)

    x = CNN_operation_modularitied_2d(x, 3, 3, tasks=[1], stride=1, activation='relu', reuse=reuse, scope='{}/RNBN_OUT'.format(scope))

    return x


'''
    ResNeXt structure
    '''

def ResNeXt_path(x, internal_filters=4, activation='relu', normalizer='GN', reuse=False, scope=''):

    tensorshape = x.get_shape().as_list()
    nb_channel = tensorshape[4]

    x = ResBlockCore(x, [internal_filters, internal_filters, nb_channel], [1, 3, 1], reuse=reuse, scope='{}/RBC'.format(scope))

    return x

def ResNeXtBottleNeck(x, internal_filters=4, reuse=False, scope=''):

    tensorshape = x.get_shape().as_list()
    nb_channel = tensorshape[4]
    cardinality = int(nb_channel/internal_filters)*args.cardinality
    _x = ResNeXt_path(x, internal_filters=4, reuse=reuse, scope='{}/pathHead'.format(scope))
    for i in range(cardinality-1):
        _x = _x + ResNeXt_path(x, internal_filters=4, reuse=reuse, scope='{}/path_{}_body'.format(scope,i))

    _x = CNN_operation_modularitied_2d(_x, tensorshape[4], 3, tasks=[1,2], reuse=reuse, scope='{}/pathTail'.format(scope))

    _x = tf.add(x, _x)

    return _x


# def ResNeXt():

'''
    DenseBlock
    '''


# def test_graph(x, reuse=False):

#     initial_dict = {}
#     initial_dict[0] = x

#     x = conv_3d_modularitied(initial_dict[0], 32, 3, tasks=[3], reuse=reuse, scope='initial_conv')

#     x = ResNeXtBottleNeck(x, internal_filters=4, reuse=reuse, scope='RBN')

#     x = conv_3d_modularitied(x, 2, 3, tasks=[3, 1, 2], reuse=reuse, scope='pre_outConv')

#     output = tf.identity(x, name='output')

#     return output
