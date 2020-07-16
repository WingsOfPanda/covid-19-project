
#Author:
#    wingsofpanda -- 熊猫之翼
#Function:
#   Group normalzation
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
slim = tf.contrib.slim


'''
    >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> tools >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    '''
def prime_divide(nb, pg_nb=4):
    '''
        nb: number to be divided
        pg_nb: prefered number of groups. if not possible, then the maximal group number will be returned
        '''
    for i in range(pg_nb):
        if ceil(nb/(pg_nb-i)) == nb/(pg_nb-i):
#            print('group number is {}'.format(pg_nb-i))
            return pg_nb-i
    return 1


'''
    >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  2d setting >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    '''

'''
    GN
    '''

def Group_norm2d(x, G=16, name_scope='', esp=1e-5, reuse=tf.AUTO_REUSE):
    with tf.variable_scope(name_scope, reuse=reuse):

        # normalize
        # tranpose: [bs, h, w, d, c] to [bs, c, d,h, w] following the paper
        x = tf.transpose(x, [0, 3, 1, 2])
        N, C, H, W = x.get_shape().as_list()
        G = min(G, C)
        x = tf.reshape(x, [-1, G, C // G, H, W])
        mean, var = tf.nn.moments(x, [2, 3, 4])
        mean=tf.reshape(mean,[-1, G, 1, 1, 1])
        var=tf.reshape(var,[-1, G, 1, 1, 1])
        x = (x - mean) / tf.sqrt(var + esp)
        # per channel gamma and beta
        gamma = tf.get_variable('gamma', [C],
                                initializer=tf.constant_initializer(1.0))
        beta = tf.get_variable('beta', [C],
                               initializer=tf.constant_initializer(0.0))
        gamma = tf.reshape(gamma, [1, C, 1, 1])
        beta = tf.reshape(beta, [1, C, 1, 1])

        output = tf.reshape(x, [-1, C, H, W]) * gamma + beta
                                # tranpose: [bs, c, d, h, w] to [bs, h, w, d, c] following the paper
        output = tf.transpose(output, [0, 2, 3, 1])
    return output

def GN_auto_channel2d(input, pg_nb=4, name='GN', reuse=tf.AUTO_REUSE):
    tensorshape = input.get_shape().as_list()
#    print('test input shape', tensorshape)
    nb_channel = tensorshape[3]
    group_nb = prime_divide(nb_channel, pg_nb = pg_nb)
    if group_nb == 1:
        return slim.instance_norm(input, reuse=reuse, scope='{}IN/{}'.format(name, group_nb))
    else:
        return Group_norm2d(input, G=group_nb, name_scope=name+'wng/{}'.format(group_nb), reuse=reuse)

'''
    FRN
    '''

def FRN_norm2d(x, nameScope='', reuse=tf.AUTO_REUSE, eps=1e-6, eps_trainable=True):
    with tf.variable_scope(nameScope, reuse=reuse):
        N, H, W, C = x.get_shape().as_list()
        nu2 = tf.reduce_mean(tf.square(x), axis=[1,2], keepdims=True)
        x = x*tf.rsqrt(nu2 + tf.abs(eps))
        if eps_trainable:
            _eps = tf.get_variable('eps', [1], initializer=tf.constant_initializer(eps))
            # print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>we are now on trainable eps! >>>>>>>>>>>>>>>>>>>')
        else:
            _eps = eps
        x = x*tf.rsqrt(nu2 + tf.abs(_eps))

        # three learnable parameters
        gamma = tf.get_variable('gamma', [C], initializer=tf.constant_initializer(1.0))
        beta = tf.get_variable('beta', [C], initializer=tf.constant_initializer(0.0))
        tau = tf.get_variable('tau', [C], initializer=tf.constant_initializer(0.0))

        return tf.maximum(gamma * x + beta, tau)


'''
    >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 3D setting >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    '''

'''
    GN
    '''

def Group_norm3d(x, G=16, name_scope='', reuse=tf.AUTO_REUSE, esp=1e-5):
    with tf.variable_scope(name_scope, reuse=reuse):

        # normalize
        # tranpose: [bs, h, w, d, c] to [bs, c, d,h, w] following the paper
        x = tf.transpose(x, [0, 4, 3, 1,2])
        N, C, D, H, W = x.get_shape().as_list()
        G = min(G, C)
        x = tf.reshape(x, [-1, G, C // G,D, H, W])
        mean, var = tf.nn.moments(x, [2, 3, 4,5])
        mean=tf.reshape(mean,[-1, G, 1,1, 1, 1])
        var=tf.reshape(var,[-1, G, 1,1, 1, 1])
        x = (x - mean) / tf.sqrt(var + esp)
        # per channel gamma and beta
        gamma = tf.get_variable('gamma', [C],
                                initializer=tf.constant_initializer(1.0))
        beta = tf.get_variable('beta', [C],
                               initializer=tf.constant_initializer(0.0))
        gamma = tf.reshape(gamma, [1, C, 1, 1,1])
        beta = tf.reshape(beta, [1, C, 1, 1,1])

        output = tf.reshape(x, [-1, C, D,H, W]) * gamma + beta
                                # tranpose: [bs, c, d, h, w, c] to [bs, h, w, d, c] following the paper
        output = tf.transpose(output, [0, 3, 4, 2, 1])
    return output

def GN_auto_channel3d(input, pg_nb=4, name='GN', reuse=tf.AUTO_REUSE):
    tensorshape = input.get_shape().as_list()
#    print('test input shape', tensorshape)
    nb_channel = tensorshape[4]
    group_nb = prime_divide(nb_channel, pg_nb = pg_nb)
    if group_nb == 1:
        return slim.instance_norm(input, reuse=reuse, scope='{}BN/{}'.format(name, group_nb))
    else:
        return Group_norm3d(input, G=group_nb, name_scope=name+'wng/{}'.format(group_nb), reuse=reuse)



'''
    FRN
    '''

def FRN_norm3d(x, nameScope='', reuse=tf.AUTO_REUSE, eps=1e-6, eps_trainable=True):
    with tf.variable_scope(nameScope, reuse=reuse):
        N, H, W, D, C = x.get_shape().as_list()
        nu2 = tf.reduce_mean(tf.square(x), axis=[1,2,3], keepdims=True)
        if eps_trainable:
            _eps = tf.get_variable('eps', [1], initializer=tf.constant_initializer(eps))
            # print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>we are now on trainable eps! >>>>>>>>>>>>>>>>>>>')
        else:
            _eps = eps
        x = x*tf.rsqrt(nu2 + tf.abs(_eps))

        # three learnable parameters
        gamma = tf.get_variable('gamma', [C], initializer=tf.constant_initializer(1.0))
        beta = tf.get_variable('beta', [C], initializer=tf.constant_initializer(0.0))
        tau = tf.get_variable('tau', [C], initializer=tf.constant_initializer(0.0))

        return tf.maximum(gamma * x + beta, tau)
