

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
from tensorflow.losses import softmax_cross_entropy
import numpy as np
from tensorflow import layers
from tensorflow.python.ops import array_ops
import args
slim = tf.contrib.slim
from sklearn.metrics import confusion_matrix, precision_score, recall_score,f1_score

# collection all possible metrics for tf training

def loss_dict(y_true, y_pred=None, y_pred_wh_softmax=None):
    
    _loss_softmax = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred))
    _loss_CE = crossEntropy_CB(y_true, y_pred_wh_softmax)
    _loss_softmax_label_smoothing = softmax_cross_entropy_label_smoothing(y_true, y_pred, label_smoothing=args.label_smoothing_level)
    _loss_dict = {}
    _loss_dict['soft_max'] =_loss_softmax
    _loss_dict['softmax_label_smoothing'] =_loss_softmax_label_smoothing
    _loss_dict['CE'] = _loss_CE
    _train_loss = \
                args.softmax_label_smoothing_coff*_loss_softmax_label_smoothing\
                + args.softmax_coff*_loss_softmax

    _label = y_true
    _logit = tf.nn.softmax(y_pred)
    _logit = tf.argmax(_logit, axis=-1)
    _loss_dict['label'] = _label
    _loss_dict['logit'] = _logit

    return _train_loss, _loss_dict

def crossEntropy_CB(y_true, y_pred):
    _eps = 1e-6
    y_true = tf.one_hot(y_true, args.nb_classes)
    y_pred = tf.clip_by_value(y_pred, _eps, 1.0-_eps)
    L = -(y_true*tf.log(y_pred) + (1-y_true)*tf.log(1-y_pred))
    return tf.reduce_mean(L*y_true)

def softmax_cross_entropy_label_smoothing(y_true, y_pred, label_smoothing=0):
    _eps = 1e-6
    onehot_labels = tf.one_hot(y_true, args.nb_classes)
    logits = tf.nn.softmax(y_pred)

    return softmax_cross_entropy(onehot_labels, logits, label_smoothing=label_smoothing)        


def dice_loss_dict(y_true, y_pred):
    smooth = 1e-6
    dict={}
    for i in range(args.num_classes):
        y_true_f_1 = layers.flatten(y_true[...,i])
        y_pred_f_1 = layers.flatten(y_pred[...,i])
        intersection = tf.reduce_sum(y_true_f_1 * y_pred_f_1)
        if i==0:
            key_name = 'dice loss on background: '
        else:
            key_name = 'dice loss on class {}: '.format(i)
        dict[key_name] = 1- (2. * intersection + smooth) / (tf.reduce_sum(y_true_f_1 * y_true_f_1) + tf.reduce_sum(y_pred_f_1 * y_pred_f_1) + smooth)
    return dict


def soft_CL(x, ths=10):

    def maxPool3d(_tensor):
        return  slim.max_pool3d(_tensor, kernel_size=[3, 3, 3], stride=[1, 1, 1], padding='SAME', scope='cl_max3d')
    def minPool3d(_tensor):
        n_tensor = -1*_tensor
        return -1*maxPool3d(n_tensor)

    for i in range(ths):
        min_pool_x = minPool3d(x)
        contour = tf.nn.relu(maxPool3d(min_pool_x)-min_pool_x)
        x = tf.nn.relu(x - contour)

    return x

def cl_dice_loss(y_true, y_pred, true_cl=None, eps = 1e-6):

    y_true = tf.expand_dims(y_true, axis=-1)
    y_pred = tf.expand_dims(y_pred, axis=-1)

    def norm_intersection_rate(cl, vessel, eps=1e-6):
        cl_flat = layers.flatten(cl)
        vessel_flat = layers.flatten(vessel)
        return (tf.reduce_sum(cl_flat*vessel_flat) + eps) / (tf.reduce_sum(cl_flat) + eps)

    cl_p = soft_CL(y_pred)
    if true_cl is None:
        cl_t = soft_CL(y_true)
    else:
        cl_t = true_cl

    p_flat = norm_intersection_rate(cl_p, y_true)
    t_flat = norm_intersection_rate(cl_t, y_pred)

    return 1 - 2. * (p_flat * t_flat) / (p_flat + t_flat + eps)

def cl_dice_loss_dict(y_true, y_pred, true_cl=None, eps = 1e-6):
    dict = {}
    for i in range(args.num_classes):
        if i == 0: continue
        key_name = 'cl dice loss on class {}: '.format(i)
        dict[key_name] = cl_dice_loss(y_true[...,i], y_pred[...,i], eps=eps)

    return dict





def softmax_loss(y_true, y_pred):
    loss_softmax = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_true, logits=y_pred)
    loss_softmax = tf.reduce_mean(loss_softmax)
    return loss_softmax


# focal loss.
# when \alpha>0.5, it could balance y part


def focal_loss(y_true, y_pred, alpha=0.8, gamma=2):
    eps = 1e-4
    # y_pred = tf.clip_by_value(y_pred, eps, 1.-eps)
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))

    pt_1 = tf.clip_by_value(pt_1, eps, 1.-eps)
    pt_0 = tf.clip_by_value(pt_0, eps, 1.-eps)

    return -tf.reduce_sum(alpha * tf.pow(1. - pt_1, gamma) * tf.log(pt_1))-tf.reduce_sum((1-alpha) * tf.pow(pt_0, gamma) * tf.log(1. - pt_0))
#
#
## focal loss + dice loss.
## note that the value of focal loss could be ~1000, while dice loss usually less than 1. Hence, to balance them, we use $-log$ to zoom out dice loss and $\alpha$ to zoom in focal loss
#
def mixedLoss(y_true,y_pred):
    dice_loss = dice_score(y_true,y_pred)
    focal_part = 0.75 *0.01* focal_loss(y_true,y_pred)
    log_dice_loss = - 1.*tf.log(dice_loss)
    return 0.1*focal_part, 100.*log_dice_loss, dice_loss


def accuracy(y_true, y_pred):
    correct_prediction = tf.equal(tf.argmax(y_pred, axis=4), tf.argmax(y_true, axis=4))
    correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy_num = tf.reduce_mean(correct_prediction)
    return accuracy_num


def weighted_accuracy(y_true, y_pred):

    correct_prediction = layers.flatten(y_pred)
    #    correct_prediction = tf.reduce_min(y_pred, axis=-1)
    correct_prediction = tf.greater(correct_prediction,0.5)

    #    correct_prediction = tf.equal(tf.argmax(y_pred, axis=-1), y_ture_tmp)
    correct_prediction = tf.cast(correct_prediction, tf.float32)
    # correct_prediction.shape = (1, 256, 256, 16)

    y_true = layers.flatten(y_true)
    #    y_true = tf.reduce_min(y_true, axis=-1)
    correct_partial = tf.multiply(correct_prediction, y_true)
    accuracy = tf.reduce_sum(correct_partial)/tf.reduce_sum(y_true)

    incorrect_partial = tf.multiply((1.-correct_prediction),(1.- y_true))
    inaccuracy = tf.reduce_sum(incorrect_partial)/tf.reduce_sum(1.-y_true)
    return accuracy, inaccuracy

'''
    # numpy version of metrics for fast evaluation
    '''

def binary_dice_np(y_true, y_pred):
    smooth = 1e-6
    y_true_f_1 = np.ndarray.flatten(y_true)
    y_pred_f_1 = np.ndarray.flatten(y_pred)
    intersection = np.sum(y_true_f_1 * y_pred_f_1)

    return 1- (2. * intersection + smooth) / (np.sum(y_true_f_1 * y_true_f_1) + np.sum(y_pred_f_1 * y_pred_f_1) + smooth)

def plot_Fscore_table(label_true, pred, num_class):
    precise  = precision_score( label_true, pred, range(num_class), average=None)
    recall   = recall_score(  label_true, pred, range(num_class), average=None)
    f1=f1_score( label_true, pred,  range(num_class), average=None)
    confusion=confusion_matrix(label_true, pred, range(num_class))
    # print("Precise is : ",precise)
    # print("Recall is : ",recall)
    # print('F1-Score is: ',f1)
    # print("Confusion Matrix is :")
    # print(confusion)
    Counts=np.sum(confusion,1)

    precise_=['{:.3f}'.format(i) for i in precise]
    recall_=['{:.3f}'.format(i) for i in recall]
    f1_=['{:.3f}'.format(i) for i in f1]

    return precise,recall,f1,confusion