'''
Author:
    wingsofpanda
Function:
    tfrecords decoder
    data loading protocol
Date:
    May 30th, 2019
Version:
    1.0.0: initial release
    1.1.0: data argumentation added

'''
#from assests.loss import binary_accuracy
import tensorflow as tf
import SimpleITK as sitk
import numpy as np
import os
import cv2
#import args
from math import ceil
import args

in_shape = args.in_shape


def to_binary(label):
    one_tmp = np.ones([in_shape[0], in_shape[1], in_shape[2]], dtype=np.float32)
    label = tf.where(tf.equal(label, 0), label, one_tmp)
    return label

def decode_n_reshape(tfrecord, type):

    if type == 'img':
        tfrecord = tf.decode_raw(tfrecord, tf.float32)
        tfrecord = tf.reshape(tfrecord, [in_shape[0], in_shape[1], in_shape[2]])

    if type == 'label':
        tfrecord = tf.one_hot(tfrecord, 2) # here goes the one_hot command
        tfrecord = tf.cast(tfrecord, tf.float32)
    return tfrecord

def data_argumentation(tfrecord, type, fraction_x,fraction_y, angle):

    offset_H = ceil(in_shape[0]*fraction_x)
    target_H = in_shape[0] - 2*offset_H

    offset_W = ceil(in_shape[1]*fraction_y)
    target_W = in_shape[1] - 2*offset_W
    #    in_shape = [320,320,24,1]
    if type =='img':
#        tfrecord = tf.image.flip_left_right(tfrecord)
        tfrecord = tf.contrib.image.rotate(tfrecord,angle,interpolation='BILINEAR')
        tfrecord = tf.image.crop_to_bounding_box(tfrecord, offset_H, offset_W, target_H,target_W)
        tfrecord = tf.image.resize_images(tfrecord,[in_shape[0], in_shape[1]], method=tf.image.ResizeMethod.BILINEAR)


    if type == 'mask':
        tfrecord = tf.contrib.image.rotate(tfrecord,angle,interpolation='NEAREST')
        tfrecord = tf.image.crop_to_bounding_box(tfrecord, offset_H, offset_W, target_H,target_W)
        tfrecord = tf.image.resize_images(tfrecord,[in_shape[0], in_shape[1]], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return tfrecord

def dimension_expansion(tfrecord):
    tfrecord = tf.expand_dims(tfrecord, axis=-1)
    return tfrecord


def parse_exmp_train(serial_exmp):

    feats = tf.parse_single_example(serial_exmp, features={'img': tf.FixedLenFeature([], tf.string), 'label_mask': tf.FixedLenFeature([], tf.string)})

    feats['img'] = decode_n_reshape(feats['img'], 'img')
    feats['label_mask'] = decode_n_reshape(feats['label_mask'], 'img')
    feats['label'] = tf.reduce_max(feats['label_mask'])

    feats['label'] = tf.cast(feats['label'], tf.int32)

    feats['img'] = dimension_expansion(feats['img'])

    return feats

def parse_exmp_valid(serial_exmp):

    feats = tf.parse_single_example(serial_exmp, features={'img': tf.FixedLenFeature([], tf.string), 'label_mask': tf.FixedLenFeature([], tf.string)})

    feats['img'] = decode_n_reshape(feats['img'], 'img')
    feats['label_mask'] = decode_n_reshape(feats['label_mask'], 'img')
    feats['label'] = tf.reduce_max(feats['label_mask'])
    feats['label'] = tf.cast(feats['label'], tf.int32)

    feats['img'] = dimension_expansion(feats['img'])
    return feats

def get_batch_data_tfdata(file_base_path, batch_size, buffer_size=10000, epochs=10, is_train_model = True):
    dataset = tf.data.TFRecordDataset(file_base_path)
    if is_train_model:
        dataset = dataset.map(parse_exmp_train)
        epoch_dataset = dataset.repeat(epochs)
        epoch_dataset = epoch_dataset.shuffle(buffer_size=buffer_size)
    else:
        dataset = dataset.map(parse_exmp_valid)
        epoch_dataset = dataset.repeat(epochs)

    batch_dataset = epoch_dataset.batch(batch_size=batch_size)

    return batch_dataset
