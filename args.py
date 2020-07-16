# coding: utf-8
# This file contains the parameter used in xxx_protocol.py

from __future__ import division, print_function

#from utils.misc_utils import parse_anchors, read_class_names
import math
import os
import pickle
import numpy as np
from helper.config_status_writer import restarting_position
import json
import random

train_file_folder = '/home/liupan/Pneumonia_v3/PneumoniaCls/dataPreprocessing/DocCleanedSevenCls/tf_dir_train' # The path of the training tfrecords file.
valid_file_folder = '/home/liupan/Pneumonia_v3/PneumoniaCls/dataPreprocessing/DocCleanedSevenCls/tf_dir_valid'  # The path of the validation tfrecords file.


def json_loader(json_dir):
    with open (json_dir, 'rb') as fp:
        json_file = json.load(fp)
    return json_file

def folder_json_list_combin(folder_dir):
    fs = os.listdir(folder_dir)
    files_lists = [os.path.join(folder_dir, f) for f in fs if '.json' in f]
    list = []
    for dir in files_lists:
        list = list + json_loader(dir)

    return list

saving_name = 'DocCleanedSevenClsMultiLabel'

if os.path.isdir(train_file_folder) and os.path.isdir(valid_file_folder):

    train_file_list = folder_json_list_combin(train_file_folder)
    valid_file_list = folder_json_list_combin(valid_file_folder)

    train_set_len = len(train_file_list)
    valid_set_len = len(valid_file_list)

    initial_epoch, valid_ckpt = restarting_position()


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

def file_lister(folder_dir, key=None):
    fs = os.listdir(folder_dir)
    file_lists = [os.path.join(folder_dir, f) for f in fs if key in f]
    return file_lists

tfrecords_folder = '/data/liupan/pneuCls/DocCleanRawImgTV2tfCrop448D64W224H224'

if os.path.isdir(tfrecords_folder):
    train_folders = file_lister(tfrecords_folder, key='training')
    valid_folders = file_lister(tfrecords_folder, key='valid')

    train_tfrecords = []; valid_tfrecords = []

    for folder in train_folders:
        train_tfrecords = train_tfrecords + file_lister(folder, key='.tfrecords')

    random.shuffle(train_tfrecords)

    for folder in valid_folders:
        valid_tfrecords = valid_tfrecords + file_lister(folder, key='.tfrecords')

ckpt_path = './' + saving_name



## some flags >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

restore_from_best_valid_ckpt = False # if false, then restore from the lastest ckpt


if restore_from_best_valid_ckpt: best_ckpt_path = '/home/liupan/Pneumonia_v3/Pneumonia_classification/multiCls/crop432d64wh224/0515_denseNet264G/fresh_start_0515/epoch_23_valid_0.8727598566308243'



### tfrecords dimensions >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

in_shape = [384, 384, 48, 1]

### pursing some configs

train_batch_size = 2
valid_batch_size = 1
buffer_size = train_batch_size*64
lr = 1E-4

### turn on if use data argumentation
crop_size_x = 0.0
crop_size_y = 0.0
rotation_angle = 0.0

normalizer = 'group'


is_classification = True
growth_rate= 48 # denseNet growth rate
nb_classes = 2

softmax_coff = 1.
softmax_label_smoothing_coff = 0.0
label_smoothing_level = 0.1

epochs = 666
