'''
Author:
    WingsOfPanda
Function:
    Auto .nii.gz locating by folders
    Auto filter out AxT1, AxT2, AxT1c dicoms
    Convert to tfrecords by shards
    return files dir that does not have desired dicoms series
Date:
    Aug. 30th, 2019
Version:
    Alpha 1.1.0
Instruction:

'''
import SimpleITK as sitk
import os,sys
import numpy as np
import scipy.ndimage as scimg
from numpy import transpose
import tensorflow as tf
from math import ceil
from tqdm import tqdm
import shutil
import time
import args

from queue import Queue

def dict_append(dict_collection, dict_to_append):
    for key, v in dict_to_append.items():
        if type(v) is dict:
            try:
                for k_l2, v_l2 in v.items():
                    try:
                        dict_collection[key][k_l2].append(v_l2)
                    except KeyError:
                        dict_collection[key][k_l2]= [v_l2]
            except KeyError:
                dict_collection[key] = {}
                for k_l2, v_l2 in v.items():
                        dict_collection[key][k_l2]= [v_l2]
        else:
            try:
                dict_collection[key].append(v)
            except KeyError:
                dict_collection[key] = [v]

    return dict_collection

def dict_print(dict_loss):
    printer = ''
    for k, v in dict_loss.items():
        if type(v) is dict:
            printer = printer + str(k)+': '
            for k1, v1 in v.items():
                printer = printer + str(k1) + ': '+ str(np.mean(v1))+'; '
            printer = printer + '\n'
        else:
            printer = printer + str(k) + ': ' + str(np.mean(v)) + '; ' + '\n'

    printer = printer + '\n'

    return printer

def valid_check():
    text_name = 'valid_record_'+args.saving_name+'.txt'
    fp = open(text_name, 'r')
    lines = fp.readlines()
    fp.close()
    valid_ckpt_list = []
    for line in lines:
        tmp_file = line.split('= ')
        tmp_file = tmp_file[-1].split(';')[0]
        valid_ckpt_list.append(float(tmp_file))

    return len(lines)-1, np.amin(valid_ckpt_list)

def restarting_position():

    text_name = 'valid_record_'+args.saving_name+'.txt'
    if os.path.isfile(text_name):
        initial_epoch, valid_ckpt = valid_check()

    else:
        initial_epoch = 0
        valid_ckpt = 10000000000

    return initial_epoch, valid_ckpt

def config_writer():
    tr = open('train_config_recorder.txt', 'a')
    initial_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    initial_epoch, valid_ckpt = restarting_position()

    tr.write('This training was commenced at {}, epoch {}, and validation value {}'.format(initial_time, initial_epoch, valid_ckpt))
    tr.write('\n\n')

    tr.write('The training acquired training data from \n {} \n and valid data from \n {} \n\n'.format(args.train_file_list, args.valid_file_list))

    tr.write('We totally have {} training points and {} valid points'.format(args.train_set_len, args.valid_set_len))
    tr.write('\n\n')

    tr.write('The input size is {} {} {}'.format(args.in_shape[0], args.in_shape[1], args.in_shape[2]))
    tr.write('\n\n')

    tr.write('The training commenced on focal with weight {}, dice with weight {}, and softmax with weight {}'.format(args.focal_loss_pmt, args.dice_loss_pmt, args.softmax_loss_pmt))
    tr.write('\n\n')

    tr.write('The learning rate is {}'.format(args.lr))
    tr.write('\n\n----------------------------------------------------\n\n')

    tr.close()

def infor_writer(writer_name, infor_list):
    writer = open(writer_name, 'a')
    for infor in infor_list:
        writer.write(infor)
        writer.write('\n')

    writer.close()



if __name__ == '__main__':
    # make sure chinese is well supported
    import io
    import sys
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')
#    main()
