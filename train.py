'''
Author:
    wingsofpanda -- 熊猫之翼
Function:
   training function for covid-19 project
Date:
    Jan. 17th, 2020
Version:
    1.0.0
'''

import tensorflow as tf
from assests.model_zoo import dense_net_264 as core_model
from assests.tf_data_protocol import get_batch_data_tfdata
import args
from assests.metrics import loss_dict
import os
import numpy as np
import pickle
from math import ceil, floor
from tqdm import tqdm
import sys
import time
import sklearn.metrics as skms
from sklearn.metrics import confusion_matrix, precision_score, recall_score,f1_score
from helper.config_status_writer import *
from allReduceProtocal.allreduce import *
from assests.metrics import *


def training_module(input_fn):
    inputs = input_fn()
    image = inputs['img']
    logits, logits_softmax = core_model(image)
    label = inputs['label']
    _train_loss, _loss_dict = loss_dict(label, y_pred=logits, y_pred_wh_softmax = logits_softmax)

    return _train_loss, _loss_dict

def do_training(train_pack, valid_pack):
    train_set_len = args.train_set_len
    valid_set_len = args.valid_set_len
    gpus_num = len(get_available_gpus())
    saver = tf.train.Saver(max_to_keep=512, var_list=tf.global_variables())
    valid_ckpt = args.valid_ckpt
    accuracy_ckpt = 0
    epoch_initial = args.initial_epoch
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        sess.run(tf.global_variables_initializer())

        if args.restore_from_best_valid_ckpt:
            saver.restore(sess, args.best_ckpt_path)
            print('>>>>>>>>>>>>>>>>>>>>>>>>> best valid ckpt restored >>>>>>>>>>>>>>>>>>>>>>>>')
        elif os.path.isdir(args.ckpt_path):
            saver.restore(sess, tf.train.latest_checkpoint(args.ckpt_path))
            print('>>>>>>>>>>>>>>>>>>>>>>>>>> last ckpt restored >>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        try:
            step = 0
            while True:
                epoch_stats = epoch_initial + ceil(gpus_num*step*args.train_batch_size/train_set_len)
                progress = gpus_num*step*args.train_batch_size/train_set_len - floor(gpus_num*step*args.train_batch_size/train_set_len)

                ts = time.time()
                sess.run(train_pack[0])
                te = time.time()
                # print('time used: {}'.format(te-ts))
                if step>1 and step % 4 == 0:
                    _, _loss_train = sess.run((train_pack[0], train_pack[1]))

                    run_per_sec = (args.train_batch_size*gpus_num)/(te-ts)
                    time_to_finish = train_set_len*(1-progress)/run_per_sec
                    stats_printing = 'Epoch {}, progress in {} %, {} mins to go'.format(epoch_stats, round(progress*10000)/100, round(time_to_finish/60*10)/10)
                    train_printing = 'Minibatch tower loss= {}, Examples/sec {}'.format( _loss_train, run_per_sec)
                    print(stats_printing)
                    print(train_printing)
                    print('\n')
                    sys.stdout.flush()

                if step > 0 and step % 10 == 0:

                    _, _loss_train_dict = sess.run((train_pack[0], train_pack[2]))
                    train_all_inf_print = dict_print(_loss_train_dict)
                    print('printing infor again')
                    print(train_all_inf_print)
                    print('\n')
                    sys.stdout.flush()

                if step > 0 and step % 200 == 0:

                    saver.save(sess, os.path.join(args.ckpt_path,'check_point'))
                    print('check point saved')
                    print('\n')
                    sys.stdout.flush()

                if step>1 and step % ceil(train_set_len/gpus_num/args.train_batch_size) ==0:

                    _valid_loss = []
                    _valid_stats = {}
                    _label_list =[]
                    _logit_list =[]
                    for _valid_step in tqdm(range(ceil(valid_set_len/gpus_num/args.valid_batch_size))):
                        _valid_loss_tmp, _valid_dict_tmp = sess.run((valid_pack[0], valid_pack[1]))
                        _valid_loss.append(_valid_loss_tmp)
                        _valid_stats = dict_append(_valid_stats, _valid_dict_tmp)

                    _valid_all_average = np.mean(_valid_loss)
                    valid_printing = 'Epoch {} step {} with current validation loss= {}'.format(epoch_stats, step, _valid_all_average)
                    print(valid_printing)

                    _crt = 0

                    for _label, _logit in zip(_valid_stats['label'], _valid_stats['logit']):
                        sys.stdout.flush()
                        _label_list += _label
                        _logit_list += _logit
                        for _label_inner, _logit_inner in zip(_label, _logit):
                            if _label_inner == _logit_inner:
                                _crt += 1

                    _total = len(_logit_list)

                    _valid_accuracy = _crt/_total
                    _valid_precision, _valid_recall, _valid_f1_score, confusion = plot_Fscore_table(_label_list, _logit_list, args.nb_classes)

                    _valid_stat_print = 'infor: \n accuracy={} \n precision={} \n recall={}\n f1 score={}'.format( _valid_accuracy, _valid_precision, _valid_recall, _valid_f1_score)
                    _valid_confusion_print = 'confusion matrix: \n {}'.format(confusion)
                    print(_valid_stat_print)
                    print(_valid_confusion_print)

                    sys.stdout.flush()

                    record_name = 'valid_record_'+args.saving_name+'.txt'
                    infor_writer(record_name, [valid_printing])
                    stats_name = 'valid_stats_'+args.saving_name+'.txt'
                    infor_writer(stats_name, ['\n', _valid_stat_print, _valid_confusion_print])



                    if accuracy_ckpt < _valid_accuracy:
                        accuracy_ckpt = _valid_accuracy
                    saver.save(sess, os.path.join(args.ckpt_path, 'epoch_{}_valid_{}'.format(epoch_stats, _valid_accuracy)))
                    print('model saved at validation loss {} at step {}, epoch {}'.format(_valid_accuracy, step, epoch_stats))
                    print('\n')
                    sys.stdout.flush()
                step += 1
        except tf.errors.OutOfRangeError:
            # we're through the dataset
            pass
    print('Final validation loss: {}'.format(_valid_all_average))


def parallel_training(model_fn, dataset_train, dataset_val):
    iterator_train = dataset_train.make_one_shot_iterator()
    iterator_valid = dataset_val.make_one_shot_iterator()
    def input_fn_train():
        with tf.device(None):
            # remove any device specifications for the input data
            return iterator_train.get_next()

    def input_fn_valid():
        with tf.device(None):
        # remove any device specifications for the input data
            return iterator_valid.get_next()

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        optimizer = tf.train.AdamOptimizer(learning_rate=args.lr)
        train_pack = create_parallel_optimization(model_fn, input_fn_train, optimizer=optimizer, is_training=True)
        valid_pack = create_parallel_optimization(model_fn, input_fn_valid, is_training=False)

    # loss_val = core_net_val(lambda: iterator_val.get_next())
    do_training(train_pack, valid_pack)


if __name__ == '__main__':
# make sure chinese is well supported
    import io
    import sys
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')

    # config_writer()
    tf.reset_default_graph()
    parallel_training(
                    training_module,
                    get_batch_data_tfdata(args.train_tfrecords, args.train_batch_size, buffer_size=args.buffer_size, epochs=args.epochs, is_train_model = True),
                    get_batch_data_tfdata(args.valid_tfrecords, args.valid_batch_size, buffer_size=args.buffer_size, epochs=args.epochs, is_train_model = False)
                    )
