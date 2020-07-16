'''
Author:
    wingsofpanda -- 熊猫之翼
Function:
   modular design for multi-GPU parallel training. 
Date:
    June 17th, 2017, CMU
Version:
    1.0.0
'''

import tensorflow as tf

import args
import os
import numpy as np
import pickle
from math import ceil
from tqdm import tqdm
import sys
import time
from helper.config_status_writer import dict_append



PS_OPS = [
          'Variable', 'VariableV2', 'AutoReloadVariable', 'MutableHashTable',
          'MutableHashTableOfTensors', 'MutableDenseHashTable'
          ]

def get_available_gpus():
    """
        Returns a list of the identifiers of all visible GPUs.
        """
    from tensorflow.python.client import device_lib
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

def assign_to_device(device, ps_device):
#    Returns a function to place variables on the ps_device.
#        Args:
#        device: Device for everything but variables
#        ps_device: Device to put the variables on. Example values are /GPU:0 and /CPU:0.
#        If ps_device is not set then the variables will be placed on the default device.
#        The best device for shared varibles depends on the platform as well as the model. Start with CPU:0 and then test GPU:0 to see if there is an improvement.
#
    def _assign(op):
        node_def = op if isinstance(op, tf.NodeDef) else op.node_def
        if node_def.op in PS_OPS:
            return ps_device
        else:
            return device
    return _assign


def create_parallel_optimization(model_fn, input_fn, optimizer=None, controller="/cpu:0", is_training = True):
    # This function is defined below; it returns a list of device ids like
    # `['/gpu:0', '/gpu:1']`
    devices = get_available_gpus()

    # This list keeps track of the gradients per tower and the losses
    tower_grads = []
    losses = []
    losses_dict = {}

    # Get the current variable scope so we can reuse all variables we need once we get
    # to the second iteration of the loop below
    with tf.variable_scope(tf.get_variable_scope()) as outer_scope:
        for i, id in enumerate(devices):
            name = 'tower_{}'.format(i)
            # Use the assign_to_device function to ensure that variables are created on the
            # controller.
            with tf.device(assign_to_device(id, controller)), tf.name_scope(name):

                # Compute loss and gradients, but don't apply them yet
                loss, loss_dict = model_fn(input_fn)

                if is_training:

                    with tf.name_scope("compute_gradients"):
                        # `compute_gradients` returns a list of (gradient, variable) pairs
                        grads = optimizer.compute_gradients(loss)
                        tower_grads.append(grads)

                losses.append(loss)
                losses_dict = dict_append(losses_dict, loss_dict)

            # After the first iteration, we want to reuse the variables.
            outer_scope.reuse_variables()

# Apply the gradients on the controlling device2
    if is_training:
        with tf.name_scope("apply_gradients"), tf.device(controller):
            # Note that what we are doing here mathematically is equivalent to returning the average loss over the towers and compute the gradients relative to that.
            # Unfortunately, this would place all gradient-computations on one device, which is why we had to compute the gradients above per tower and need to average them here.
            # This function is defined below; it takes the list of (gradient, variable) lists and turns it into a single (gradient, variables) list.
            gradients = average_gradients(tower_grads)
            global_step = tf.train.get_or_create_global_step()
            apply_gradient_op = optimizer.apply_gradients(gradients, global_step)
            avg_loss = tf.reduce_mean(losses)

        return [apply_gradient_op, avg_loss, losses_dict]

    else:
        avg_loss = tf.reduce_mean(losses)
        return [avg_loss, losses_dict]



def average_gradients(tower_grads):
#    Calculate the average gradient for each shared variable across all towers.
#        Note that this function provides a synchronization point across all towers.
#        Args:
#           tower_grads: List of lists of (gradient, variable) tuples. The outer list ranges over the devices. The inner list ranges over the different variables.
#        Returns:
#           List of pairs of (gradient, variable) where the gradient has been averaged across all towers.
#
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = [g for g, _ in grad_and_vars]
        grad = tf.reduce_mean(grads, 0)
        # The Variables are redundant because they are shared across towers. So we will just return the first tower's pointer to the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads



if __name__ == '__main__':
# make sure chinese is well supported
    import io
    import sys
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')
