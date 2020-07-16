
#Author:
#    WingsOfPanda
#Function:
#    Auto dicoms/nii locating by folders
#Date:
#    Aug. 30th, 2019
#Version:
#    1.1.0
#Instruction:
#

import SimpleITK as sitk
import os,sys
import numpy as np
from helper.auto_faltten import is_last_folder
from helper.auto_delete import delete_viewer_related
import scipy.ndimage as scimg
from numpy import transpose
import tensorflow as tf
INPUT_HEIGHT, INPUT_WIDTH = 320,320
INPUT_DEPTH = 24

from queue import Queue


def find_last_folders(root):
    '''
    Find all of the last folders
    using a queue，广度优先遍历，get a set including all of the last folders
    '''
    assert os.path.isdir(root)
    last_folders = set()

    q = Queue(100000)
    if is_last_folder(root):
        last_folders.add(root)
        return last_folders
    else:
        q.put(root)

    while not q.empty():
        cur = q.get()
        if is_last_folder(cur):
            last_folders.add(cur)
        else:
            fs = os.listdir(cur)
            for f in fs:
                sub = os.path.join(cur,f)
                if os.path.isdir(sub):
                    q.put(sub)
                    # print(sub)
    return last_folders

def is_desired_folder(last_folder,file_type):

    '''
        args:
        last_folder: folders that contains non-folder files only
        file_type: str, that could be .dcm or .nii.gz

        return: boolen value that a folder contains a desired file type or not

    '''


    flag = True
    fs = os.listdir(last_folder)
    fs = [f for f in fs if not f.startswith('.')]
    fsp = [os.path.join(last_folder,f) for f in fs]
    ext = ''

    if file_type == '.dcm':

        for f in fs:
            if f.endswith('.dcm'):
                ext = '.dcm'
                break
        if ext == '.dcm':
            fsp = [f for f in fsp if f.endswith('.dcm')]
        else:
            fsp = [f for f in fsp if os.path.splitext(f)[-1]=='']

        if len(fsp)<2: flag = False

#        print('flag value: ', flag)

    if file_type == '.nii.gz':

        for f in fs:
            if f.endswith('.nii.gz'):
                ext = '.nii.gz'
                break
        if ext == '.nii.gz':
            fsp = [f for f in fsp if f.endswith('.nii.gz')]
        else:
            fsp = [f for f in fsp if os.path.splitext(f)[-1]=='']

        if len(fsp)<1: flag = False

#        print('flag value: ', flag)

    return flag


def folder_listing(root, file_type):

    delete_viewer_related(root)

    if os.path.isfile(root):
        print('root is a file, not folder.')
        sys.exit()
    folders = find_last_folders(root)
#    print(folders)

    if root in folders:
        print('root is a dicom folder, convert to a nii.')
        last_folder2nii(root,root,out_dir)
        sys.exit()

    folder_dir_list = []
    for fod in folders:
        if is_desired_folder(fod,file_type): folder_dir_list.append(fod)
    return folder_dir_list



if __name__ == '__main__':
    # make sure chinese is well supported
    import io
    import sys
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')
    root = '/Volumes/Elements/张杰_20190902'
    folder_list = folder_listing(root, '.dcm')
    print(folder_list)
