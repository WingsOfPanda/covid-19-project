'''
Author:
    IFSO (Mr.F)
Function:
    Find all 2nd_last_folders, output in .txt file, faltten sub folders to
    2nd_last_folders together;
    Rename the .dcm files(if needed) in the 2nd_last_folders, make sure
    total of .dcm files before and post of rename & faltten is same.
Date:
    July 18th, 2019
Version:
    Alpha 1.0.1
'''

import os
def is_last_folder(folder):
    '''
    Check whether is a folder without sub folder(s)
    '''
    assert os.path.isdir(folder)
    fs = os.listdir(folder)
    for f in fs:
        # print(os.path.join(folder,f))
        if os.path.isdir(os.path.join(folder,f)):
            return False
    return True

def is_last2nd_folder(folder):
    '''
    Check whether is a folder with just one layer of sub folder(s)
    '''
    assert os.path.isdir(folder)
    has_subfolder = False
    sub_is_last_folder = True
    fs = os.listdir(folder)
    for f in fs:
        sub = os.path.join(folder,f)
        # print(sub)
        if os.path.isdir(sub):
            has_subfolder = True
            if is_last_folder(sub)==False:
                sub_is_last_folder = False
                return False
    if has_subfolder==True and sub_is_last_folder==True:
        return True
    else:
        return False

from queue import Queue
def find_last2nd_folder(folder):
    '''
    Find all of the 2nd last folders
    using a queue，广度优先遍历，get a set including all of 2nd last folders
    '''
    assert os.path.isdir(folder)
    last2nd_folders = set()

    q = Queue(100000)
    if not is_last_folder(folder):
        q.put(folder)
    else:
        return None

    while not q.empty():
        cur = q.get()
        if is_last2nd_folder(cur):
            last2nd_folders.add(cur)
        else:
            fs = os.listdir(cur)
            for f in fs:
                sub = os.path.join(cur,f)
                if os.path.isdir(sub):
                    q.put(sub)
                    # print(sub)
    return last2nd_folders

def folders2txt(root_path, file_name):
    folders_set = find_last2nd_folder(root_path)
    folders = list(folders_set)
    folders.sort()
    file = open(os.path.join(root_path, file_name),'a',encoding='UTF-8')
    for f in folders:
        file.write(f+'\n')
    file.close()
    return folders

def write_file(root_path,file_name,content):
    file = open(os.path.join(root_path, file_name),'a',encoding='UTF-8')
    file.write(content+'\n')
    file.close()

def dcm_rename(last_folder):
    '''
    文件夹内.dcm文件重命名
    格式:foldername_000.dcm
    '''
    folder_name = os.path.split(last_folder)[-1]
    fs = os.listdir(last_folder)
    i = 0
    for f in fs:
        extension = os.path.splitext(f)[-1]
        file_name = (folder_name + '_{:03d}' + extension).format(i)
        os.rename(os.path.join(last_folder,f), os.path.join(last_folder,file_name))
        # print(file_name)
        i += 1

def is_rename(last2nd_folder):
    '''
    功能：判断是否需要对子文件夹内的.dcm文件进行重命名
    原因在于某些数据中，不同序列的dcm文件可能出现重名的现象，直接展开会覆盖掉
    比如不同序列的文件夹（比如101，201）均含有1_0.dcm, 2_1.dcm
    输入：请确保输入为倒数第二层文件夹
    '''
    fs = os.listdir(last2nd_folder)
    flag1 = False
    flag2 = False
    # 条件1：任意一个子文件夹名称长度<8，并且文件夹名称中不包括View（比如文件夹名为301，1之类的名称）
    for f in fs:
        if os.path.isdir(os.path.join(last2nd_folder,f)) and len(f) < 8 and f.find('View')<0:
            flag1 = True
    if flag1 == True:
        # 条件1成立的情况下，任意子文件夹内的dcm文件名称长度（包括.dcm）<12时，条件2成立
        # 条件1和条件2是根据实际情况总结出来的，如出现其他情况，可修改或增加新规则
        for f in fs:
            if os.path.isdir(os.path.join(last2nd_folder,f)):
                last_folder = os.path.join(last2nd_folder,f)
                lfs = os.listdir(last_folder)
                for lf in lfs:
                    if lf.find('.dcm')>=0 and len(lf) < 12:
                        flag2 = True
    return flag2

import shutil
import sys
def flatten_folder(last2nd_folder):
    '''
    文件夹展开
    输入：请确保输入为倒数第二层文件夹
    '''
    fs = os.listdir(last2nd_folder)
    for f in fs:
        last_folder = os.path.join(last2nd_folder,f)
        if os.path.isdir(last_folder):
            lfs = os.listdir(last_folder)
            for lf in lfs:
                file_path = os.path.join(last_folder,lf)
                if os.path.isfile(file_path):
                    shutil.move(file_path, os.path.join(last2nd_folder,lf))
                else:
                    print('Please make sure input is a 2nd last folder.')
                    print('an error happened')
                    sys.exit()
            os.rmdir(last_folder)

def count_dcm(folder):
    '''
    统计文件夹内.dcm文件的数量
    '''
    number = 0
    for root, dirs, files in os.walk(folder):
        for filename in files:
            filepath= os.path.join(root,filename)
            if filepath.find('.dcm') > 0:
                number += 1
    return number

def main():
    root_path = '/Users/panliu/BioMind/TestingData/surgery'
    # root_path = '/Users/biomind/Desktop/多发性硬化（2次查找）'
    file_name = 'last2nd_folders.txt'
    error_file_name = 'error.txt'
    folders = folders2txt(root_path, file_name)
    print(len(folders))
    # i = 0
    for fol in folders:
        pre_num = count_dcm(fol)
        if is_rename(fol):
            # i += 1
            # print(i,fol)
            fs = os.listdir(fol)
            for f in fs:
                last_folder = os.path.join(fol,f)
                if os.path.isdir(last_folder):
                    dcm_rename(last_folder)
        flatten_folder(fol)
        post_num = count_dcm(fol)
        if pre_num != post_num:
            print(fol)
            content = '{}: before: {}, after: {}'.format(fol,pre_num,post_num)
            write_file(root_path,error_file_name,content)

if __name__ == '__main__':
    # make sure chinese is well supported
    import io
    import sys
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')
    main()
