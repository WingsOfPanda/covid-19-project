'''
Author:
    IFSO (Mr.F)
Function:
    Find and delete irrelevant files and folders.
Date:
    July 19th, 2019
Version:
    Alpha 1.0.1
'''

import os
def is_viewer_related(path):
    '''
    Check whether files or folders is related to Viewer
    '''
    file_name = os.path.split(path)[-1]
    if os.path.isfile(path):
        if file_name.find('Viewer')>=0 or file_name.find('DICOMDIR')>=0:
            return True
    elif os.path.isdir(path):
        if file_name.find('Viewer')>=0:
            return True
    else:
        return False

def del_folder(folder):
    '''
    Delete folder recursively(递归的)
    delete the folder and what's in, including files and sub folders
    '''
    fs = os.listdir(folder)
    for f in fs:
        path = os.path.join(folder,f)
        if os.path.isdir(path):
            del_folder(path)
        elif os.path.isfile(path):
            os.remove(path)
        else:
            pass
    os.rmdir(folder)

def delete_viewer_related(folder):
    '''
    Remove Viewer folder and related files in the folder
    '''
    assert os.path.isdir(folder)
    for root,dirs,files in os.walk(folder):
        for name in files:
            path = os.path.join(root, name)
            if is_viewer_related(path):
                print('del:'+path)
                os.remove(path)
        for name in dirs:
            path = os.path.join(root, name)
            if is_viewer_related(path):
                print('del:'+path)
                del_folder(path)

def main():
    folder = '/Volumes/Elements/脊髓肿瘤'
    delete_viewer_related(folder)

if __name__ == '__main__':
    # make sure chinese is well supported
    import io
    import sys
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')
    main()
