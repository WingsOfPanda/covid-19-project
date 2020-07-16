'''
Author:
    wingsofpanda -- 熊猫之翼
Function:
   eval trained model on testing set
Date:
    Jan. 17th, 2020
Version:
    1.0.0
'''
import SimpleITK as sitk
import tensorflow as tf
import os
import numpy as np
import pickle
import args
import sys
from numpy import transpose
import scipy.ndimage as scimg
import time
from tqdm import tqdm
import scipy.ndimage as scimg
from numpy import transpose
from math import ceil, floor, sqrt
import datetime
import json, pickle
import skimage.measure as msr
from assests.metrics import *
import sklearn.metrics as skms
from sklearn.metrics import confusion_matrix, precision_score, recall_score,f1_score
from sklearn.manifold import TSNE

from skimage import filters, morphology, img_as_int, measure, img_as_float32
import cv2
import numpy as np
import scipy


_EPS = 1.e-8
def softmax_smooth(a):
    if a<0.5:
        score_1=np.array([4.00337500e-05, 6.36561200e-03, 1.74092750e-03, 1.77820570e-02,
       7.49090600e-05, 3.29521370e-02, 9.87742000e-04, 1.64114030e-01,
       7.33439700e-05, 2.27455560e-01, 7.60453500e-02, 9.58791700e-04,
       2.43259070e-04, 7.23258000e-04, 4.29786600e-05, 6.83872400e-04,
       2.31531170e-04, 1.35104870e-03, 2.79385100e-04, 1.00729260e-02,
       1.01642740e-02, 2.25803840e-01, 5.80992030e-02, 7.63616300e-05,
       5.50293630e-05, 4.90132950e-03, 1.86293250e-04, 9.46523100e-04,
       1.42556060e-03, 1.27594990e-01, 1.08752676e-04, 4.73630800e-04,
       1.29308280e-03, 1.97696140e-04, 2.43446500e-02, 4.17136730e-03,
       2.34865380e-04, 1.24499750e-03, 1.16030194e-01, 1.36802570e-03,
       3.42411320e-02, 4.81401160e-02, 2.51708500e-04, 7.68779950e-04,
       1.86474500e-03, 3.28385700e-04, 1.59020130e-04, 2.49050100e-04,
       5.10894300e-05, 3.92904880e-02, 3.96730060e-04, 2.25261440e-04,
       6.57851200e-03, 9.85753700e-04, 1.03753480e-03, 1.10212710e-02,
       3.04624030e-04, 2.52960180e-03, 9.78490000e-04, 1.85045130e-02,
       5.72145040e-04, 2.74208580e-04, 1.79517300e-03, 1.39022160e-02,
       1.16686446e-04, 2.90168260e-03, 8.83703800e-03, 1.30336340e-02,
       2.13588790e-04, 2.04890440e-04, 3.17319720e-03, 2.94411800e-04,
       8.01155100e-04, 3.67287200e-03, 2.43461800e-03, 3.28790770e-04,
       2.76328270e-04, 8.99205500e-03, 1.52899320e-01, 9.49676000e-03,
       2.18428450e-02, 1.80220520e-04, 1.51614870e-03, 4.35180770e-05,
       2.94719190e-02, 1.63903310e-03, 3.97979140e-03, 5.25086100e-04,
       1.19733205e-02, 1.43245000e-03, 1.73398240e-04, 1.58737320e-02,
       4.63924650e-03, 4.15662050e-01, 1.32555680e-04, 1.42876100e-04,
       1.15913720e-02, 1.35442930e-03, 2.30991080e-02, 8.54422200e-02])
        temp=np.sum(score_1>a)/len(score_1)
        return 0.5-temp**0.5*0.5
    else:
        score_2=np.array([0.98828363, 0.95896626, 0.99995077, 0.99802905, 0.9999175 ,
       0.9999387 , 0.9997931 , 0.99688536, 0.9821769 , 0.9999949 ,
       0.9999201 , 0.98233163, 0.9999002 , 0.9974427 , 0.99980396,
       0.9987031 , 0.99848914, 0.99996316, 0.9986721 , 0.98657995,
       0.9672934 , 0.9922072 , 0.88175744, 0.9896987 , 0.9673885 ,
       0.99741215, 0.9996486 , 0.9999993 , 0.9653799 , 0.998495  ,
       0.99999726, 0.97648484, 0.99999356, 0.99996483, 0.9998683 ,
       0.9959985 , 0.7743    , 0.9999416 , 0.99999833, 0.9998092 ,
       0.9999379 , 0.99152476, 0.9998247 , 0.98990434, 0.99994564,
       0.99979645, 0.9998368 , 0.96858346, 0.9980932 , 0.99979144,
       0.85772353, 0.9993056 , 0.9811917 , 0.9999993 , 0.99970067,
       0.999956  , 0.9947208 , 0.8432001 , 0.8515797 , 0.9951421 ,
       0.9275501 , 0.6361144 , 0.94015414, 0.9999329 , 0.9986721 ,
       0.9999943 , 0.9968292 , 0.9997552 , 0.97813714, 0.9990458 ,
       0.99659616, 0.9940058 , 0.9831678 , 0.9951074 , 0.99954164,
       0.63550645, 0.99897134, 0.9999149 , 0.99989235, 0.84570384,
       0.994544  , 0.9997695 , 0.9267402 , 0.9999422 , 0.9997228 ,
       0.99886644, 0.99992955, 0.9261854 , 0.58992887, 0.99999905,
       0.8874202 , 0.8878896 , 0.9982052 , 0.9766027 , 0.9994537 ,
       0.99263   , 0.99977225, 0.9999771 , 0.98465145, 0.9648839 ])
        temp=np.sum(score_2<a)/len(score_2)
        return 0.5+temp**0.5*0.5

def plot_Fscore_table(label_true,pred,num_class,plot_flag=False):
    precise  = precision_score( label_true, pred, range(num_class), average=None)
    recall   = recall_score(  label_true, pred, range(num_class), average=None)
    f1=f1_score( label_true, pred,  range(num_class), average=None)
    confusion=confusion_matrix(label_true, pred, range(num_class))
    Counts=np.sum(confusion,1)

    precise_=['{:.3f}'.format(i) for i in precise]
    recall_=['{:.3f}'.format(i) for i in recall]
    f1_=['{:.3f}'.format(i) for i in f1]

    return precise,recall,f1,confusion

def getChestMask(input_image):
    #output_image = np.zeros(input_image.shape).astype('int32')
    output_masks = np.zeros(input_image.shape).astype('int32')
    for i in range(0, input_image.shape[0]):
        image_out = input_image[i, :, :]
        image_out[image_out[:, :] < -1024] = -1024
        image_out = image_out + 1024
        output_mask = remove_raw_radiation(image_out)
        #image_out = np.multiply(image_out, output_mask)
        #output_image[:, :, i] = image_out
        output_masks[i, :, :] = output_mask
    return output_masks #output_image,


def remove_raw_radiation(input_image):

    input_image = img_as_float32(input_image)
    val = filters.threshold_otsu(input_image)
    ret1, bw_image = cv2.threshold(input_image, val, 1, cv2.THRESH_BINARY)
    labeled_mask = measure.label(bw_image, 4)
    props = measure.regionprops(labeled_mask)
    # print len(props)
    maxAreaLabel = 1
    maxArea = 0
    for prop in props:
        if prop.area > maxArea:
            maxAreaLabel = prop.label
            maxArea = prop.area
    # print maxAreaLabel
    labeled_mask[labeled_mask != maxAreaLabel] = 0
    labeled_mask[labeled_mask >0] = 1
    output_mask = scipy.ndimage.binary_fill_holes(labeled_mask) #morphology.convex_hull_image(labeled_mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_ERODE, (7, 7))
    output_mask = output_mask.astype('uint8')

    return output_mask

def np2itk_writer(img_np, ref_itk, itk_name, no_iso=True):
    img_np = img_np.astype(np.float32)
    img = sitk.GetImageFromArray(img_np)
    img.SetOrigin(ref_itk.GetOrigin())
    img.SetDirection(ref_itk.GetDirection())
    if no_iso:
        img.SetSpacing(ref_itk.GetSpacing())
    sitk.WriteImage(img, itk_name)
    return img

def nii_image_reader(img_dir):
    img_itk = sitk.ReadImage(img_dir)
    img_np = sitk.GetArrayFromImage(img_itk)
    return img_np, img_itk

def sd_normalize(img_np, eps=None):

    std = np.std(img_np)
    mean = np.mean(img_np)
    if eps is not None:
        std_normal = (img_np - mean)/(std+_EPS)
    else:
        std_normal = (img_np - mean)/std
    return std_normal


def center_crop_normalize_fixed_size(img_np, headMask=None, series_class=None, resize_order=0, special_clip=None, processedSize=512, cropSize=416, resize2=None):

    zoom =  [1, processedSize/img_np.shape[1], processedSize/img_np.shape[2]]

    if headMask is not None:
        _headMask = scimg.interpolation.zoom(headMask.copy(), zoom, order=0)

    if series_class == 'mask':
        xTmp = scimg.interpolation.zoom(img_np, zoom, order=0)
        xTmp = center_crop(_headMask, xTmp, cropSize=cropSize)
        if resize2 is not None:
            zoom2resize = [resize2[0]/xTmp.shape[0], resize2[1]/xTmp.shape[1], resize2[2]/xTmp.shape[2]]
            xTmp = scimg.interpolation.zoom(xTmp, zoom2resize, order=0)
    elif series_class == 'image':
        xTmp = scimg.interpolation.zoom(img_np, zoom, order=resize_order)
        HU_cut = center_crop(_headMask, xTmp, cropSize=cropSize)
        if resize2 is not None:
            zoom2resize = [resize2[0]/HU_cut.shape[0], resize2[1]/HU_cut.shape[1], resize2[2]/HU_cut.shape[2]]
            HU_cut = scimg.interpolation.zoom(HU_cut, zoom2resize, order=resize_order)
        HU_cut[HU_cut<-1024] = -1024
        if special_clip is not None:
            HU_cut = np.clip(HU_cut, special_clip[0], special_clip[1])
        xTmp = sd_normalize(HU_cut)
        xTmp = (xTmp-np.amin(xTmp)) / (np.amax(xTmp)-np.amin(xTmp))
    else:
        print('series type is not defined!')
        return None

    x = xTmp.astype(np.float32)
    return x

def center_crop(skull_label, img_to_crop, cropSize=416, precessedSize=512):

    centerLayers = ceil(skull_label.shape[0]/2)
    label = skull_label[centerLayers,...]
    label = label.astype(int)
    proceedingSize = int(cropSize/2)

    for prop in msr.regionprops(label):
        min_row, min_col, max_row, max_col =  prop.bbox #(min_depth, min_row, min_col, max_depth, max_row, max_col)

    radius_x = round((max_row - min_row)/2)+1
    radius_y = round((max_col - min_col)/2)+1

    center_x = radius_x + min_row
    center_y = radius_y + min_col

    leftMin = np.amin((center_x, proceedingSize))
    rightMax = np.amin((precessedSize - center_x, proceedingSize))

    topMin = np.amin((center_y, proceedingSize))
    bottomMax = np.amin((precessedSize - center_y, proceedingSize))


    cropped_centered = np.ones([skull_label.shape[0], cropSize, cropSize])*(-1024)

    cropped_centered[::, proceedingSize-leftMin: proceedingSize+rightMax, proceedingSize-topMin: proceedingSize+bottomMax] = img_to_crop[::, center_x-leftMin: center_x+rightMax, center_y-topMin: center_y+bottomMax]

    return cropped_centered


def batch_eval_save(sess, img_dir, X, y, bodyMaskDir=None, readyMaskDir=None):

    if readyMaskDir is None:

        img_np, img_itk = nii_image_reader(img_dir)

        if bodyMaskDir is None:
            print('no bodymask, creating one')
            try:
                chestMask_np = getChestMask(img_np)
            except:
                print('possible img error: {}'.format(img_dir))
                return None
        else:
            chestMask_np,_ = nii_image_reader(bodyMaskDir)

        chestMask_np[chestMask_np>0] = 1
        chestMask_np[chestMask_np!=1] = 0

        centerCropRatio = 448
        resize2 = [64, 224, 224]
        
        chestMaskCrop = center_crop_normalize_fixed_size(chestMask_np, headMask=chestMask_np, series_class='mask', resize_order=0, cropSize=centerCropRatio, resize2=resize2)
        chestMaskCrop[chestMaskCrop>0] = 1
        # np2itk_writer(labelCrop, img_itk, os.path.join(last_folder, 'dpmaskCrop{}.nii.gz'.format(minSize)), no_iso=True)

        maxClip = 1024
        minClip = -1024
        imgCropMaxminClip = center_crop_normalize_fixed_size(img_np.copy(), headMask=chestMask_np, series_class='image', resize_order=1, special_clip=[minClip, maxClip], cropSize=centerCropRatio, resize2=resize2)
        imgCropMaxminClip = imgCropMaxminClip*chestMaskCrop

    else:
        imgCropMaxminClip, _ = nii_image_reader(readyMaskDir)


    img_ready = imgCropMaxminClip.transpose((1,2,0)).astype(np.float32)

    img_ready = np.expand_dims(img_ready, axis=-1)
    img_ready = np.expand_dims(img_ready, axis=0)

    ts = time.time()
    patch_tmp = sess.run(y, feed_dict={X: img_ready.astype(np.float32)})
    te = time.time()


    return np.argmax(patch_tmp, axis=-1)[0]

def json_loader(json_dir):
    with open (json_dir, 'rb') as fp:
        files = json.load(fp)
    return files

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

def folder_counting(roots, key=None):
    folder_list = []
    for (root, dirs, files) in os.walk(roots):
        for file in files:
            if key in file:
                folder_list.append(root)
                break
    return folder_list

def folder_dir_concating(roots_list, key = None):
    dir_list = []
    for roots in roots_list:
        dir_list += folder_counting(roots, key='.nii.gz')
    return dir_list


if __name__ == '__main__':


    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  use graphdef to pred >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


    graphdef_dir = './model_transform_tmp/epoch_2_valid_0.8546666666666667model.graphdef'
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True

    with tf.gfile.GFile(graphdef_dir, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())


    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def,
            input_map=None,
            return_elements=None,
            name=None,
            op_dict=None,
            producer_op_list=None
        )
        inputs = graph.get_tensor_by_name('import/input:0')
        outputs = graph.get_tensor_by_name('import/output_softmax:0')
        print('Model restored from graphdef: {}'.format(graphdef_dir))

        with tf.Session(graph=graph, config=config) as sess:


    #         #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> this is used for pred a single imgs >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

            img_file_dir = None # file that need to be tested

            prob = batch_eval_save(sess, img_file_dir, inputs, outputs)
            prob_soft = softmax_smooth(prob)

            print('covid prob:{}, and skewed: {}'.format(prob, prob_soft))

   