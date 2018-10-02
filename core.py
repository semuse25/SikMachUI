import re
import os
import shutil

import cv2
import yaml
import numpy as np
#import skimage.io as io
from tqdm import tqdm
from sklearn.metrics import confusion_matrix 
from keras import backend as K
import tensorflow as tf

from utils import bgr_float32, load_imgs
from utils import assert_exists, assert_not_exists

import traceback
import logging

rect3 = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
rect5 = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
rect7 = cv2.getStructuringElement(cv2.MORPH_RECT,(7,7))
rect9 = cv2.getStructuringElement(cv2.MORPH_RECT,(9,9))

def binarization(img, threshold=100):
    binarized = (img >= threshold).astype(np.uint8) * 255
    return binarized

def dilation(img, kernel=rect3, iterations=1):
    dilated = cv2.dilate(img, kernel, iterations=iterations)
    #cv2.imshow('dilated',dilated); cv2.waitKey(0)
    return dilated

def modulo_padded(img, modulo=16):
    h,w = img.shape[:2]
    h_padding = (modulo - (h % modulo)) % modulo
    w_padding = (modulo - (w % modulo)) % modulo
    if len(img.shape) == 3:
        return np.pad(img, [(0,h_padding),(0,w_padding),(0,0)], mode='reflect')
    elif len(img.shape) == 2:
        return np.pad(img, [(0,h_padding),(0,w_padding)], mode='reflect')


def segment_or_oom(segnet, inp, modulo=16):
    ''' If image is too big, return None '''
    org_h,org_w = inp.shape[:2]

    img = modulo_padded(inp, modulo) #----------------------------- padding
    img_shape = img.shape #NOTE grayscale!
    img_bat = img.reshape((1,) + img_shape) # size 1 batch
    try:
        segmap = segnet.predict(img_bat, batch_size=1)#, verbose=1)
        segmap = segmap[:,:org_h,:org_w,:].reshape((org_h,org_w)) # remove padding
        return segmap
    except Exception as e: # ResourceExhaustedError:
        logging.error(traceback.format_exc())
        print(img_shape,'OOM error: image is too big. (in segnet)')
        return None

seg_limit = 755564# lab-machine
def segment(segnet, inp, modulo=16):
    ''' oom-free segmentation '''
    global seg_limit
    
    h,w = inp.shape[:2]
    result = None
    if h*w < seg_limit:
        result = segment_or_oom(segnet, inp, modulo)
        if result is None: # seg_limit: Ok but OOM occur!
            seg_limit = h*w
            print('segmentation seg_limit =', seg_limit, 'updated!')
    else:
        print('segmentation seg_limit exceed! img_size =', 
              h*w, '>', seg_limit, '= seg_limit')

    if result is None: # exceed seg_limit or OOM
        if h > w:
            upper = segment(segnet, inp[:h//2,:], modulo) 
            downer= segment(segnet, inp[h//2:,:], modulo)
            return np.concatenate((upper,downer), axis=0)
        else:
            left = segment(segnet, inp[:,:w//2], modulo)
            right= segment(segnet, inp[:,w//2:], modulo)
            return np.concatenate((left,right), axis=1)
    else:
        return result # image inpainted successfully!
    

def inpaint_or_oom(img, segmap, complnet, complnet_ckpt_dir, 
                   dilate_kernel=rect5):
    ''' If image is too big, return None '''
    image = img.copy()
    mask = segmap.copy()
    #cv2.imshow('image',image); cv2.imshow('mask',mask); cv2.waitKey(0)

    image = image.reshape(image.shape[:2])
    image = np.stack((image,)*3, -1)
    image = (image * 255).astype(np.uint8)

    mask = binarization(mask, 0.5)
    if dilate_kernel is not None:
        mask = dilation(mask,dilate_kernel)
    mask = np.stack((mask,)*3, -1)

    #print(np.unique(image), np.unique(mask))
    #print(image.shape, mask.shape)
    assert image.shape == mask.shape

    h, w, _ = image.shape
    grid = 8
    image = image[:h//grid*grid, :w//grid*grid, :]
    mask = mask[:h//grid*grid, :w//grid*grid, :]
    #print('Shape of image: {}'.format(image.shape))

    image = np.expand_dims(image, 0)
    mask = np.expand_dims(mask, 0)
    input_image = np.concatenate([image, mask], axis=2)
    sess_config = tf.ConfigProto()
    with tf.Session(config=sess_config) as sess:
        input_image = tf.constant(input_image, dtype=tf.float32)
        output = complnet.build_server_graph(input_image,reuse=tf.AUTO_REUSE)
        output = (output + 1.) * 127.5
        output = tf.reverse(output, [-1])
        output = tf.saturate_cast(output, tf.uint8)
        # load pretrained complnet
        vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        assign_ops = []
        for var in vars_list:
            vname = var.name
            from_name = vname
            var_value = tf.contrib.framework.load_variable(complnet_ckpt_dir, from_name)
            assign_ops.append(tf.assign(var, var_value))
        sess.run(assign_ops)
        print('Model loaded.')
        try:
            result = sess.run(output)
            return result[0][:, :, ::-1]
        except Exception as e: # ResourceExhaustedError:
            logging.error(traceback.format_exc())
            print((h,w), 'OOM error in inpainting')
            return None

compl_limit = 657666 #lab-machine #1525920
def inpaint(img, mask, complnet, complnet_ckpt_dir, dilate_kernel):
    ''' oom-free inpainting '''
    global compl_limit

    cnet_dir = complnet_ckpt_dir
    kernel = dilate_kernel

    h,w = img.shape[:2]
    result = None
    if h*w < compl_limit:
        result = inpaint_or_oom(img, mask, complnet, cnet_dir, dilate_kernel=kernel)
        if result is None: # compl_limit: Ok but OOM occur!
            compl_limit = h*w
            print('compl_limit =', compl_limit, 'updated!')
    else:
        print('compl_limit exceed! img_size =', h*w, '>', compl_limit, '= compl_limit')

    if result is None: # exceed compl_limit or OOM
        if h > w:
            upper = inpaint(img[:h//2,:], mask[:h//2,:], complnet, cnet_dir, kernel) 
            downer= inpaint(img[h//2:,:], mask[h//2:,:], complnet, cnet_dir, kernel)
            return np.concatenate((upper,downer), axis=0)
        else:
            left = inpaint(img[:,:w//2], mask[:,:w//2], complnet, cnet_dir, kernel)
            right= inpaint(img[:,w//2:], mask[:,w//2:], complnet, cnet_dir, kernel)
            return np.concatenate((left,right), axis=1)
    else:
        return result # image inpainted successfully!
