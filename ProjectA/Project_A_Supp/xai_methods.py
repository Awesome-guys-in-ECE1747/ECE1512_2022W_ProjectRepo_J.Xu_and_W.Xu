################################ DEPENDENCIES #################################

import cv2
from time import time
import os
import numpy as np
import json

import tensorflow as tf
from tensorflow import keras
from scipy.ndimage.interpolation import zoom
from keras.preprocessing.image import load_img, img_to_array
import keras.backend as K
from PIL import Image, ImageDraw
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras import backend as K
import numpy as np
import cv2
import matplotlib.pyplot as plt

################################################################################

from PIL import Image
from tensorflow.keras.preprocessing import image as image_
import tensorflow as tf
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model

import numpy as np
import keras
from keras.applications.imagenet_utils import decode_predictions
import skimage.io 
import skimage.segmentation
import copy
import sklearn
import sklearn.metrics
from sklearn.linear_model import LinearRegression
import math

##################################### LIME #####################################

np.random.seed(50)

def perturb_img_1d(img, perturb, sec_size):
    mask = np.zeros(img.shape)[0]
    not_masked = np.where(perturb == 1)[0]
    for i, val in enumerate(not_masked):
        if val != 0:
            for j in range(sec_size):
                mask[sec_size*i + j] = 1
    perturbed_img = img*mask[:,np.newaxis]
    return perturbed_img

def perturb_img(img, perturb, segs):
    mask = np.zeros(segs.shape)
    not_masked = np.where(perturb == 1)[0]
    for i in not_masked:
        mask[segs == i] = 1
    perturbed_img = copy.deepcopy(img)
    perturbed_img = perturbed_img*mask[:,:,np.newaxis]
    return perturbed_img


def LIME_2d(img, model, label, num_perturb=300, kernel_size=4,max_dist=200, ratio=0.2, kernel_w=0.25, num_feats=10):
    superpixels = skimage.segmentation.quickshift(img, kernel_size=kernel_size,max_dist=max_dist, ratio=ratio)
    num_superpixels = np.unique(superpixels).shape[0]
    perturbs = np.random.binomial(1, 0.5, size=(num_perturb, num_superpixels))
    preds = []
    for i in perturbs:
        perturbed_img = perturb_img(img, i, superpixels)
        pred = model.predict(perturbed_img[np.newaxis,:,:,:])
        preds.append(pred)
    preds = np.array(preds)
    orig_img = np.ones(num_superpixels)[np.newaxis,:]
    dists = sklearn.metrics.pairwise_distances(perturbs, orig_img, metric='cosine').ravel()
    weights = np.sqrt(np.exp(-(dists**2)/kernel_w**2))
    lime_model = LinearRegression()
    lime_model.fit(perturbs, preds[:,:,label], weights)
    c = lime_model.coef_[0]
    top_feats = np.argsort(c)[-num_feats:]
    mask = np.zeros(num_superpixels) 
    mask[top_feats]= True 
    perturbed_img = perturb_img(img,mask,superpixels)
    return perturbed_img

def LIME_1d(img, model, label, num_perturb=300, sec_size=4, kernel_w=0.25, num_feats=4):

    
    num_superpixels = math.ceil(img.shape[1]/sec_size)
    perturbs = np.random.binomial(1, 0.5, size=(num_perturb, num_superpixels))
    preds = []
    for i in perturbs:
        perturbed_img = perturb_img_1d(img, i, sec_size)
        pred = model.predict(perturbed_img)
        preds.append(pred)
    preds = np.array(preds)
    orig_img = np.ones(num_superpixels)[np.newaxis,:]
    dists = sklearn.metrics.pairwise_distances(perturbs, orig_img, metric='cosine').ravel()
    weights = np.sqrt(np.exp(-(dists**2)/kernel_w**2))
    lime_model = LinearRegression()
    lime_model.fit(perturbs, preds[:,:,label], weights)
    c = lime_model.coef_[0]
    top_feats = np.argsort(c)[-num_feats:]
    mask_superpixel = np.zeros(num_superpixels) 
    mask_superpixel[top_feats]= True 
    mask = np.zeros(img.shape[1])
    for i, val in enumerate(mask_superpixel):
        if val != 0:
            for j in range(sec_size):
                mask[sec_size*i + j] = 1
    perturbed_img = img*mask[:,np.newaxis]
    
    c_exp = np.zeros(img.shape[1])
    for i, c_i in enumerate(c):
        for j in range(sec_size):
                if c_i > 0:
                    c_exp[sec_size*i + j] = c_i

    return c_exp, perturbed_img[0]
################################# ABLATION-CAM #################################
def ablation_cam_1d(input_model, image, layer_name):

    y_c = input_model.output
    conv_output = input_model.get_layer(layer_name).output
    ac_model = keras.models.Model([input_model.input], [conv_output, y_c])

    ff_results=ac_model([image])
    all_fmap_masks, predictions = ff_results[0], ff_results[-1]
    # print(image.shape)
    # print(predictions)
    #index
    _pred = np.argmax(predictions, axis=1)[0]
    # print(_pred)
    # Real Y_C

    _y_c = predictions[0][_pred]
    # print(_y_c)

    _wt = np.zeros(input_model.get_layer(layer_name).get_weights()[1].shape)
    # print(_wt)
    # print(_wt.shape)
    all_weigths = input_model.get_layer(layer_name).get_weights().copy()
    zero_weigth = all_weigths[0][:, :, 0] * 0
    weigth_local = [np.zeros(all_weigths[0].shape)]
    weigth_local.append(np.zeros(all_weigths[1].shape))
    for i in range(_wt.shape[0]):
        weigth_local[0] = all_weigths[0].copy()
        weigth_local[0][:, :, i] = zero_weigth
        input_model.get_layer(layer_name).set_weights(weigth_local)
        y_k = input_model.predict([image])[0][_pred]
        _wt[i] = (_y_c - y_k) / _y_c

    a_k = all_fmap_masks[0]

    ab_map = a_k * _wt
    # print(ab_map)
    explanation = np.sum(ab_map, axis=1)
    return explanation


def ablation_cam_2d(input_model, image, layer_name):

    y_c = input_model.output
    conv_output = input_model.get_layer(layer_name).output
    ac_model = keras.models.Model([input_model.input], [conv_output, y_c])

    ff_results=ac_model([image])
    all_fmap_masks, predictions = ff_results[0], ff_results[-1]

    _pred = np.argmax(predictions, axis=1)[0]
    # print(_pred)
    # Real Y_C

    _y_c = predictions[0][_pred]
    # print(_y_c)

    # print(input_model.get_layer(layer_name))
    _wt = np.zeros(input_model.get_layer(layer_name).get_weights()[1].shape)
    # print(_wt)
    # print(_wt.shape)
    all_weigths = input_model.get_layer(layer_name).get_weights().copy()
    zero_weigth = all_weigths[0][:, :, :, 0] * 0
    weigth_local = [np.zeros(all_weigths[0].shape)]
    weigth_local.append(np.zeros(all_weigths[1].shape))
    for i in range(_wt.shape[0]):
        weigth_local[0] = all_weigths[0].copy()
        weigth_local[0][:, :, :, i] = zero_weigth
        input_model.get_layer(layer_name).set_weights(weigth_local)
        y_k = input_model.predict([image])[0][_pred]
        _wt[i] = (_y_c - y_k) / _y_c

    a_k = all_fmap_masks[0]
    # print(a_k)

    ab_map = a_k * _wt
    # print(ab_map)
    explanation = np.sum(ab_map, axis=2)
    explanation = np.maximum(explanation, 0)
    # print(explanation)
    return explanation



################################# Deprecated #################################
def grad_cam_plus_plus(input_model, image, layer_name, class_index=None):
    # if class_index is None:
    #     class_index=np.argmax(input_model.predict(np.array([image])), axis=-1)[0]
    """GradCAM method for visualizing input saliency."""
    cls = np.argmax(input_model.predict(image))
    y_c = input_model.output
    conv_output = input_model.get_layer(layer_name).output
    feedforward1 = keras.models.Model([input_model.input], [conv_output, y_c])
    # with tf.GradientTape() as tape1:
    #     with tf.GradientTape() as tape2:
    #         with tf.GradientTape() as tape3:
    #             ff_results = feedforward1([image])
    #             all_fmap_masks, predictions = ff_results[0], ff_results[-1]
    if class_index==None:
        cls=np.argmax(input_model.predict(image))
    else:
        cls=class_index
    #             loss = predictions[:, cls]
    #         grads_val = tape3.gradient(loss, all_fmap_masks)
    #     grads_val2 = tape2.gradient(grads_val, all_fmap_masks)
    # grads_val3 = tape1.gradient(grads_val2, all_fmap_masks)
    with tf.GradientTape() as tape:
        ff_results=feedforward1([image])
        all_fmap_masks, predictions = ff_results[0], ff_results[-1]
        loss = predictions[:, cls]
    grads_val = tape.gradient(loss, all_fmap_masks)
    grads_val2=grads_val**2
    grads_val3=grads_val2*grads_val
    if len(image.shape) == 3:
        axis = (0, 1)
    elif len(image.shape) == 4:
        axis = (0, 1, 2)
    alpha_div=(2.0 * grads_val2 + grads_val3 * np.sum(all_fmap_masks, axis))
    alpha_div = np.where(alpha_div != 0.0, alpha_div, 0)
    alpha = grads_val2 / alpha_div
    weights = np.maximum(grads_val, 0.0) * alpha
    weights = np.sum(weights, axis=axis)
    # weights = np.mean(grads_val, axis=axis)
    cam = np.dot(all_fmap_masks[0], weights)
    # print (cam)
    H, W = image.shape[1:3]
    cam = np.maximum(cam, 0)
    # cam = resize(cam, (H, W))
    cam = zoom(cam, H / cam.shape[0])
    # cam = np.maximum(cam, 0)
    cam = cam / cam.max()
    return cam