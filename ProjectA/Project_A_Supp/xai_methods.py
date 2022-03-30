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

################################################################################

from tensorflow.keras.preprocessing import image as image_
import cv2 as cv
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model,load_model

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

np.random.seed(331)


def perturb_1d(input, perturb, sec_size):
    mask = np.zeros(input.shape)[0]
    not_masked = np.where(perturb == 1)[0]
    for i, val in enumerate(not_masked):
        if val != 0:
            for j in range(sec_size):
                mask[sec_size * i + j] = 1
    perturbed = input * mask[:, np.newaxis]
    return perturbed

def LIME_1d(input,
            model,
            label,
            num_perturb=300,
            sec_size=4,
            kernel_w=0.25,
            num_feats=4):

    num_superpixels = math.ceil(input.shape[1] / sec_size)
    perturbs = np.random.binomial(1, 0.5, size=(num_perturb, num_superpixels))
    preds = []
    for i in perturbs:
        perturbed = perturb_1d(input, i, sec_size)
        pred = model.predict(perturbed)
        preds.append(pred)
    preds = np.array(preds)
    orig_img = np.ones(num_superpixels)[np.newaxis, :]
    dists = sklearn.metrics.pairwise_distances(perturbs,
                                               orig_img,
                                               metric='cosine').ravel()
    weights = np.sqrt(np.exp(-(dists**2) / kernel_w**2))
    lime_model = LinearRegression()
    lime_model.fit(perturbs, preds[:, :, label], weights)
    c = lime_model.coef_[0]
    top_feats = np.argsort(c)[-num_feats:]
    mask_superpixel = np.zeros(num_superpixels)
    mask_superpixel[top_feats] = True
    mask = np.zeros(input.shape[1])
    for i, val in enumerate(mask_superpixel):
        if val != 0:
            for j in range(sec_size):
                mask[sec_size * i + j] = 1
    perturbed = input * mask[:, np.newaxis]

    c_exp = np.zeros(input.shape[1])
    for i, c_i in enumerate(c):
        for j in range(sec_size):
            if c_i > 0:
                c_exp[sec_size * i + j] = c_i

    return c_exp, perturbed[0]

def perturb_2d(input, perturb, segs):
    mask = np.zeros(segs.shape)
    not_masked = np.where(perturb == 1)[0]
    for i in not_masked:
        mask[segs == i] = 1
    perturbed = copy.deepcopy(input)
    perturbed = perturbed * mask[:, :, np.newaxis]
    return perturbed


def LIME_2d(input,
            model,
            label,
            num_perturb=300,
            kernel_size=4,
            max_dist=200,
            ratio=0.2,
            kernel_w=0.25,
            num_feats=10):
    superpixels = skimage.segmentation.quickshift(input,
                                                  kernel_size=kernel_size,
                                                  max_dist=max_dist,
                                                  ratio=ratio)
    num_superpixels = np.unique(superpixels).shape[0]
    perturbs = np.random.binomial(1, 0.5, size=(num_perturb, num_superpixels))
    preds = []
    for i in perturbs:
        perturbed = perturb_2d(input, i, superpixels)
        pred = model.predict(perturbed[np.newaxis, :, :, :])
        preds.append(pred)
    preds = np.array(preds)
    orig_img = np.ones(num_superpixels)[np.newaxis, :]
    dists = sklearn.metrics.pairwise_distances(perturbs,
                                               orig_img,
                                               metric='cosine').ravel()
    weights = np.sqrt(np.exp(-(dists**2) / kernel_w**2))
    lime_model = LinearRegression()
    lime_model.fit(perturbs, preds[:, :, label], weights)
    c = lime_model.coef_[0]
    top_feats = np.argsort(c)[-num_feats:]
    mask = np.zeros(num_superpixels)
    mask[top_feats] = True
    perturbed = perturb_2d(input, mask, superpixels)
    return perturbed


################################# ABLATION-CAM #################################
def ablation_cam_1d(input_model, image, layer_name):

    output = input_model.output
    conv_layer_output = input_model.get_layer(layer_name).output
    ac_model = keras.models.Model([input_model.input],
                                  [conv_layer_output, output])

    ff_results = ac_model([image])
    all_fmap_masks, predictions = ff_results[0], ff_results[-1]

    predicted_label = np.argmax(predictions, axis=1)[0]

    y_c = predictions[0][predicted_label]

    w_t = np.zeros(input_model.get_layer(layer_name).get_weights()[1].shape)

    all_weigths = input_model.get_layer(layer_name).get_weights().copy()
    zero_weigth = all_weigths[0][:, :, 0] * 0
    weigth_local = [np.zeros(all_weigths[0].shape)]
    weigth_local.append(np.zeros(all_weigths[1].shape))
    for i in range(w_t.shape[0]):
        weigth_local[0] = all_weigths[0].copy()
        weigth_local[0][:, :, i] = zero_weigth
        input_model.get_layer(layer_name).set_weights(weigth_local)
        y_k = input_model.predict([image])[0][predicted_label]
        w_t[i] = (y_c - y_k) / y_c

    a_k = all_fmap_masks[0]
    ab_map = a_k * w_t

    explanation = np.sum(ab_map, axis=1)
    return explanation


def ablation_cam_2d(input_model, image, layer_name):

    output = input_model.output
    conv_layer_output = input_model.get_layer(layer_name).output
    ac_model = keras.models.Model([input_model.input], [conv_layer_output, output])

    ff_results = ac_model([image])
    all_fmap_masks, predictions = ff_results[0], ff_results[-1]

    pred_label = np.argmax(predictions, axis=1)[0]

    y_c = predictions[0][pred_label]

    w_t = np.zeros(input_model.get_layer(layer_name).get_weights()[1].shape)

    all_weigths = input_model.get_layer(layer_name).get_weights().copy()
    zero_weigth = all_weigths[0][:, :, :, 0] * 0
    weigth_local = [np.zeros(all_weigths[0].shape)]
    weigth_local.append(np.zeros(all_weigths[1].shape))
    for i in range(w_t.shape[0]):
        weigth_local[0] = all_weigths[0].copy()
        weigth_local[0][:, :, :, i] = zero_weigth
        input_model.get_layer(layer_name).set_weights(weigth_local)
        y_k = input_model.predict([image])[0][pred_label]
        w_t[i] = (y_c - y_k) / y_c

    a_k = all_fmap_masks[0]
    ab_map = a_k * w_t

    explanation = np.sum(ab_map, axis=2)
    explanation = np.maximum(explanation, 0)

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
    if class_index == None:
        cls = np.argmax(input_model.predict(image))
    else:
        cls = class_index
    #             loss = predictions[:, cls]
    #         grads_val = tape3.gradient(loss, all_fmap_masks)
    #     grads_val2 = tape2.gradient(grads_val, all_fmap_masks)
    # grads_val3 = tape1.gradient(grads_val2, all_fmap_masks)
    with tf.GradientTape() as tape:
        ff_results = feedforward1([image])
        all_fmap_masks, predictions = ff_results[0], ff_results[-1]
        loss = predictions[:, cls]
    grads_val = tape.gradient(loss, all_fmap_masks)
    grads_val2 = grads_val**2
    grads_val3 = grads_val2 * grads_val
    if len(image.shape) == 3:
        axis = (0, 1)
    elif len(image.shape) == 4:
        axis = (0, 1, 2)
    alpha_div = (2.0 * grads_val2 + grads_val3 * np.sum(all_fmap_masks, axis))
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