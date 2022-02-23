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
#############################

def layer_finder(k_model, model_arch, pool_input=True):

  '''
  Returns a list of all of the last layers in each block of the model.

    Parameters:
      k_model (Keras model): Either a VGG or ResNet
      model_arch (str): Either "VGG" or "ResNet"

    Returns:
      last_layers (list): A list of all of the last layers in each block of the
      model.
  '''
  
  if type(model_arch) != str:
    raise TypeError("Input argument \"model_arch\" must be a string that is\
                      either \"VGG\" or \"ResNet\".")

  last_layers = []
  pool_flag=False
  block_end_detected=False
  first_layer=True
  j=0

  if model_arch == "VGG":
 
    for layer in k_model.layers:
      if type(layer) == tf.keras.layers.MaxPool2D:
        last_layers.append(layer.name)

  elif model_arch == "ResNet":


    for i in range(len(k_model.layers)):
      if i<j: continue
      #print(k_model.layers[i])
      if len(k_model.layers[i+1].output.get_shape()) < 4:
        # only save a layer if the block before the end was a convolutional block
            last_layers.append(k_model.layers[i].name)
            break
      
      if k_model.layers[i+1].output.get_shape()[2]<k_model.layers[i].output.get_shape()[2]-4:
          if pool_input==True:
              if type(k_model.layers[i]) == tf.keras.layers.InputLayer: continue
              if 'ZeroPadding2D' in str(type(k_model.layers[i])):
                  if type(k_model.layers[i-1]) == tf.keras.layers.InputLayer: continue
                  last_layers.append(k_model.layers[i-1].name)
              else:
                  last_layers.append(k_model.layers[i].name)
          else:
              if first_layer:
                  j=i+1
                  pool_flag=True
                  while(pool_flag):
                      j += 1
                      #print(str(type(k_model.layers[j])))
                      if  'Conv2D' in str(type(k_model.layers[j])):
                          #print('Here')
                          last_layers.append(k_model.layers[j-1].name)
                          first_layer=False
                          pool_flag=False
              else:
                  j=i
                  pool_flag=True
                  while(pool_flag):
                      j += 1
                      #print(str(type(k_model.layers[j])))
                      if 'merge.Add' in str(type(k_model.layers[j])):
                          block_end_detected=True
                          #print(j)
                      elif block_end_detected==True and 'Conv2D' in str(type(k_model.layers[j])):
                          #print('Here')
                          last_layers.append(k_model.layers[j-1].name)
                          block_end_detected=False
                          pool_flag=False
  else:
    
    print("Input argument \"model_arch\" must be either \"VGG\" or \"ResNet\".")

  return [[lay] for lay in last_layers]
  
def create_random_mask(h=7, w=7, H=224, W=224, p_1=0.5, resample=Image.BILINEAR):
    '''
    Generates one random mask utilized in RISE
    inputs:
        h, w: initial size of binary mask
        H, W: final size of the upsampled mask
        p_1: probability of actiating pixels in the down-sampled masks.
        interp: upsampling technique.
    returns:
        mask: a smooth mask with the values in range [0,1] with size of HxW.
    '''
    assert H>h, 'Masks should be resized to higher dimensions.'
    assert W>w, 'Masks should be resized to higher dimensions.'
    # create random binary hxw mask
    mask=np.random.choice([0, 1], size=(h, w), p=[1-p_1, p_1])

    # upsample mask to (h+H,w+W)
    mask = Image.fromarray(mask*255.)
    mask = mask.resize((H + h, W + w), resample=resample)
    mask = np.array(mask)

    # randomly crop mask to HxW
    w_crop = np.random.randint(0,w+1)
    h_crop = np.random.randint(0,h+1)
    mask = mask[h_crop:H + h_crop, w_crop:W + w_crop]

    # normalize between 0 and 1
    mask /= np.max(mask)

    return mask

def create_attribution_masks(img, model, layers, class_index, max_mask_num, interp='bilinear'):
    '''
    Derives feature maps from one, or a couple of layers, and post-processes them
    to convert them to attribution masks.

    inputs:
        img: a 4-D tensor image.
        model: the classification model
        layers: list of layers to be visualized either individually or mutually.
        class_index: the output class according to whom the layer(s) are visualized.
        max_mask_num: the threshold "normalized gradient" value for sampling attribution masks (\mu in our paper)
        interp: upsampling technique.
        For now, 'bilinear' and 'nearest' are supported.
    returns:
        masks: a set of attribution masks normalized between 0 and 1.
    '''
    assert interp in ['bilinear', 'nearest'], 'Selected upsampling type undefined or unsupported.'
    # Forward pass to get attribution masks.
    conv_outputs=[]
    for layer in model.layers:
        if np.isin(layer.name,layers):
            conv_outputs.append(layer.output)
    conv_outputs.append(model.output)
    feedforward1=keras.models.Model([model.input], [conv_outputs])
    with tf.GradientTape() as tape:
        ff_results=feedforward1([img])[0]
        all_fmap_masks, predictions = ff_results[:-1], ff_results[-1]
        loss = predictions[:, class_index]
    grads = tape.gradient(loss, all_fmap_masks)
    ###
    
    # upsample and normalize masks.
    num_masks=0
    masks=[]
    for i in range(len(layers)):
        tmp_mask = all_fmap_masks[i][0].numpy()
        if len(img.shape)==3:
            axis=0
            size=img.shape[1:]
            tmp_mask = np.expand_dims(tmp_mask, axis=1)
        elif len(img.shape)==4:
            axis=(0,1)
            size=img.shape[1:-1]
        significance = np.mean(grads[i][0], axis=axis)
        #idxs = np.argpartition(significance, -1*max_mask_num)[-1*max_mask_num:]
        idxs = np.where(significance>max_mask_num*np.max(significance))[0]
        if interp == 'bilinear':
            fmap = tf.image.resize(tmp_mask[...,idxs], size, method='bilinear').numpy()
        elif interp == 'nearest':
            fmap = tf.image.resize(tmp_mask[...,idxs], size, method='nearest').numpy()
        else: raise ValueError('You have selected an unsupported interpolation type.')
        
        num_masks+=fmap.shape[2]
        fmap -= np.min(fmap, axis=(0,1))
        fmap /= (np.max(fmap, axis=(0,1))+10e-7)
        masks.append(fmap) 
    return masks

def visualize_layers(img, model, class_index, masks, H=224, W=224, C=3, batch_size = 128):
    '''
    Combines attribution masks using the RISE-based framework mentioned in
    SISE white paper.
    inputs:
        img: a 3-D tensor image.
        model: the classification model
        class_index: the output class according to whom the layer(s) are visualized.
        masks: a set of attribution masks normalized between 0 and 1.
    returns:
        sum_masks: visualization map of the selected layer(s).
    This function follows 'create_attribution_masks()'.
    '''
    # creates perturbed images to probe model.
    img = img if len(img.shape)==3 else np.expand_dims(img, axis=1)
    X = np.einsum('hwc,hwn->nhwc', img, masks)
    # second forward pass to valuate attribution maps
    preds_masked = np.empty([0])
    if masks.shape[2] <= batch_size :
      preds_masked=np.append(preds_masked, model(X, training=False)[:,class_index],axis=0)
    else :
      for i in range (0, masks.shape[2]-batch_size, batch_size) :
        preds_masked=np.append(preds_masked, model(X[i:i+batch_size], training=False)[:,class_index],axis=0)
      preds_masked=np.append(preds_masked, model(X[i+batch_size:], training=False)[:,class_index],axis=0)
    
    # Linear combination of attribution masks.
    masks /= (masks.sum(axis=(0,1))+10e-7)
    sum_mask = np.einsum('hwn,n->hw', masks, preds_masked)

    sum_mask -= np.min(sum_mask)
    sum_mask /= np.max(sum_mask)
    return sum_mask
    
def otsu(I, nbins=256, tau=1.5):
    '''
    Finds the optimum adaptive threshold value for a 2-D image.
    inputs:
        I: a 2-D image (visualization map/ heat-map/ etc.)
        nbins: resolution of histogram. Increasing this parameter yields to more
        precise threshold value, achieved in longer time.
        tau: bottleneck amplititude
        returns: Otsu adaptive threshold value
    '''
    I = np.round(I*nbins)
    #histogram of the image
    hist, bins = np.histogram(I.ravel(),nbins,[0,nbins])
    #CDF/ mean/ variance terms for multiple values
    i = np.arange(nbins)
    varsb = np.zeros(nbins)
    for j in range(1, nbins):
        w0 = np.sum(hist[0:j])
        w1 = np.sum(hist[j:nbins])
        u0 = np.sum(np.multiply(hist[0:j], i[0:j])) / w0
        u1 = np.sum(np.multiply(hist[j:nbins], i[j:nbins])) / w1
        varsb[j] = w0 * w1 * (u0-u1) * (u0-u1)
    # the threshold value is the one maximizing the variance term.
    t = np.argmax(varsb)
    #print(t)
    k = round(t*tau)
    if np.sum(hist[int(k):256]) < .1 * np.sum(hist):
        #print('happened')
        return t*tau/nbins
    else:
        return t/nbins

def otsu_sigmoid(I, nbins=256, T=100., tau=1.5):
    '''
        Thresholds the 2-D visualization map softly, combining Otsu's method and
        sigmoid function.
        inputs:
            I: a 2-D image (visualization map/ heat-map/ etc.)
            nbins: resolution of histogram. Increasing this parameter yields to more
            precise threshold value, achieved in longer time.
            T: sigmoid temparature (preferred to be set to high values.)     
        returns:
            the soft-thresholded heat-map according to the input.
    '''
    thr=otsu(I, nbins=256, tau=1.5)
    return 1/(1 + np.exp(-(I-thr)*T)) 

def fuse_visualization_maps(exmaps, fusion_type='otsu', T=100.):
    '''
    Fuses visualization maps to a unique explanation map. Visualization maps should
    be given with the correct order (low-level layer to high-level layer)

    '''
    assert fusion_type in ['simple', 'otsu']
    ex=exmaps[0]
    if fusion_type=='simple':
        for i in range(1, len(exmaps)):
            ex += exmaps[i]
            ex *= exmaps[i]
    elif fusion_type=='otsu':
        for i in range(1, len(exmaps)):
            ex += exmaps[i]
            ex *= otsu_sigmoid(exmaps[i], T=T)
    return ex
    
def SISE(img, model, class_index, layers, grad_thr, interp='bilinear', 
         fusion_type='otsu', T=100.):

    '''
    For now, this function supports VGG16, ResNet50, and ResNet101.
    img: a 4-D image, or a 3-D array.
    model: the classification model
    layers: list of layers to be visualized either individually or mutually.
    interp: upsampling technique.
    Check the supproted upsampling types in function 'create_attribution_masks'.
	grad_thr: Threshold on the average gradient values to select the most appropriate feature maps.
    fusion_type: the fusion technipue for visualization maps:
        simple: Using only addition and multiplication blocks.
        otsu: Using addition, soft otsu threshold, and multiplication blocks.
    auto_layer_finder: if 'True', the layers are automatically selected. Otherwise,
        pre-defined layers for the models experimented are used.
    pool_input_select: If True, the inputs of pooling layers are detected automatically.
        Otherwise,  the outputs of pooling layers are detected automatically.
        If 'auto_layer_finder=False', this parameter is ineffective.
    '''
    masks = create_attribution_masks(img, model, layers, class_index=class_index, max_mask_num = grad_thr, interp=interp)
    exmaps=[]
    for mask_set in masks:
        exmaps.append(visualize_layers(img[0], model, class_index, mask_set))
    return fuse_visualization_maps(exmaps, fusion_type=fusion_type, T=T)
    
    
def weighted_fusion(w,exmaps, T=100.):
    '''
    Objective: weighted fusion using weighted addition, unweighted multiplication, and otsu threshold blocks.
    inputs:
        w: an array of weight factors of length N-1.
        exmaps: a 3-D array of explanation maps of length H x W x N.
    parameters:
        N: number of visualiation maps received
        H x W: size of visualization maps.
    outputs:
        e_out: fused explanation map.
    '''
    #w_post=np.maximum(w,0)
    w_post=np.clip(a=w, a_min=0, a_max=2)
    e23=np.multiply((exmaps[:,:,0]*w_post[0]+exmaps[:,:,1]*(2-w_post[0])),
                    otsu_sigmoid(exmaps[:,:,1], T=T))
    e234=np.multiply((e23*w_post[1]+exmaps[:,:,2]*(2-w_post[1])),
                    otsu_sigmoid(exmaps[:,:,2], T=T))
    e2345=np.multiply((e234*w_post[2]+exmaps[:,:,3]*(2-w_post[2])),
                    otsu_sigmoid(exmaps[:,:,3], T=T))
    e23456=np.multiply((e2345*w_post[3]+exmaps[:,:,4]*(2-w_post[3])),
                    otsu_sigmoid(exmaps[:,:,4], T=T))
    e_out = e23456
    return e_out

def grad_cam(input_model, image, layer_name):
    cls = np.argmax(input_model.predict(image))
    def normalize(x):
        """Utility function to normalize a tensor by its L2 norm"""
        return (x + 1e-10) / (K.sqrt(K.mean(K.square(x))) + 1e-10)
    """GradCAM method for visualizing input saliency."""
    y_c = input_model.output
    conv_output = input_model.get_layer(layer_name).output
    feedforward1 = keras.models.Model([input_model.input], [conv_output, y_c])
    with tf.GradientTape() as tape:
        ff_results=feedforward1([image])
        all_fmap_masks, predictions = ff_results[0], ff_results[-1]
        loss = predictions[:, cls]
    grads_val = tape.gradient(loss, all_fmap_masks)
    if len(image.shape)==3:
        axis=(0, 1)
    elif len(image.shape)==4:
        axis=(0, 1, 2)
    weights = np.mean(grads_val, axis=axis)
    cam = np.dot(all_fmap_masks[0], weights)
    #print (cam)
    H,W= image.shape[1:3]
    cam = np.maximum(cam, 0)
    #cam = resize(cam, (H, W))
    cam = zoom(cam,H/cam.shape[0])
    #cam = np.maximum(cam, 0)
    cam = cam / cam.max()
    return cam

def grad_cam_plus_plus(input_model, image, layer_name):
    cls = np.argmax(input_model.predict(image))

    def normalize(x):
        """Utility function to normalize a tensor by its L2 norm"""
        return (x + 1e-10) / (K.sqrt(K.mean(K.square(x))) + 1e-10)

    """GradCAM method for visualizing input saliency."""
    y_c = input_model.output
    conv_output = input_model.get_layer(layer_name).output
    feedforward1 = keras.models.Model([input_model.input], [conv_output, y_c])
    with tf.GradientTape() as tape1:
        with tf.GradientTape() as tape2:
            with tf.GradientTape() as tape3:
                ff_results = feedforward1([image])
                all_fmap_masks, predictions = ff_results[0], ff_results[-1]
                loss = predictions[:, cls]
            grads_val1 = tape3.gradient(loss, all_fmap_masks)
        grads_val2=tape2.gradient(grads_val1,all_fmap_masks)
    grads_val3=tape1.gradient(grads_val2,all_fmap_masks)

    if len(image.shape) == 3:
        axis = (0, 1)
    elif len(image.shape) == 4:
        axis = (0, 1, 2)
    alpha= grads_val2/(2.0*grads_val2+grads_val3*np.sum(conv_output,axis=axis))
    alpha=np.where(grads_val1!=0,alpha,0)
    weights = np.maximum(grads_val1,0.0)*alpha
    weights=np.sum(weights,axis=axis)
    cam = np.dot(all_fmap_masks[0], weights)
    # print (cam)
    H, W = image.shape[1:3]
    cam = np.maximum(cam, 0)
    # cam = resize(cam, (H, W))
    cam = zoom(cam, H / cam.shape[0])
    # cam = np.maximum(cam, 0)
    cam = cam / cam.max()
    return cam
	
def RISE(img, model, class_index, N_MASKS=8000, H=224, W=224, C=3):
    '''
	img: a 3-D input image
	model: a trained model
	class_index; The class of interest
	N_MASKS: The number of random masks to be generated
	H,W,C: The desired dimensions of the random masks
	'''
    X = np.zeros(shape=(N_MASKS, H, W, C), dtype=np.float32)
    masks = np.zeros((N_MASKS,H,W), dtype=np.float32)
    #for i in tqdm(range(N_MASKS)):
    for i in range(N_MASKS):
        m =create_random_mask(H=H, W=W)
        masks[i] = m
        x = img.copy()
        x[:, :, 0] *= m
        x[:, :, 1] *= m
        x[:, :, 2] *= m
        X[i] = x
    preds_masked = model.predict(X, verbose=0)
    sum_mask = np.zeros(masks[0].shape, dtype=np.float32)

    # np.einsum???
    for i, mask in enumerate(masks):
        m = mask * preds_masked[i, class_index]
        sum_mask += m

    sum_mask -= np.min(sum_mask)
    sum_mask /= np.max(sum_mask)
    return sum_mask


'''
Copyright 2020 Vignesh Kotteeswaran <iamvk888@gmail.com>
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''

''' Tensorflow version of FullGrad saliency algorithm with gradient completeness check and single
    skeleton to create FullGrad model of different backbones'''




class FullGrad():
    def __init__(self, base_model, num_classes=1000, class_names=None, verbose=False):
        self.base_model = base_model
        self.num_classes = num_classes
        self.model = self.linear_output_model(self.base_model)
        self.verbose = verbose
        assert (self.num_classes > 0), 'Output classes must be greater than 1 but found' + str(self.num_classes)
        self.blockwise_biases = self.getBiases()
        self.check = True
        self.class_names = class_names
        if self.class_names != None:
            assert (len(self.class_names) == self.num_classes), 'Num classes and class names not matched'
        else:
            self.class_names = [None] * self.num_classes

    def linear_output_model(self, model):
        x = Dense(self.num_classes)(model.layers[-2].output)
        temp = Model(model.input, x)
        temp.set_weights(self.base_model.get_weights())
        return temp

    def getBiases(self):
        reset_act = False
        self.biases = []
        for count, layer in enumerate(self.model.layers):
            if count == 0:
                self.biases.append(0)
                x = np.zeros(shape=(1, self.model.layers[0].input_shape[0][1], self.model.layers[0].input_shape[0][2],
                                    self.model.layers[0].input_shape[0][3])).astype(np.float32)
            if isinstance(layer, Conv2D) or isinstance(layer, Dense):
                # print(layer.get_config()['activation'])
                if layer.get_config()['activation'] != 'linear':
                    reset_act = True
                    # old_act=layer.get_config()['activation']
                    old_act = 'relu'
                    layer.activation = Activation('linear')
                if len(layer.input_shape) == 2:
                    x = layer(np.zeros(shape=(1, layer.input_shape[1])).astype(np.float32))
                    self.biases.append(x)
                if len(layer.input_shape) == 4:
                    x = layer(
                        np.zeros(shape=(1, layer.input_shape[1], layer.input_shape[2], layer.input_shape[3])).astype(
                            np.float32))
                    self.biases.append(x)
                if reset_act == True:
                    layer.activation = Activation(old_act)
                    reset_act = False
            if isinstance(layer, BatchNormalization):
                x = layer(np.zeros(shape=(1, layer.input_shape[1], layer.input_shape[2], layer.input_shape[3])).astype(
                    np.float32))
                self.biases.append(x)

        return self.biases

    def getFeatures(self, input):
        self.features = []

        values = [i for i in np.zeros(shape=(len(self.model.layers))).astype(np.float32)]
        keys = [str(i.output.name) for i in self.model.layers]
        data = dict(zip(keys, values))
        # print(data)

        for count, layer in enumerate(self.model.layers):
            if count == 0:
                if not isinstance(input, tf.Tensor):
                    # input=tf.convert_to_tensor(input_)
                    data[layer.output.name] = tf.convert_to_tensor(input)
                self.features.append(input)
            if count == 1:
                # print(layer)

                if isinstance(layer, Dense) or isinstance(layer, Conv2D):

                    # print('dense conv')
                    if layer.get_config()['activation'] != 'linear':
                        # old_act=layer.get_config()['activation']
                        old_act = 'relu'
                        # print(old_act)
                        layer.activation = Activation('linear')
                        x = layer(input)
                        self.features.append(x)
                        data[layer.output.name] = Activation(old_act, name='cut_' + old_act + '_conv' + str(count))(x)

                    if layer.get_config()['activation'] == 'linear':
                        # print(layer.name,layer,layer.input.name,layer.output.name)
                        data[layer.output.name] = layer(input)
                        self.features.append(data[layer.output.name])

                elif isinstance(layer, BatchNormalization):
                    # print('BN')
                    # data[layer.output.name]=layer(input)
                    data[layer.output.name] = layer(data[layer.input.name])
                    self.features.append(data[layer.output.name])

                else:
                    data[layer.output.name] = layer(input)

            if count > 1:
                if isinstance(layer, Add) or isinstance(layer, Concatenate) or isinstance(layer, Multiply):
                    data[layer.output.name] = layer([data[i.name] for i in layer.input])

                elif isinstance(layer, Conv2D):
                    if layer.get_config()['activation'] != 'linear':
                        # old_act=layer.get_config()['activation']
                        old_act = 'relu'
                        # print(old_act)
                        layer.activation = Activation('linear')
                        x = layer(data[layer.input.name])
                        self.features.append(x)
                        data[layer.output.name] = Activation(old_act, name='cut_' + old_act + '_conv' + str(count))(x)

                    else:
                        data[layer.output.name] = layer(data[layer.input.name])
                        self.features.append(data[layer.output.name])


                # data[layer.name]=layer(data[layer.input.name.split('/')[0]])
                elif isinstance(layer, Dense):
                    if layer.get_config()['activation'] != 'linear':
                        # old_act=layer.get_config()['activation']
                        old_act = 'relu'
                        # print(old_act)
                        layer.activation = Activation('linear')
                        x = layer(data[layer.input.name])
                        self.features.append(x)
                        data[layer.output.name] = Activation(old_act, name='cut_' + old_act + '_dense' + str(count))(x)

                    else:
                        data[layer.output.name] = layer(data[layer.input.name])
                        self.features.append(data[layer.output.name])

                # data[layer.name]=layer(data[layer.input.name.split('/')[0]])

                elif isinstance(layer, BatchNormalization):
                    # print(layer.name,layer,layer.input.name,layer.output.name)
                    data[layer.output.name] = layer(data[layer.input.name])
                    self.features.append(data[layer.output.name])

                else:
                    # print(layer)
                    data[layer.output.name] = layer(data[layer.input.name])

        lastname = layer.output.name
        # return data[lastname],self.features
        self.layer_data = data
        # lastname=layer.name
        return data[lastname], self.features

    def fullGradientDecompose(self, image, target_class=None):
        """
        Compute full-gradient decomposition for an image
        """
        out, features = self.getFeatures(image)
        ### out--> imagenet 1000 probs , features--> conv_block_features ####

        if target_class is None:
            target_class = tf.argmax(out, axis=1)

        if self.check:
            check_target_class = K.eval(target_class)[0]
            print('class:', check_target_class, 'class name:', self.class_names[check_target_class])
            self.check = False
        assert (len(features) == len(
            self.blockwise_biases)), 'Number of features {} not equal to number of blockwise biases {}'.format(
            len(features), len(self.blockwise_biases))
        agg = tf.gather_nd(features[-1], [[0, tf.squeeze(tf.argmax(features[-1], axis=1))]])
        gradients = tf.gradients(agg, features)
        # print('gradients:',gradients)

        for grad in gradients:
            if grad == None and self.verbose:
                print(grad)
            if grad != None and self.verbose:
                print('grad:', grad.shape)

        # print(gradients[0])
        # First element in the feature list is the image
        input_gradient = gradients[0] * image

        # Loop through remaining gradients
        bias_gradient = []
        for i in range(1, len(gradients)):
            # print('mul:',gradients[i].shape,)
            # print('mul:',input_grad[0].shape,image.shape,'max:',input_grad[0].max(),image.max())
            if self.verbose:
                print('mul:', gradients[i].shape, self.blockwise_biases[i].shape, 'max:', K.eval(gradients[i]).max(),
                      self.blockwise_biases[i].max())
            bias_gradient.append(gradients[i] * self.blockwise_biases[i])

        return (input_gradient), bias_gradient, out

    def checkCompleteness(self, input):
        self.check = True
        # print('starting completeness test')
        # Random input image
        # input=np.random.randn(1,224,224,3).astype(np.float32)
        input = tf.convert_to_tensor(input.astype(np.float32))

        # Compute full-gradients and add them up

        input_grad, bias_grad, raw_output = self.fullGradientDecompose(input, target_class=None)

        # input_grad=K.eval(input_grad)
        if self.verbose:
            print(input_grad.sum(), input_grad.max())
            # print(K.eval(i).sum() for i in bias_grad)
        # print(input_grad,input)

        fullgradient_sum = tf.reduce_sum(input_grad)
        for i in range(len(bias_grad)):
            if self.verbose:
                print('fullgrad sum:', K.eval(fullgradient_sum), 'biasgrad sum:', K.eval(bias_grad[i]).sum(),
                      K.eval(bias_grad[i]).max(), K.eval(bias_grad[i]).shape)
            fullgradient_sum += tf.reduce_sum(bias_grad[i])

        raw_output = K.eval(raw_output)
        fullgradient_sum = K.eval(fullgradient_sum)

        print('Running completeness test.....')
        print('final_layer_max_class_linear_output:', raw_output.max())
        print('sum of FullGrad:', fullgradient_sum)

        # Compare raw output and full gradient sum
        err_message = "\nThis is due to incorrect computation of bias-gradients.Saliency may not completely represent input&bias gradients, use at your own risk "
        err_string = "Completeness test failed! Raw output = " + str(raw_output.max()) + " Full-gradient sum = " + str(
            fullgradient_sum)
        assert np.isclose(raw_output.max(), fullgradient_sum, atol=0.00001), err_string + err_message
        print('Completeness test passed for FullGrad.')

    def postprocess(self, inputs):
        # Absolute value
        inputs = tf.math.abs(inputs)
        # Rescale operations to ensure gradients lie between 0 and 1
        inputs = inputs - tf.keras.backend.min(inputs)
        inputs = inputs / (tf.keras.backend.max(inputs) + K.epsilon())
        return inputs

    def saliency(self, image, target_class=None):
        # FullGrad saliency
        input_grad, bias_grad, _ = self.fullGradientDecompose(tf.convert_to_tensor(image.astype(np.float32)),
                                                              target_class=target_class)

        # Input-gradient * image
        # print('input_mul:',input_grad[0].shape,image.shape,'max:',input_grad.max(),image.max())
        grd = input_grad * tf.convert_to_tensor(image.astype(np.float32))
        gradient = tf.reduce_sum(self.postprocess(grd), axis=-1)
        ### input grad postprocessed and summed across axis 1 ###
        cam = gradient
        # print(cam.shape)

        # Bias-gradients of conv layers
        for i in range(len(bias_grad)):
            # Checking if bias-gradients are 4d / 3d tensors
            if len(bias_grad[i].shape) == len(image.shape):
                # temp = self.postprocess(bias_grad[i])
                if self.verbose:
                    print(temp.shape, image.shape)
                if len(image.shape) == 4:
                    # gradient=skimage.transform.resize(temp,(temp.shape[0],image.shape[1],image.shape[2],temp.shape[-1]))
                    cam = tf.math.add(cam, tf.reduce_sum(tf.image.resize(self.postprocess(bias_grad[i]), (
                    self.model.layers[0].input_shape[0][1], self.model.layers[0].input_shape[0][2])), axis=-1))
                # print(cam.shape,gradient.shape,gradient.sum(1, keepdim=True).shape)
                # cam += tf.reduce_sum(gradient,axis=-1)
                if self.verbose:
                    print(cam.shape, gradient.shape)

        return K.eval(cam)

    def postprocess_saliency_map(self, saliency_map):
        saliency_map = saliency_map
        saliency_map = saliency_map - saliency_map.min()
        saliency_map = saliency_map / saliency_map.max()
        saliency_map = saliency_map.clip(0, 1)
        saliency_map = saliency_map.squeeze()
        # print(saliency_map.shape)
        saliency_map = np.uint8(saliency_map * 255)
        return saliency_map
