from pylab import *
import matplotlib.pyplot as plt

import sys
import caffe

import os

# If you don't have mnist lmdb format, then get from here : https://drive.google.com/file/d/1526YI_Nrsr4lMCeea4m1F4eQBVzagMaB/view

from caffe import layers as L, params as P
from caffe.proto import caffe_pb2
from my_image_data_layer import Custom_Data_Layer

# ref : https://github.com/fabiocarrara/pyffe/blob/master/models/resnet.py

def conv_relu(bottom, ksize, nout, stride=1, pad=0, group=1):
    conv = layers.Convolution(bottom, kernel_size=ksize, stride=stride,
                                num_output=nout, pad=pad, group=group)
    return conv, layers.ReLU(conv, in_place=True)

def conv_bn(bottom, ksize=3, nout, stride=1, pad=0, group=1, train=True):
    if train:
        param = [dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)]
    else:
        param = [dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)]

    conv = layers.Convolution(bottom, kernel_size=ksize, stride=stride,
                                num_output=nout, pad=pad, param=param, weight_filler=dict(type='msra'),
                                bias_filler=dict(type='constant'), group=group)
    bn = layers.BatchNorm(conv)
    return conv, bn

def resnet18_mnist():
    n = caffe.NetSpec()
    #n.data, n.label = L.ImageData(image_data_param=dict(source='tmp' , batch_size=100),
    #                               ntop=2, transform_param=dict(mean_file='tmp'))
    n.data, n.label = Custom_Data_Layer(src_file=, batch_size=64, im_shape=(64, 64), ntop=2) # 64 x 64

    #data, label = layers.Python(module='AsyncImageDataLayer', layer='AsyncImageDataLayer', ntop=2, param_str=str(image_data_param))
    #conv1 = 