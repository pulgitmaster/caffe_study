from pylab import *
import matplotlib.pyplot as plt

import sys
import caffe

import os

# If you don't have mnist lmdb format, then get from here : https://drive.google.com/file/d/1526YI_Nrsr4lMCeea4m1F4eQBVzagMaB/view

from caffe import layers as L, params as P
from caffe.proto import caffe_pb2
from my_image_data_layer import Custom_Data_Layer

def max_pool(bottom, ks, stride=1):
    return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=ks, stride=stride)

def conv_factory(bottom, ks, nout, stride=1, pad=0):
    conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
                                num_output=nout, pad=pad, bias_term=False, weight_filler=dict(type='msra'))
    batch_norm = L.BatchNorm(conv, in_place=True, batch_norm_param=dict(moving_average_fraction=0.9))
    scale = L.Scale(batch_norm, bias_term=True, in_place=True)
    relu = L.ReLU(scale, in_place=True)
    return relu

def layer_block(bottom, nout, upper_stride=1):
    conv1 = L.Convolution(bottom, kernel_size=3, stride=upper_stride, num_output=nout, pad=1, weight_filler=dict(type='msra'), bias_term=False)
    batch_norm1 = L.BatchNorm(conv1, in_place=True, batch_norm_param=dict(moving_average_fraction=0.9))
    scale1 = L.Scale(batch_norm1, bias_term=True, in_place=True)
    relu = L.ReLU(scale1, in_place=True)
    conv2 = L.Convolution(relu, kernel_size=3, stride=1, num_output=nout, pad=1, weight_filler=dict(type='msra'), bias_term=False)
    batch_norm2 = L.BatchNorm(conv2, in_place=True, batch_norm_param=dict(moving_average_fraction=0.9))
    scale2 = L.Scale(batch_norm2, bias_term=True, in_place=True)
    return scale2

def part_block(bottom, nout, stride=2):
    conv = L.Convolution(bottom, kernel_size=1, stride=stride, num_output=nout, pad=0, weight_filler=dict(type='msra'), bias_term=False)
    batch_norm = L.BatchNorm(conv, in_place=True, batch_norm_param=dict(moving_average_fraction=0.9))
    scale = L.Scale(batch_norm, bias_term=True, in_place=True)
    return scale

def concat_block(bottom1, bottom2):
    concat = L.Eltwise(bottom1, bottom2, operation=P.Eltwise.SUM)
    relu = L.ReLU(concat, in_place=True)
    return relu

def resnet18_mnist(lmdb, batch_size, train=True):
    n = caffe.NetSpec()
    #n.data, n.label = L.ImageData(image_data_param=dict(source='tmp' , batch_size=100),
    #                               ntop=2, transform_param=dict(mean_file='tmp'))
    #n.data, n.label = Custom_Data_Layer(src_file=, batch_size=64, im_shape=(64, 64), ntop=2) # 64 x 64

    if train:
        n.data, n.label = L.Data(source=lmdb, backend=P.Data.LMDB, batch_size=batch_size, ntop=2,
                transform_param=dict(crop_size=224, mean_value=[104, 117, 123], mirror=True))
    else:
        n.data = L.Layer() # something like place-holder
        n.label = L.Layer()

    output_size = 64
    n.conv_layer = conv_factory(n.data, ks=7, nout=output_size, stride=2, pad=3)
    n.pool = max_pool(n.conv_layer, ks=3, stride=2)

    n.part1 = part_block(n.pool, nout=output_size, stride=1)
    n.layer1 = layer_block(n.pool, nout=output_size, upper_stride=1) # 64
    n.concat1 = concat_block(n.part1, n.layer1)

    n.layer2 = layer_block(n.concat1, nout=output_size, upper_stride=1) # 64
    n.concat2 = concat_block(n.concat1, n.layer2)

    output_size *= 2
    n.part2 = part_block(n.concat2, nout=output_size)
    n.layer3 = layer_block(n.concat2, nout=output_size, upper_stride=2) # 128
    n.concat3 = concat_block(n.part2, n.layer3)

    n.layer4 = layer_block(n.concat3, nout=output_size, upper_stride=1) # 128
    n.concat4 = concat_block(n.concat3, n.layer4)

    output_size *= 2
    n.part3 = part_block(n.concat4, nout=output_size)
    n.layer5 = layer_block(n.concat4, nout=output_size, upper_stride=2) # 256
    n.concat5 = concat_block(n.part3, n.layer5)

    n.layer6 = layer_block(n.concat5, nout=output_size, upper_stride=1) # 256
    n.concat6 = concat_block(n.concat5, n.layer6)

    output_size *= 2
    n.part4 = part_block(n.concat6, nout=output_size)
    n.layer7 = layer_block(n.concat6, nout=output_size, upper_stride=2) # 512
    n.concat7 = concat_block(n.part4, n.layer7)

    n.layer8 = layer_block(n.concat7, nout=output_size, upper_stride=1) # 512
    n.concat8 = concat_block(n.concat7, n.layer8)

    n.pool2 = L.Pooling(n.concat8, pool=P.Pooling.AVE, global_pooling=True)
    n.fc = L.InnerProduct(n.pool2, num_output=1000, weight_filler=dict(type='xavier'))

    if train:
        n.loss = L.SoftmaxWithLoss(n.fc, n.label)
    n.acc = L.Accuracy(n.fc, n.label, include=dict(phase=getattr(caffe_pb2, 'TEST'))) # same as caffe.TEST

    return n.to_proto()

with open('resnet18_sample.prototxt', 'w') as f:
    f.write(str(resnet18_mnist(lmdb='test_lmdb',batch_size=50)))
