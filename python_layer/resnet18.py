from __future__ import print_function
from caffe import layers, params, to_proto
from caffe.proto import caffe_pb2

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

def resnet18():
    net = caffe.NetSpec()
    
    data, label = layers.Python(module='AsyncImageDataLayer', layer='AsyncImageDataLayer', ntop=2, param_str=str(image_data_param))
    conv1 = 