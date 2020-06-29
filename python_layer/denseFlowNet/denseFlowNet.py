# 3rd party
import cv2
import numpy as np
# caffe lib
import caffe
from caffe import layers as L, params as P
from caffe.proto import caffe_pb2

def conv_relu(bottom, ks, nout, stride=1, pad=0):
    conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
                                num_output=nout, pad=pad, bias_term=False, weight_filler=dict(type='msra'))
    relu = L.ReLU(conv, in_place=True)
    return relu

def block(bottom, ks, nout, dilation=1, stride=1, pad=0):
    conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
                                num_output=nout, pad=pad, dilation=dilation, bias_term=False, weight_filler=dict(type='msra'))
    relu = L.ReLU(conv, in_place=True)
    ccat = L.Concat(relu, bottom, axis=1)
    relu = L.ReLU(ccat, in_place=True)
    return relu

def fc_block(bottom, nout):
    fc = L.InnerProduct(bottom,
                        num_output=nout,
                        param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                        weight_filler=dict(type="gaussian", std=0.005),
                        bias_filler=dict(type="constant", value=1))
    relu = L.ReLU(fc, in_place=True)
    drop = L.Dropout(relu, dropout_ratio=0.5, in_place=True)
    return drop

def denseFlowNet():
    n = caffe.NetSpec()
    n.data = L.Input(input_param=dict(shape=[dict(dim=[1, 3, 150, 150])]),
                     transform_param=dict(mirror=True, crop_size=150, mean_file='/docker/deep_learning_caffe/input/mean.binaryproto'))
    n.conv_relu1 = conv_relu(n.data, 3, 14, 1, 1) # init
    n.block1_add1 = block(n.conv_relu1, 3, 12, 2, 1, 2)
    n.block1_add2 = block(n.block1_add1, 3, 12, 2, 1, 2)
    n.block1_add3 = block(n.block1_add2, 3, 12, 2, 1, 2)
    n.conv_relu2 = conv_relu(n.block1_add3, 3, 50, 1, 1) # trans1
    n.block2_add1 = block(n.conv_relu2, 3, 12, 1, 1, 1)
    n.block2_add2 = block(n.block2_add1, 3, 12, 4, 1, 4)
    n.block2_add3 = block(n.block2_add2, 3, 12, 4, 1, 4)
    n.conv_relu3 = conv_relu(n.block2_add3, 3, 86, 1, 1) # trans2
    n.block3_add1 = block(n.conv_relu3, 3, 12, 8, 1, 8)
    n.block3_add2 = block(n.block3_add1, 3, 12, 8, 1, 8)
    n.block3_add3 = block(n.block3_add2, 3, 12, 8, 1, 8)
    n.conv_relu4 = conv_relu(n.block3_add3, 3, 122, 1, 1) # trans3
    n.block4_add1 = block(n.conv_relu4, 3, 12, 16, 1, 16)
    n.block4_add2 = block(n.block4_add1, 3, 12, 16, 1, 16)
    n.block4_add3 = block(n.block4_add2, 3, 12, 16, 1, 16)
    n.conv_relu5 = conv_relu(n.block4_add3, 3, 158, 1, 1) # trans4
    n.flow1_conv1 = conv_relu(n.conv_relu5, 3, 158, 1, 1)
    n.flow1_conv2 = conv_relu(n.flow1_conv1, 3, 128, 1, 1)
    # fc blocks
    n.fc1_block = fc_block(n.flow1_conv2, 64)
    n.fc2_block = fc_block(n.fc1_block, 64)
    n.fc3 = L.InnerProduct(n.fc2_block,
                            num_output=2,
                            param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                            weight_filler=dict(type="gaussian", std=0.01),
                            bias_filler=dict(type="constant", value=0))

    return n.to_proto()


if __name__ == '__main__':

    with open('denseFlowNet.prototxt', 'w') as f:
        f.write(str(denseFlowNet()))