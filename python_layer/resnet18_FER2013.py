from pylab import *
import matplotlib.pyplot as plt

import sys
import caffe

import os

# If you don't have mnist lmdb format, then get from here : https://drive.google.com/file/d/1526YI_Nrsr4lMCeea4m1F4eQBVzagMaB/view

from caffe import layers as L, params as P
from caffe.proto import caffe_pb2
from my_image_data_layer import Custom_Data_Layer

#############################################################
#    Must edit this line as your caffe_study root path      #
#############################################################
caffe_study_root = "/home/yb/Desktop/caffe_study/"
data_dir = os.path.join(caffe_study_root, "data")
fer2013_output_path = os.path.join(caffe_study_root, "python_layer", "fer2013")

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

def resnet18_fer2013(source, batch_size):
    n = caffe.NetSpec()
    #n.data, n.label = L.ImageData(image_data_param=dict(source='tmp' , batch_size=100),
    #                               ntop=2, transform_param=dict(mean_file='tmp'))
    #n.data, n.label = Custom_Data_Layer(src_file=, batch_size=64, im_shape=(64, 64), ntop=2) # 64 x 64

    n.data, n.label = L.ImageData(batch_size=batch_size,
                            source=source,
                            transform_param=dict(scale=1./255),
                            shuffle=True, # Don't omit this pllllllllllllllllllllllllllz!!!!
                            new_width=64,
                            new_height=64,
                            ntop=2,
                            is_color=True)


    output_size = 64
    n.conv_layer = conv_factory(n.data, ks=3, nout=output_size, stride=1, pad=1)
    #n.pool = max_pool(n.conv_layer, ks=3, stride=2)

    n.part1 = part_block(n.conv_layer, nout=output_size, stride=1)
    n.layer1 = layer_block(n.conv_layer, nout=output_size, upper_stride=1) # 64
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
    n.fc = L.InnerProduct(n.pool2, num_output=7, weight_filler=dict(type='xavier'))

    n.loss = L.SoftmaxWithLoss(n.fc, n.label)
    n.acc = L.Accuracy(n.fc, n.label, include=dict(phase=getattr(caffe_pb2, 'TEST'))) # same as caffe.TEST

    return n.to_proto()

def create_solver(train_net_path, test_net_path=None, base_lr=0.01):
    s = caffe_pb2.SolverParameter() # generate solver
    # Specify locations of the train and (maybe) test networks.
    s.train_net = train_net_path
    if test_net_path is not None:
        s.test_net.append(test_net_path) # The type of s.test_net is "list"
        s.test_interval = 500  # Test after every 1000 training iterations.
        s.test_iter.append(1000) # Test on 100 batches each time we test. # shuffled
    else:
        s.test_initialization = False
    # The number of iterations over which to average the gradient.
    # Effectively boosts the training batch size by the given factor, without affecting memory utilization.
    s.iter_size = 1
    s.max_iter = 5000     # # of times to update the net (training iterations)
    # Solve using the stochastic gradient descent (SGD) algorithm, Other choices include 'Adam' and 'RMSProp'.
    s.type = 'Adam'
    s.base_lr = base_lr
    # Set `lr_policy` to define how the learning rate changes during training.
    # Here, we 'step' the learning rate by multiplying it by a factor `gamma`
    # every `stepsize` iterations.
    s.lr_policy = 'step'
    s.gamma = 0.75 # 3/4 reduced
    #s.power = 0.75
    s.stepsize = 2000 # Very much similar to pytorch optimizer params.
    # Set other SGD hyperparameters. Setting a non-zero `momentum` takes a
    # weighted average of the current gradient and previous gradients to make
    # learning more stable. L2 weight decay regularizes learning, to help prevent
    # the model from overfitting.
    s.momentum = 0.9
    s.weight_decay = 5e-4
    # Display the current training loss and accuracy every 100 iterations.
    s.display = 100
    # Snapshots are files used to store networks we've trained.  Here, we'll
    # snapshot every 1K iterations -- (max_iter / snapshot) times during training.
    s.snapshot = 1000
    s.snapshot_prefix = 'fer2013/fer2013'
    # Train on the GPU.
    s.solver_mode = caffe_pb2.SolverParameter.GPU
    # Write the solver to a temporary file and return its filename.
    return s

def generate_net():
    # with imagedata
    with open(os.path.join(fer2013_output_path,'resnet18_auto_train.prototxt'), 'w') as f:
        f.write(str(resnet18_fer2013(os.path.join(data_dir, 'fer2013', 'train_list.txt'), 80)))
        
    with open(os.path.join(fer2013_output_path,'resnet18_auto_test.prototxt'), 'w') as f:
        f.write(str(resnet18_fer2013(os.path.join(data_dir, 'fer2013', 'test_list.txt'), 16)))

if not os.path.exists(fer2013_output_path):
    os.makedirs(fer2013_output_path)

# 1. generate network as ".prototxt"
generate_net()

# set caffe as gpu
caffe.set_mode_gpu()

# 2. generate solver
solver_path = os.path.join(fer2013_output_path, 'resnet18_auto_solver.prototxt')
with open(solver_path, 'w') as f:
    f.write(str(create_solver(train_net_path=os.path.join(fer2013_output_path,'resnet18_auto_train.prototxt'),
                            test_net_path=os.path.join(fer2013_output_path,'resnet18_auto_test.prototxt'),
                            base_lr=0.005))
            )

# 3. load solver.prototxt ( you must generate ".prototxt" first! )
solver = caffe.get_solver(solver_path)

# 4. train / test
solver.solve()
