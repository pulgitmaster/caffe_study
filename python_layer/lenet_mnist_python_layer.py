# 3rd party
import os
import cv2
import time
import numpy as np
import argparse

# caffe lib
import caffe
from caffe import layers as L, params as P
from caffe.proto import caffe_pb2

#############################################################
#    Must edit this line as your caffe_study root path      #
#############################################################
caffe_study_root = "/home/yb/Desktop/caffe_study/"
data_dir = os.path.join(caffe_study_root, "data")
lenet_path = os.path.join(caffe_study_root, "python_layer", "lenet_pylayer")

'''
[INFO] : PYTHON LAYER IN CAFFE
You must have a python_param dictionary with at least the module and layer parameters;
    * module refers to the file where you implemented your layer (without the .py);
    * layer refers to the name of your class;
    * You can pass parameters to the layer using param_str (more on accessing them bellow);
'''

def lenet_PythonLayer(phase, list_path, batch_size):
    n = caffe.NetSpec()
    python_layer_params = dict(list_path=list_path, batch_size=batch_size, im_shape=(28, 28), shuffle=True, phase=phase, crop_size=None)

    n.data, n.label = L.Python(module='my_image_data_layer', # python script name
                                layer='MnistImageDataLayer', # python class name
                                ntop=2,
                                param_str=str(python_layer_params)
                                )
    n.conv1 = L.Convolution(n.data, kernel_size=5, num_output=20, weight_filler=dict(type='xavier'))
    n.pool1 = L.Pooling(n.conv1, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.conv2 = L.Convolution(n.pool1, kernel_size=5, num_output=50, weight_filler=dict(type='xavier'))
    n.pool2 = L.Pooling(n.conv2, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.fc1 =   L.InnerProduct(n.pool2, num_output=500, weight_filler=dict(type='xavier'))
    n.relu1 = L.ReLU(n.fc1, in_place=True)
    n.score = L.InnerProduct(n.relu1, num_output=10, weight_filler=dict(type='xavier'))
    n.loss =  L.SoftmaxWithLoss(n.score, n.label)
    n.acc = L.Accuracy(n.score, n.label)

    return n.to_proto()

def lenet_deploy():
    n = caffe.NetSpec()
    # n.data = L.Input(input_param=dict(shape=[dict(dim=[1, 1, 28, 28])]))
    # n.conv1 = L.Convolution(n.data, kernel_size=5, num_output=20, weight_filler=dict(type='xavier'))
    # n.pool1 = L.Pooling(n.conv1, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    # n.conv2 = L.Convolution(n.pool1, kernel_size=5, num_output=50, weight_filler=dict(type='xavier'))
    # n.pool2 = L.Pooling(n.conv2, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    # n.fc1 =   L.InnerProduct(n.pool2, num_output=500, weight_filler=dict(type='xavier'))
    # n.relu1 = L.ReLU(n.fc1, in_place=True)
    # n.score = L.InnerProduct(n.relu1, num_output=10, weight_filler=dict(type='xavier'))
    # n.pred =  L.Softmax(n.score)
    return n.to_proto()

def create_solver(train_net_path, test_net_path, base_lr=0.01):
    s = caffe_pb2.SolverParameter() # generate solver
    # Specify locations of the train and (maybe) test networks.
    s.train_net = train_net_path
    s.test_net.append(test_net_path) # The type of s.test_net is "list"
    s.test_interval = 10  # Test after every 10 training iterations.
    s.test_iter.append(1000) # Test on 100 batches each time we test.

    # The number of iterations over which to average the gradient.
    # Effectively boosts the training batch size by the given factor, without affecting memory utilization.
    s.max_iter = 1000     # # of times to update the net (training iterations)
    # Solve using the stochastic gradient descent (SGD) algorithm, Other choices include 'Adam' and 'RMSProp'.
    s.type = 'SGD'
    s.base_lr = base_lr
    # Set `lr_policy` to define how the learning rate changes during training.
    # Here, we 'step' the learning rate by multiplying it by a factor `gamma`
    # every `stepsize` iterations.
    s.lr_policy = 'inv'
    s.gamma = 0.1
    #s.power = 0.75
    s.stepsize = 20 # Very much similar to pytorch optimizer params.
    # Set other SGD hyperparameters. Setting a non-zero `momentum` takes a
    # weighted average of the current gradient and previous gradients to make
    # learning more stable. L2 weight decay regularizes learning, to help prevent
    # the model from overfitting.
    s.momentum = 0.9
    s.weight_decay = 5e-4
    # Display the current training loss and accuracy every 100 iterations.
    s.display = 10
    # Snapshots are files used to store networks we've trained.  Here, we'll
    # snapshot every 1K iterations -- ten times during training.
    s.snapshot = 1000
    s.snapshot_prefix = os.path.join(lenet_path, 'lenet_mnist_pylayer')
    # Train on the GPU.
    s.solver_mode = caffe_pb2.SolverParameter.GPU
    # Write the solver to a temporary file and return its filename.
    return s

def generate_net(solver_path):
    # with python data layer
    with open(os.path.join(lenet_path,'lenet_pylayer_train.prototxt'), 'w') as f:
        f.write(str(lenet_PythonLayer(phase='train',
                                      list_path=os.path.join(data_dir, 'mnist', 'train_list.txt'),
                                      batch_size=240)))

    with open(os.path.join(lenet_path,'lenet_pylayer_test.prototxt'), 'w') as f:
        f.write(str(lenet_PythonLayer(phase='test',
                                      list_path=os.path.join(data_dir, 'mnist', 'test_list.txt'),
                                      batch_size=240)))

    with open(solver_path, 'w') as f:
        f.write(str(create_solver(train_net_path=os.path.join(lenet_path, 'lenet_pylayer_train.prototxt'),
                                test_net_path=os.path.join(lenet_path, 'lenet_pylayer_test.prototxt'),
                                base_lr=0.001))
                )

def train():
    # preparation
    caffe.set_mode_gpu()
    caffe.set_device(0)
    solver_path = os.path.join(lenet_path, 'lenet_mnist_pylayer_solver.prototxt')

    # 1. generate net
    generate_net(solver_path)

    # 2. load solver
    solver = caffe.get_solver(solver_path)

    # 3. train & test
    solver.solve()

    # 4. save weight
    solver.net.save(os.path.join(lenet_path, 'weights.caffemodel')) # save train result

def deploy(img_path):
    # set mode : cpu --> global usage
    caffe.set_mode_cpu()

    # 1. generate deploy net
    #with open(os.path.join(lenet_path, 'lenet_pylayer_deploy.prototxt'), 'w') as f:
    #    f.write(str(lenet_deploy))

    # 2. load weight parameter
    model = os.path.join(lenet_path, 'lenet_pylayer_deploy.prototxt')
    weights = os.path.join(lenet_path, 'weights.caffemodel')

    net = caffe.Net(model, weights, caffe.TEST)

    # 3. get input data
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img.shape[0] != 28 or img.shape[1] != 28:
        img = cv2.resize(img, dsize=(28, 28), interpolation=cv2.INTER_CUBIC)

    img_data = img.reshape((1, 1, 28, 28)) # (Batch, Channel, Height, Width)

    # 4. inference
    # pass input data as data
    net.blobs['data'].data[...] = img_data
    start = time.time()
    res = net.forward()
    end = time.time()

    pred = np.argmax(res['score'][0])
    print 'pred: ', pred
    #pred = np.argmax(res['loss'][0]) # [0] --> since ther is only one batch!!!!

def get_args():
    parser = argparse.ArgumentParser(description='training lenet with Mnist dataset with custom-python layer',
                                     epilog='Example of use: $python lenet_mnist_python_layer.py -p train',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-p', '--phase', dest='phase', type=str, default='deploy',
                        help='phase: train or deploy, default=deploy')
    return parser.parse_args()

if __name__ == '__main__':
    # get args
    args = get_args()
    if args.phase == 'train':
        # train
        train()
    else:
        # deploy
        deploy(os.path.join(data_dir, 'mnist', 'mnist_test_png', '8', '391.png'))

    
