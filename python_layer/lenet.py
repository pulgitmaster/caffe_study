from pylab import *
import matplotlib.pyplot as plt

import sys
import caffe

import os

# If you don't have mnist lmdb format, then get from here : https://drive.google.com/file/d/1526YI_Nrsr4lMCeea4m1F4eQBVzagMaB/view

from caffe import layers as L, params as P
from caffe.proto import caffe_pb2

#############################################################
#    Must edit this line as your caffe_study root path      #
#############################################################
caffe_study_root = "/home/yb/Desktop/caffe_study/"
data_dir = os.path.join(caffe_study_root, "data")
lenet_path = os.path.join(caffe_study_root, "python_layer", "lenet")

plt.ion()

def lenet_lmdb(lmdb, batch_size):
    # our version of LeNet: a series of linear and simple nonlinear transformations
    n = caffe.NetSpec()
    
    n.data, n.label = L.Data(batch_size=batch_size, backend=P.Data.LMDB, source=lmdb,
                             transform_param=dict(scale=1./255), ntop=2)
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

def lenet_imagedata(source, batch_size):
    n = caffe.NetSpec()

    n.data, n.label = L.ImageData(batch_size=batch_size,
                                source=source,
                                transform_param=dict(scale=1./255),
                                shuffle=True, # Don't omit this pllllllllllllllllllllllllllz!!!!
                                ntop=2,
                                is_color=False)
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

def write_solver(path):
    solver_txt = (# The train/test net protocol buffer definition
                  "train_net: \"lenet/lenet_auto_train.prototxt\"\n"
                  "test_net: \"lenet/lenet_auto_test.prototxt\"\n"
                # test_iter specifies how many forward passes the test should carry out.
                # In the case of MNIST, we have test batch size 100 and 100 test iterations,
                # covering the full 10,000 testing images.
                  "test_iter: 100\n"
                # Carry out testing every 500 training iterations.
                  "test_interval: 500\n"
                # The base learning rate, momentum and the weight decay of the network.
                  "base_lr: 0.01\n"
                  "momentum: 0.9\n"
                  "weight_decay: 0.0005\n"
                # The learning rate policy
                  "lr_policy: \"inv\""
                  "gamma: 0.0001\n"
                  "power: 0.75\n"
                # Display every 10 iterations
                  "display: 10\n"
                # The maximum number of iterations
                  "max_iter: 10000\n"
                # snapshot intermediate results
                  "snapshot: 5000\n"
                  "snapshot_prefix: \"lenet/lenet\""
                  )
    with open(path, 'w') as f:
        f.write(solver_txt)

if not os.path.exists("lenet"):
    os.makedirs("lenet")

def generate_net(data_type='ImageData'):
    if type(data_type) is not str:
        print("data_type must be string")
        print(type(data_type))
        exit()

    if data_type.lower() == 'lmdb':
        # with lmdb data
        with open(os.path.join(lenet_path, 'lenet_auto_train.prototxt'), 'w') as f:
            f.write(str(lenet_lmdb(os.path.join(data_dir, "mnist", "mnist_train_lmdb"), 64)))

        with open(os.path.join(lenet_path, 'lenet_auto_test.prototxt'), 'w') as f:
            f.write(str(lenet_lmdb(os.path.join(data_dir, "mnist", "mnist_test_lmdb"), 100)))

    elif data_type.lower() == 'imagedata':
        # with imagedata
        with open(os.path.join(lenet_path,'lenet_auto_train.prototxt'), 'w') as f:
            f.write(str(lenet_imagedata(os.path.join(data_dir, 'train_list.txt'), 64)))
            
        with open(os.path.join(lenet_path,'lenet_auto_test.prototxt'), 'w') as f:
            f.write(str(lenet_imagedata(os.path.join(data_dir, 'test_list.txt'), 100)))
    else:
        print("unknown data_type")
        exit()

# 1. generate network as ".prototxt"
generate_net(data_type='ImageData')

# 2. set caffe as gpu
caffe.set_mode_gpu()

# 3. load the solver and create train and test nets
write_solver(os.path.join(lenet_path, 'lenet_auto_solver.prototxt')) # write prototxt first! always!!!
solver = caffe.SGDSolver(os.path.join(lenet_path,'lenet_auto_solver.prototxt'))

# each output is (batch size, feature dim, spatial dim)
print('output size of net:',[(k, v.data.shape) for k, v in solver.net.blobs.items()])

# just print the weight sizes (we'll omit the biases)
print([(k, v[0].data.shape) for k, v in solver.net.params.items()])

# 4. train/test network
solver.net.forward()  # train net
solver.net.save(os.path.join(lenet_path, 'weights.caffemodel')) # save train result

solver.net.copy_from(os.path.join(lenet_path, 'weights.caffemodel')) # load from saved weight parmeters, actually unnecessary in this code(already loaded)
solver.test_nets[0].forward()  # test net (there can be more than one) # type(tes_nets) = list (cautions!!)

# 5. show some demo results
fig = plt.figure("train")
plt.imshow(solver.net.blobs['data'].data[:8, 0].transpose(1, 0, 2).reshape(28, 8*28), cmap='gray'); axis('off')
plt.title('predicted(trained): ' + str(solver.net.blobs['label'].data[:8]))
plt.draw()
plt.pause(0.001)
print('train labels:', solver.net.blobs['label'].data[:8])

fig = plt.figure("test")
plt.imshow(solver.test_nets[0].blobs['data'].data[:8, 0].transpose(1, 0, 2).reshape(28, 8*28), cmap='gray'); axis('off')
plt.title('predicted(test): ' + str(solver.test_nets[0].blobs['label'].data[:8]))
plt.draw()
plt.pause(0.001)
print('test labels:', solver.test_nets[0].blobs['label'].data[:8])

niter = 200
test_interval = 25

# losses will also be stored in the log
train_loss = zeros(niter)
test_acc = zeros(int(np.ceil(niter / test_interval)))
output = zeros((niter, 8, 10))

# the main solver loop
for it in range(niter):
    solver.step(1)  # SGD by Caffe
    
    # store the train loss
    train_loss[it] = solver.net.blobs['loss'].data
    
    # store the output on the first test batch
    # (start the forward pass at conv1 to avoid loading new data)
    solver.test_nets[0].forward(start='conv1')
    output[it] = solver.test_nets[0].blobs['score'].data[:8]
    
    # run a full test every so often
    # (Caffe can also do this for us and write to a log, but we show here
    #  how to do it directly in Python, where more complicated things are easier.)
    if it % test_interval == 0:
        print 'Iteration', it, 'testing...'
        correct = 0
        for test_it in range(100):
            solver.test_nets[0].forward()
            correct += sum(solver.test_nets[0].blobs['score'].data.argmax(1)
                           == solver.test_nets[0].blobs['label'].data)
        test_acc[it // test_interval] = correct / 1e4

_, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(arange(niter), train_loss)
ax2.plot(test_interval * arange(len(test_acc)), test_acc, 'r')
ax1.set_xlabel('iteration')
ax1.set_ylabel('train loss')
ax2.set_ylabel('test accuracy')
ax2.set_title('Test Accuracy: {:.2f}'.format(test_acc[-1]))
plt.ioff()
plt.show()
