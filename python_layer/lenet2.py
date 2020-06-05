from pylab import *
import matplotlib.pyplot as plt

import sys
import caffe

import os

# If you don't have mnist lmdb format, then get from here : https://drive.google.com/file/d/1526YI_Nrsr4lMCeea4m1F4eQBVzagMaB/view

from caffe import layers as L, params as P
from caffe.proto import caffe_pb2
import tempfile

#############################################################
#    Must edit this line as your caffe_study root path      #
#############################################################
caffe_study_root = "/home/yb/Desktop/caffe_study/"
data_dir = os.path.join(caffe_study_root, "data")
lenet2_path = os.path.join(caffe_study_root, "python_layer", "lenet2")

plt.ion()

def lenet_train(train_list_path, batch_size=1):
    n = caffe.NetSpec()
    n.data, n.label = L.ImageData(batch_size=batch_size,
                                source=train_list_path,
                                include={'phase':caffe.TRAIN},
                                transform_param=dict(scale=1./255),
                                shuffle=True,
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

def lenet_test(test_list_path, batch_size=1): # num_classes : for mnist, whether label=None
    n = caffe.NetSpec()
    n.data, n.label = L.ImageData(batch_size=batch_size,
                                source=test_list_path,
                                include={'phase':caffe.TEST},
                                transform_param=dict(scale=1./255),
                                shuffle=True,
                                ntop=2,
                                is_color=False)
    frozen_param = [dict(lr_mult=0)] * 2

    n.conv1 = L.Convolution(n.data, kernel_size=5, num_output=20, param=frozen_param)
    n.pool1 = L.Pooling(n.conv1, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.conv2 = L.Convolution(n.pool1, kernel_size=5, num_output=50, param=frozen_param)
    n.pool2 = L.Pooling(n.conv2, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.fc1 =   L.InnerProduct(n.pool2, num_output=500, param=frozen_param)
    n.relu1 = L.ReLU(n.fc1, in_place=True)
    n.score = L.InnerProduct(n.relu1, num_output=10, param=frozen_param)
    n.loss =  L.SoftmaxWithLoss(n.score, n.label)
    n.acc = L.Accuracy(n.score, n.label)

    return n.to_proto()

def create_solver(train_net_path, test_net_path=None, base_lr=0.01):
    s = caffe_pb2.SolverParameter() # generate solver
    # Specify locations of the train and (maybe) test networks.
    s.train_net = train_net_path
    if test_net_path is not None:
        s.test_net.append(test_net_path) # The type of s.test_net is "list"
        s.test_interval = 500  # Test after every 500 training iterations.
        s.test_iter.append(100) # Test on 100 batches each time we test.
    else:
        s.test_initialization = False
    # The number of iterations over which to average the gradient.
    # Effectively boosts the training batch size by the given factor, without affecting memory utilization.
    s.iter_size = 1
    s.max_iter = 100     # # of times to update the net (training iterations)
    # Solve using the stochastic gradient descent (SGD) algorithm, Other choices include 'Adam' and 'RMSProp'.
    s.type = 'Adam'
    s.base_lr = base_lr
    # Set `lr_policy` to define how the learning rate changes during training.
    # Here, we 'step' the learning rate by multiplying it by a factor `gamma`
    # every `stepsize` iterations.
    s.lr_policy = 'step'
    s.gamma = 0.5
    #s.power = 0.75
    s.stepsize = 1000 # Very much similar to pytorch optimizer params.
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
    s.snapshot_prefix = 'lenet2/lenet'
    # Train on the GPU.
    s.solver_mode = caffe_pb2.SolverParameter.GPU
    # Write the solver to a temporary file and return its filename.
    return s

# when you run more than one solvers
def run_solvers(solvers, niter, disp_interval=10):
    """Run solvers for niter iterations,
       returning the loss and accuracy recorded each iteration.
       `solvers` is a list of (name, solver) tuples."""
    blobs = ('loss', 'acc')
    loss, acc = ({name: np.zeros(niter) for name, _ in solvers}
                 for _ in blobs)
    for it in range(niter):
        for name, s in solvers:
            s.step(1)  # run a single SGD step in Caffe
            loss[name][it], acc[name][it] = (s.net.blobs[b].data.copy()
                                             for b in blobs)
        if it % disp_interval == 0 or it + 1 == niter:
            loss_disp = '; '.join('%s: loss=%.3f, acc=%2d%%' %
                                  (n, loss[n][it], np.round(100*acc[n][it]))
                                  for n, _ in solvers)
            print '%3d) %s' % (it, loss_disp)     
    # Save the learned weights from both nets.
    weight_dir = tempfile.mkdtemp()
    weights = {}
    for name, s in solvers:
        filename = 'weights.%s.caffemodel' % name
        weights[name] = os.path.join(weight_dir, filename)
        s.net.save(weights[name])
    return loss, acc, weights

# when you run more than one solvers
def run_solver(solver, niter, disp_interval=10, test_interval=25, weight_path=None):
    blobs = ('loss', 'acc')
    train_loss, train_acc = (np.zeros(niter), np.zeros(niter))
    test_acc = np.zeros(niter)
    for it in range(niter): # it = epoch
        solver.step(1) # run a single SGD step in caffe
        train_loss[it], train_acc[it] = (solver.net.blobs[b].data.copy() for b in blobs)
        
        if it % disp_interval == 0 or it + 1 == niter:
            loss_disp = 'loss=%.3f, acc=%2d%%' % (train_loss[it], np.round(100*train_acc[it])) # f'loss={loss[it]:.3f}, acc={acc[it]:2d}' <--- python 3.x
            print '%3d) %s' % (it, loss_disp)

        if it != 0: test_acc[it] = test_acc[it-1]
        if it % test_interval == 0 or it + 1 == niter:
            print 'Iteration', it, 'testing...'
            correct = 0
            for test_it in range(100): # test for 100 batch
                solver.test_nets[0].forward()
                correct += sum(solver.test_nets[0].blobs['score'].data.argmax(1)
                            == solver.test_nets[0].blobs['label'].data)
            test_acc[it] = correct / 1e4
            print ('test_acc =', correct / 1e4)

    # Save the learned weights from net.
    weight = solver.net.save(os.path.join(weight_path, 'weights.caffemodel'))
    return train_loss, test_acc, weight

#print(os.path.join(data_dir,'train_list.txt'))
#exit()
# img_train_list, label_train_list = L.ImageData(batch_size=1, source=os.path.join(data_dir,'train_list.txt'), ntop=2)
# print(len(img_train_list.blobs))

if not os.path.exists("lenet2"):
    os.makedirs("lenet2")

# with open('lenet2/lenet_auto_train.prototxt', 'w') as f:
#     f.write(str(lenet('../data/mnist/mnist_train_lmdb', 64)))
    
# with open('lenet2/lenet_auto_test.prototxt', 'w') as f:
#     f.write(str(lenet('../data/mnist/mnist_test_lmdb', 100)))

with open(os.path.join(lenet2_path, 'lenet_train_with_ImageData.prototxt'), 'w') as f:
    f.write(str(lenet_train(os.path.join(data_dir, 'mnist', 'train_list.txt'), 64)))
with open(os.path.join(lenet2_path, 'lenet_test_with_ImageData.prototxt'), 'w') as f:
    f.write(str(lenet_test(os.path.join(data_dir,'mnist', 'test_list.txt'), 100)))

caffe.set_mode_gpu()

### create solver --- old
#solver_path = 'lenet2/lenet_auto_solver.prototxt'
# with open(solver_path, 'w') as f:
#     f.write(str(create_solver(train_net_path='lenet2/lenet_auto_train.prototxt',
#                             test_net_path='lenet2/lenet_auto_test.prototxt',
#                             base_lr=0.001))
#             )
### create solver --- new
solver_path = os.path.join(lenet2_path, 'lenet_auto_solver_with_ImageData.prototxt')
# first generate prototxt
with open(solver_path, 'w') as f:
    f.write(str(create_solver(train_net_path=os.path.join(lenet2_path, 'lenet_train_with_ImageData.prototxt'),
                            test_net_path=os.path.join(lenet2_path, 'lenet_test_with_ImageData.prototxt'),
                            base_lr=0.01))
            )

# load solver.prototxt ( you must generate ".prototxt" first! FUCK CAFFE )
solver = caffe.get_solver(solver_path)
#solver.solve() # train and test immediatly # we will run step by step instead.

# run solver : A.K.A train this network(lenet),, and save '.caffemodel' file(=weight parameters)
train_loss, test_acc, weights = run_solver(solver, 200, 10, 25, 'lenet2/')
solver.save(os.path.join(lenet2_path, 'weights.caffemodel')) # save train result)

plt.plot(np.vstack([train_loss, test_acc]).T)
plt.xlabel('Iteration')
plt.ylabel('Train Loss & Test Accuracy')

plt.ioff()
plt.show()