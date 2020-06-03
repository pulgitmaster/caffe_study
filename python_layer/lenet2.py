from pylab import *
import matplotlib.pyplot as plt

import sys
import caffe

import os

# If you don't have mnist lmdb format, then get from here : https://drive.google.com/file/d/1526YI_Nrsr4lMCeea4m1F4eQBVzagMaB/view

from caffe import layers as L, params as P
from caffe.proto import caffe_pb2

plt.ion()

def lenet(lmdb, batch_size):
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
                # Display every 100 iterations
                  "display: 100\n"
                # The maximum number of iterations
                  "max_iter: 10000\n"
                # snapshot intermediate results
                  "snapshot: 5000\n"
                  "snapshot_prefix: \"lenet/lenet\""
                  )
    with open(path, 'w') as f:
        f.write(solver_txt)

def create_solver(train_net_path, test_net_path=None, base_lr=0.001):
    s = caffe_pb2.SolverParameter() # generate solver
    # Specify locations of the train and (maybe) test networks.
    s.train_net = train_net_path
    if test_net_path is not None:
        s.test_net.append(test_net_path) # The type of s.test_net is "list"
        s.test_interval = 500  # Test after every 500 training iterations.
        s.test_iter.append(100) # Test on 100 batches each time we test.
    # The number of iterations over which to average the gradient.
    # Effectively boosts the training batch size by the given factor, without affecting memory utilization.
    s.iter_size = 1
    s.max_iter = 10000     # # of times to update the net (training iterations)
    # Solve using the stochastic gradient descent (SGD) algorithm, Other choices include 'Adam' and 'RMSProp'.
    s.type = 'Adam'
    s.base_lr = base_lr
    # Set `lr_policy` to define how the learning rate changes during training.
    # Here, we 'step' the learning rate by multiplying it by a factor `gamma`
    # every `stepsize` iterations.
    s.lr_policy = 'step'
    s.gamma = 0.1
    s.stepsize = 1000 # Very much similar to pytorch optimizer params.
    # Set other SGD hyperparameters. Setting a non-zero `momentum` takes a
    # weighted average of the current gradient and previous gradients to make
    # learning more stable. L2 weight decay regularizes learning, to help prevent
    # the model from overfitting.
    s.momentum = 0.9
    s.weight_decay = 5e-4
    # Display the current training loss and accuracy every 100 iterations.
    s.display = 100
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
def run_solver(solver, niter, disp_interval=10, weight_path=None):
    blobs = ('loss', 'acc')
    loss, acc = (np.zeros(niter), np.zeros(niter))
    for it in range(niter): # it = epoch
        solver.step(1) # run a single SGD step in caffe
        loss[it], acc[it] = (solver.net.blobs[b].data.copy() for b in blobs)
        
        if it % disp_interval == 0 or it + 1 == niter:
            loss_disp = 'loss=%.3f, acc=%2d%%' % (loss[it], np.round(100*acc[it])) # f'loss={loss[it]:.3f}, acc={acc[it]:2d}' <--- python 3.x
            print '%3d) %s' % (it, loss_disp)
    # Save the learned weights from net.
    weight = solver.net.save(os.path.join(weight_path, 'weights.caffemodel'))
    return loss, acc, weight
    

if not os.path.exists("lenet2"):
    os.makedirs("lenet2")

with open('lenet2/lenet_auto_train.prototxt', 'w') as f:
    f.write(str(lenet('../data/mnist/mnist_train_lmdb', 64)))
    
with open('lenet2/lenet_auto_test.prototxt', 'w') as f:
    f.write(str(lenet('../data/mnist/mnist_test_lmdb', 100)))

caffe.set_mode_gpu()

### create solver
solver_path = 'lenet2/lenet_auto_solver.prototxt'
with open(solver_path, 'w') as f:
    f.write(str(create_solver(train_net_path='lenet2/lenet_auto_train.prototxt',
                            test_net_path='lenet2/lenet_auto_test.prototxt',
                            base_lr=0.001))
            )

# load solver
solver = caffe.get_solver(solver_path)
# solver.solve() # train # we will run step by step instead.

# run solver : A.K.A train this network(lenet)
train_loss, train_acc, weights = run_solver(solver, 200, 10, 'lenet2/')

plt.plot(np.vstack([train_loss, train_acc]).T)
plt.xlabel('Iteration')
plt.ylabel('Loss & Accuracy')

plt.ioff()
plt.show()

#def test_net(weights):
    #test_net = 

'''
# each output is (batch size, feature dim, spatial dim)
print([(k, v.data.shape) for k, v in solver.net.blobs.items()])

# just print the weight sizes (we'll omit the biases)
print([(k, v[0].data.shape) for k, v in solver.net.params.items()])

solver.net.forward()  # train net
solver.test_nets[0].forward()  # test net (there can be more than one)

# we use a little trick to tile the first eight images
plt.imshow(solver.net.blobs['data'].data[:8, 0].transpose(1, 0, 2).reshape(28, 8*28), cmap='gray'); axis('off')
plt.draw()
plt.pause(0.001)
print 'train labels:', solver.net.blobs['label'].data[:8]

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
'''
