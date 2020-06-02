from __future__ import print_function
from caffe import layers, params, to_proto
from caffe.proto import caffe_pb2

# helper function for common structures

def conv_relu(bottom, ksize, nout, stride=1, pad=0, group=1):
    conv = layers.Convolution(bottom, kernel_size=ksize, stride=stride,
                                num_output=nout, pad=pad, group=group)
    return conv, layers.ReLU(conv, in_place=True)

def fc_relu(bottom, nout):
    fc = layers.InnerProduct(bottom, num_output=nout)
    return fc, layers.ReLU(fc, in_place=True)

def max_pool(bottom, ks, stride=1):
    return layers.Pooling(bottom, pool=params.Pooling.MAX, kernel_size=ks, stride=stride)

def caffenet(lmdb, batch_size=256, include_acc=False):
    data, label = layers.Data(source=lmdb, backend=params.Data.LMDB, batch_size=batch_size, ntop=2,
        transform_param=dict(crop_size=227, mean_value=[104, 117, 123], mirror=True))

    # the net itself
    conv1, relu1 = conv_relu(data, 11, 96, stride=4)
    pool1 = max_pool(relu1, 3, stride=2)
    norm1 = layers.LRN(pool1, local_size=5, alpha=1e-4, beta=0.75)
    conv2, relu2 = conv_relu(norm1, 5, 256, pad=2, group=2)
    pool2 = max_pool(relu2, 3, stride=2)
    norm2 = layers.LRN(pool2, local_size=5, alpha=1e-4, beta=0.75)
    conv3, relu3 = conv_relu(norm2, 3, 384, pad=1)
    conv4, relu4 = conv_relu(relu3, 3, 384, pad=1, group=2)
    conv5, relu5 = conv_relu(relu4, 3, 256, pad=1, group=2)
    pool5 = max_pool(relu5, 3, stride=2)
    fc6, relu6 = fc_relu(pool5, 4096)
    drop6 = layers.Dropout(relu6, in_place=True)
    fc7, relu7 = fc_relu(drop6, 4096)
    drop7 = layers.Dropout(relu7, in_place=True)
    fc8 = layers.InnerProduct(drop7, num_output=1000)
    loss = layers.SoftmaxWithLoss(fc8, label)

    if include_acc:
        acc = layers.Accuracy(fc8, label)
        return to_proto(loss, acc)
    else:
        return to_proto(loss)

def make_net():
    with open('train.prototxt', 'w') as f:
        print(caffenet('/path/to/caffe-train-lmdb'), file=f)

    with open('test.prototxt', 'w') as f:
        print(caffenet('/path/to/caffe-val-lmdb', batch_size=50, include_acc=True), file=f)

if __name__ == '__main__':
    make_net()
