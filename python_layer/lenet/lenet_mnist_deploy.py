# 3rd party
import os
import cv2
import time
import numpy as np
# caffe lib
import caffe
from caffe import layers as L, params as P
from caffe.proto import caffe_pb2

#############################################################
#    Must edit this line as your caffe_study root path      #
#############################################################
caffe_study_root = "/home/yb/Desktop/caffe_study/"
lenet_path = os.path.join(caffe_study_root, "python_layer", "lenet")
data_dir = os.path.join(caffe_study_root, "data")

model = os.path.join(lenet_path, 'lenet_deploy.prototxt')
weights = os.path.join(lenet_path, 'weights.caffemodel')

caffe.set_mode_cpu()
caffe.set_device(0)

net = caffe.Net(model, weights, caffe.TEST)


img_path = os.path.join(data_dir, 'mnist', 'mnist_test_png', '7', '97.png')

img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
if img.shape != (28, 28):
    img = cv2.resize(img, dsize=(28, 28), interpolation=cv2.INTER_CUBIC)
img = img.reshape((1, 1, 28, 28))

# pass input data as data
net.blobs['data'].data[...] = img
res = net.forward()
pred = np.argmax(res['loss'][0]) # [0] --> since ther is only one batch!!!!

#print res
print "predicted : ", pred