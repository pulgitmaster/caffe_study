from pylab import *
import matplotlib.pyplot as plt

import sys
import caffe

import os

from caffe import layers as L, params as P
from caffe.proto import caffe_pb2

#############################################################
#    Must edit this line as your caffe_study root path      #
#############################################################
caffe_study_root = "/home/yb/Desktop/caffe_study/"
data_dir = os.path.join(caffe_study_root, "data")
fer2013_output_path = os.path.join(caffe_study_root, "python_layer", "fer2013")

label_name_list = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
emotion_dict = {'Angry':0, 'Disgust':1, 'Fear':2, 'Happy':3, 'Sad':4, 'Surprise':5, 'Neutral':6}

caffe.set_mode_gpu()
solver = caffe.get_solver(os.path.join(fer2013_output_path, 'resnet18_1batch_solver.prototxt')) # load solver
solver.net.copy_from(os.path.join(fer2013_output_path, 'fer2013_iter_10000.caffemodel')) # load weights
# net = caffe.Net("/home/yb/Desktop/caffe_study/python_layer/fer2013/resnet18_auto_test.prototxt", caffe.TEST)
# net.copy_from(os.path.join(fer2013_output_path, 'fer2013_iter_6000.caffemodel'))

# net.test_nets[0].forward()

fig = plt.figure("test")
num = 16

#output = np.zeros(num)
#solver.test_nets[0].forward(start='conv_layer')
solver.step(1)
#solver.test_nets[0].forward()
#print(solver.test_nets[0].blobs['data'].data)

print(len(solver.test_nets[0].blobs['data'].data))

plt.imshow(solver.test_nets[0].blobs['data'].data[0][0,:,:].reshape(64, 64), cmap='gray')
plt.title('label: '+ label_name_list[int(solver.test_nets[0].blobs['label'].data[0])] + '\n'
                    + 'predicted: ' + label_name_list[int(solver.test_nets[0].blobs['fc'].data.argmax(1)[0])])
# plt.text(0.5, 0.5, 'label: ' + label_name_list[int(solver.test_nets[0].blobs['label'].data[0])])
# plt.text(0.5, 1.5, 'predicted: ' + label_name_list[int(solver.test_nets[0].blobs['fc'].data.argmax(1)[0])])
plt.show()
print(solver.test_nets[0].blobs['label'].data)
print(solver.test_nets[0].blobs['fc'].data.argmax(1))