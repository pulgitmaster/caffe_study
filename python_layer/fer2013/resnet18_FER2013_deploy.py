# 3rd party
import os
import cv2
import time
import numpy as np
import random
# caffe lib
import caffe
from caffe import layers as L, params as P
from caffe.proto import caffe_pb2

label_name_list = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

def gray_to_rgb(img, dtype=np.uint8):
    return np.asarray(np.dstack((img, img, img)), dtype=dtype)

#############################################################
#    Must edit this line as your caffe_study root path      #
#############################################################
caffe_study_root = "/home/yb/Desktop/caffe_study/"
data_dir = os.path.join(caffe_study_root, "data")
fer2013_output_path = os.path.join(caffe_study_root, "python_layer", "fer2013")
model = os.path.join(fer2013_output_path, 'resnet18_deploy.prototxt')
weights = os.path.join(fer2013_output_path, 'fer2013_iter_9000.caffemodel')

caffe.set_mode_cpu()
caffe.set_device(0)

net = caffe.Net(model, weights, caffe.TEST)

# one of 7178
rdn = random.randint(0, 7178) # 0 ~ 7178
with open(os.path.join(data_dir, 'fer2013/test_list.txt'), 'r') as f:
    for i in range(7178):
        if i != rdn:
            f.readline()
            continue
        txt = f.readline()[:-3]
        print txt
        break
img_path = txt

img = cv2.imread(img_path)

if len(img.shape) != 3:
    img_data = gray_to_rgb(img)
else:
    img_data = img

img = cv2.resize(img, dsize=(64, 64), interpolation=cv2.INTER_CUBIC)
#cv2.imshow("img", img)
img_data = img.reshape((1, 3, 64, 64)) # (Batch, Channel, Height, Width)

# pass input data as data
net.blobs['data'].data[...] = img_data
start = time.time()
res = net.forward()
pred = np.argmax(res['loss'][0]) # [0] --> since ther is only one batch!!!!
end = time.time()

#print res
print "time labs:", (end-start)
print "predicted : ", label_name_list[pred]
img = cv2.resize(img, dsize=(256, 256), interpolation=cv2.INTER_CUBIC) # for visualizing
cv2.putText(img, label_name_list[pred], (80, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
cv2.imshow("img", img)
cv2.waitKey(0)