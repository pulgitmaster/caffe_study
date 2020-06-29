import os
import random
import numpy as np
import caffe
import cv2

class MnistImageDataLayer(caffe.Layer):
    """
    This is a simple synchronous datalayer for training lenet with mnist.
    """
    def setup(self, bottom, top):
        self.top_names = ['data', 'label']
        # params is a python dictionary with layer parameters.
        params = eval(self.param_str)
        # store input as class variables
        self.batch_size = params['batch_size']
        self.im_shape = params['"im_shape'] # expect tuple (m, n)
        self.shuffle = params['shuffle']
        self.phase = params['phase']
        self.crop_size = params['crop_size']
        self.img_list = []
        self.label_list = []

        # list of train/test images set & labels.
        with open(params['list_path'], 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.split()
                self.img_list.append(line[0])
                self.label_list.append(line[1])

        #self.label_list = np.array(self.label_list).astype(np.float)
        self._cur = 0  # current image

    def load_next_image(self):
        """
        Load the next image in a batch.
        """
        # shuffle
        if self.shuffle and self.phase == 'train':
            if self._cur == len(self.img_list):
                self._cur = 0
                img_label_zip = list(zip(self.img_list, self.label_list))
                random.shuffle(img_label_zip)
                self.img_list, self.label_list = zip(*img_label_zip)
                self.img_list = list(self.img_list)
                self.label_list = list(self.label_list)
        else:
            if self._cur == len(self.img_list):
                self._cur = 0

        # Load an image & label
        image_file_name = self.img_list[self._cur]
        im = cv2.imread(image_file_name, cv2.IMREAD_GRAYSCALE)
        label = self.label_list[self._cur]

        self._cur += 1
        return im, label

    def forward(self, bottom, top):
        for itt in range(self.batch_size):
            # Use the batch loader to load the next image.
            im, label = self.load_next_image()
            # Add directly to the top blob
            top[0].data[itt, ...] = np.float32(im)
            top[1].data[itt, ...] = np.float32(label)

    def reshape(self, bottom, top):
        # img
        if self.crop_size:
            top[0].reshape(self.batch_size, 1, self.crop_size, self.crop_size) # 2nd param : channel
        else:
            top[0].reshape(self.batch_size, 1, self.im_shape[0], self.im_shape[1])
        # label
        top[1].reshape(self.batch_size, 1)

    def backward(self, bottom, top):
        """
        This layer does not back propagate
        """
        pass


class CustomDataLayer(caffe.Layer):
    def setup(self, bottom, top):
        #Check top shape
        if len(top) != 2:
            raise Exception("Need to define top blobs (data and label)")
        
        #Check bottom shape
        # no "bottom"s for input layer
        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")
        
        #Read parameters
        params = eval(self.param_str)
        src_file = params["src_file"]
        self.batch_size = params["batch_size"]
        self.im_shape = params["im_shape"] # expect tuple (m, n)
        self.crop_size = params.get("crop_size", False)

        ###### Reshape top ######
        #This could also be done in Reshape method, but since it is a one-time-only
        #adjustment, we decided to do it on Setup
        if self.crop_size:
            top[0].reshape(self.batch_size, 3, self.crop_size, self.crop_size)
        else:
            top[0].reshape(self.batch_size, 3, self.im_shape[0], self.im_shape[1])

        # Note the 7 channels (because FER2013 has 7 classes.)
        top[1].reshape(self.batch_size, 7)

        #Read source file
        #I'm just assuming we have this method that reads the source file
        #and returns a list of tuples in the form of (img, label)

        #self.imgTuples = readSrcFile(src_file) 
        
        self._cur = 0 #use this to check if we need to restart the list of imgs
        
    def forward(self, bottom, top):
        for itt in range(self.batch_size):
            # Use the batch loader to load the next image.
            im, label = self.load_next_image()
            
            #Here we could preprocess the image
            # ...
            
            # Add directly to the top blob
            top[0].data[itt, ...] = im
            top[1].data[itt, ...] = label
    
    def load_next_img(self):
        #If we have finished forwarding all images, then an epoch has finished
        #and it is time to start a new one
        if self._cur == len(self.imgTuples):
            self._cur = 0
            shuffle(self.imgTuples)
        
        im, label = self.imgTuples[self._cur]
        self._cur += 1
        
        return im, label
    
    def reshape(self, bottom, top):
        """
        There is no need to reshape the data, since the input is of fixed size
        (img shape and batch size)
        
        If we were processing a fixed-sized number of images (for example in Testing)
        and their number wasn't a  multiple of the batch size, we would need to
        reshape the top blob for a smaller batch.
        """
        pass

    def backward(self, bottom, top):
        """
        This layer does not back propagate
        """
        pass