import caffe

class Custom_Data_Layer(caffe.Layer):
    def setup(self, bottom, top):
        # Check top shape
        if len(top) != 2:
                raise Exception("Need to define top blobs (data and label)")
        
        #Check bottom shape
        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")
        
        #Read parameters
        params = eval(self.param_str)
        src_file = params["src_file"]
        self.batch_size = params["batch_size"]
        self.im_shape = params["im_shape"]
        self.crop_size = params.get("crop_size", False)
        
        ###### Reshape top ######
        #This could also be done in Reshape method, but since it is a one-time-only
        #adjustment, we decided to do it on Setup
        if self.crop_size:
            top[0].reshape(self.batch_size, 3, self.crop_size, self.crop_size)
        else:
            top[0].reshape(self.batch_size, 3, self.im_shape, self.im_shape)
            
        top[1].reshape(self.batch_size)

        #Read source file
        #I'm just assuming we have this method that reads the source file
        #and returns a list of tuples in the form of (img, label)
        self.imgTuples = readSrcFile(src_file) 
        
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