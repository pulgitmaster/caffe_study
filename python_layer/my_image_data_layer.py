import caffe

class MyImageDataLayer(caffe.Layer):
    """
    This is a simple synchronous datalayer for training lenet with mnist.
    """
    def setup(self, bottom, top):
        self.top_names = ['data', 'label']
        # params is a python dictionary with layer parameters.
        params = eval(self.param_str)
        # store input as class variables
        self.batch_size = params['batch_size']
        # Create a batch loader to load the images.
        self.batch_loader = BatchLoader(params, None)

class BatchLoader(object):
    """
    This class abstracts away the loading of images.
    Images can either be loaded singly, or in a batch. The latter is used for
    the asyncronous data layer to preload batches while other processing is
    performed.
    """
    def __init__(self, params,):
        self.batch_size = params['batch_size']
        self.img_path = params['pascal_root']
        self.im_shape = params['im_shape']
        # get list of image indexes.
        list_file = params['split'] + '.txt'
        self.indexlist = [line.rstrip('\n') for line in open(
            osp.join(self.img_path, 'ImageSets/Main', list_file))]
        self._cur = 0  # current image
        # this class does some simple data-manipulations
        self.transformer = SimpleTransformer()

        print "BatchLoader initialized with {} images".format(
            len(self.indexlist))

    def load_next_image(self):
        """
        Load the next image in a batch.
        """
        # Did we finish an epoch?
        if self._cur == len(self.indexlist):
            self._cur = 0
            shuffle(self.indexlist)

        # Load an image
        index = self.indexlist[self._cur]  # Get the image index
        image_file_name = index + '.jpg'
        im = np.asarray(Image.open(
            osp.join(self.img_path, 'JPEGImages', image_file_name)))
        im = scipy.misc.imresize(im, self.im_shape)  # resize

        # do a simple horizontal flip as data augmentation
        flip = np.random.choice(2)*2-1
        im = im[:, ::flip, :]

        # Load and prepare ground truth
        multilabel = np.zeros(20).astype(np.float32)
        anns = load_pascal_annotation(index, self.img_path)
        for label in anns['gt_classes']:
            # in the multilabel problem we don't care how MANY instances
            # there are of each class. Only if they are present.
            # The "-1" is b/c we are not interested in the background
            # class.
            multilabel[label - 1] = 1

        self._cur += 1
        return self.transformer.preprocess(im), multilabel

class Custom_Data_Layer(caffe.Layer):
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
        
        # Create a batch loader to load the images.
        self.batch_loader = BatchLoader(params, None)

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