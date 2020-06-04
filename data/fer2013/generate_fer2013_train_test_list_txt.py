import os

caffe_study_root = "/home/yb/Desktop/caffe_study/"

fer2013_path = os.path.join(caffe_study_root, "data", "fer2013")

emotion_dict = {'Angry':0, 'Disgust':1, 'Fear':2, 'Happy':3, 'Sad':4, 'Surprise':5, 'Neutral':6}

if not os.path.isdir(fer2013_path):
    print("There is no dir :", fer2013_path)
    exit()
else:
    fer2013_train_png_path = os.path.join(fer2013_path, "train")
    fer2013_test_png_path = os.path.join(fer2013_path, "test")

    if os.path.isdir(fer2013_train_png_path)==False or os.path.isdir(fer2013_test_png_path)==False:
        print("There is no dir :", fer2013_train_png_path, fer2013_test_png_path)
        exit()
    
    with open(os.path.join(fer2013_path, 'train_list.txt'), 'w') as f:
        for dirname in os.listdir(fer2013_train_png_path):
            print(dirname)
            dirname_full = os.path.join(fer2013_train_png_path, dirname)
            for filename in os.listdir(dirname_full):
                filename = os.path.join(dirname_full, filename)
                f.write(filename+' ')
                f.write(str(emotion_dict[dirname])+'\n')

    with open(os.path.join(fer2013_path, 'test_list.txt'), 'w') as f:
        for dirname in os.listdir(fer2013_test_png_path):
            print(dirname)
            dirname_full = os.path.join(fer2013_test_png_path, dirname)
            for filename in os.listdir(dirname_full):
                filename = os.path.join(dirname_full, filename)
                f.write(filename+' ')
                f.write(str(emotion_dict[dirname])+'\n')

