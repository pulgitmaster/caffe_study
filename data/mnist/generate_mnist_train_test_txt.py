import os

caffe_study_root = "/home/yb/Desktop/caffe_study/"

mnist_path = os.path.join(caffe_study_root, "data", "mnist")
if not os.path.isdir(mnist_path):
    print("There is no dir :", mnist_path)
    exit()
else:
    mnist_train_png_path = os.path.join(mnist_path, "mnist_train_png")
    mnist_test_png_path = os.path.join(mnist_path, "mnist_test_png")

    if os.path.isdir(mnist_train_png_path)==False or os.path.isdir(mnist_test_png_path)==False:
        print("There is no dir :", mnist_train_png_path, mnist_test_png_path)
        exit()
    
    with open('train_list.txt', 'w') as f:
        for dirname in os.listdir(mnist_train_png_path):
            #print(dirname)
            dirname_full = os.path.join(mnist_train_png_path, dirname)
            for filename in os.listdir(dirname_full):
                filename = os.path.join(dirname_full, filename)
                f.write(filename+' ')
                f.write(dirname+'\n')

    with open('test_list.txt', 'w') as f:
        for dirname in os.listdir(mnist_test_png_path):
            #print(dirname)
            dirname_full = os.path.join(mnist_test_png_path, dirname)
            for filename in os.listdir(dirname_full):
                filename = os.path.join(dirname_full, filename)
                f.write(filename+' ')
                f.write(dirname+'\n')

