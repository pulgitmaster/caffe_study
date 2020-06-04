from __future__ import print_function
import os
import sys
import numpy as np
import pandas as pd
import cv2
import random

##############################################################################
def gray_to_rgb(img, dtype=np.uint8):                                       ##
        return np.asarray(np.dstack((img, img, img)), dtype=dtype)          ##
##############################################################################

def my_print(text):
    sys.stdout.write(str(text))
    sys.stdout.flush()

def reproduce():
    # read csv file
    file = 'fer2013.csv'
    data = pd.read_csv(file)

    trainset = data[data.Usage=='Training']
    testset = data[data.Usage=='Testing']

    trainX = []
    trainY = []
    testX = []
    testY = []

    my_print("Get trainset from csv...\n")
    for i in range(len(trainset)):
        # get img
        train_img = list(map(np.uint8, trainset['pixels'].iloc[i].split(" ")))
        train_img = np.array(train_img).reshape(48, 48) # type conversion : from list to numpy array
        # get label
        train_label = trainset['emotion'].iloc[i].astype(np.int64)  # 0, 1, 2, 3, 4, 5, 6

        trainX.append(gray_to_rgb(train_img))
        trainY.append(train_label)

    my_print("Get testset from csv...\n")
    for i in range(len(testset)):
        # get img
        test_img = list(map(np.uint8, testset['pixels'].iloc[i].split(" ")))
        test_img = np.array(test_img).reshape(48, 48) # type conversion : from list to numpy array
        # get label
        test_label = testset['emotion'].iloc[i].astype(np.int64)  # 0, 1, 2, 3, 4, 5, 6

        testX.append(gray_to_rgb(test_img))
        testY.append(test_label)

    train_path = "train/"
    test_path = "test/"

    if not os.path.isdir(train_path):
        os.mkdir(train_path)
    if not os.path.isdir(test_path):
        os.mkdir(test_path)

    
    # make directory for labels(7)
    label_name_list = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
    for label_name in label_name_list:
        if not os.path.isdir(os.path.join(train_path, label_name)):
            os.mkdir(os.path.join(train_path, label_name))
        if not os.path.isdir(os.path.join(test_path, label_name)):
            os.mkdir(os.path.join(test_path, label_name))
    
    my_print("Generate trainset...\n")
    global_cnt = [0 for _ in range(7)]
    for i in range(len(trainset)):
        for idx, label_name in enumerate(label_name_list):
            if trainY[i] == idx:
                # save img
                cv2.imwrite(os.path.join(train_path, label_name, '{:04d}.png'.format(global_cnt[idx])), trainX[i])
                global_cnt[idx]+=1

    my_print("Generate testset...\n")
    global_cnt = [0 for _ in range(7)]
    for i in range(len(testset)):
        for idx, label_name in enumerate(label_name_list):
            if testY[i] == idx:
                # save img
                cv2.imwrite(os.path.join(test_path, label_name, '{:04d}.png'.format(global_cnt[idx])), testX[i])
                global_cnt[idx]+=1

if __name__ == '__main__':
    reproduce()