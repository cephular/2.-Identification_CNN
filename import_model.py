import os
import glob
import numpy as np
import math

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Convolution1D, Dropout, MaxPool1D
from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import random as rn
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from keras.models import load_model


# 초기설정--------------------------------------------------------------------------------------------

NUMOFPERSON = 2
WINDOWSIZE = 60 * 1  # 1sample/20ms
FILTERS = 32
POOLSIZE = 1
KERNELSIZE = 10
BATCHSIZE = 128
SEED = 1
PATIENCE = 5
SKIPIDLE_S = 50 * 40
SKIPIDLE_E = 50 * 30
SHUFFLEDATA = True
PREPROCESSED = True
TRAINED = True


drivePath = "./Study_Authentication/"
inputPath = drivePath + "dataInput"
npPath = drivePath + 'npData_p' + \
    str(NUMOFPERSON) + "_winSize" + str(WINDOWSIZE) + "/"

# @title
# 서브루틴--------------------------------------------------------------------------------------------


def cleaningFilename(flist, remove):
    flist = flist.copy()
    for idx, fname in enumerate(flist):
        flist[idx] = flist[idx].replace(remove, "")
    return flist


def seed_everything(seed: int = 42):
    rn.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    tf.random.set_seed(seed)


seed_everything(SEED)


# 데이터 병합---------------------------------------------------------------------------------------------
X_train, X_test, y_train, y_test = [], [], [], []
isPut = False

npFiles = os.listdir(npPath)
for fname in tqdm(npFiles):
    fname = npPath + "/" + fname
    if not isPut:
        X_train_tp, X_test_tp, y_train_tp, y_test_tp = np.load(
            fname, allow_pickle=True)
        X_train = X_train_tp
        X_test = X_test_tp
        y_train = y_train_tp
        y_test = y_test_tp
        isPut = True
    else:
        X_train_tp, X_test_tp, y_train_tp, y_test_tp = np.load(
            fname, allow_pickle=True)
        X_train = np.concatenate((X_train, X_train_tp), axis=0)
        X_test = np.concatenate((X_test, X_test_tp), axis=0)
        y_train = np.concatenate((y_train, y_train_tp), axis=0)
        y_test = np.concatenate((y_test, y_test_tp), axis=0)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
print(X_train[0][0])

# @title
# 모델저장--------------------------------------------------------------------------------------------

model_dir = drivePath + '/model'
if not os.path.exists(model_dir):
    os.mkdir(model_dir)
model_path = model_dir + '/img_classification.model' + \
    str(NUMOFPERSON)+"_WSIZE"+str(WINDOWSIZE)+"_SEED"+str(SEED)

model = load_model(model_path)

# @title
# 결과저장--------------------------------------------------------------------------------------------
res_x = model.predict(X_test)
with open(drivePath + "/result/res_P"+str(NUMOFPERSON)+"_WSIZE"+str(WINDOWSIZE)+"_SEED"+str(SEED)+".csv", "w")as file:

    file.write("predict, actual")
    for i in range(1, NUMOFPERSON+1):
        file.write("," + "P"+str(i))
    file.write("\n")

    for idx, p in enumerate(res_x):
        # file.write(str(p))
        file.write(str(np.argmax(p)))
        file.write(","+str(np.argmax(y_test[idx])))
        for i in p:
            file.write(","+str(float(i)))
        file.write("\n")

# @title
# 정확도 확인-----------------------------------------------------------------------------------------
with open(drivePath + "/result/res_P"+str(NUMOFPERSON)+"_WSIZE"+str(WINDOWSIZE)+"_SEED"+str(SEED)+".csv", "r")as file:
    numOfCases_equal = 0
    numOfCases_total = 0
    for line in file.readlines():
        line = line.replace("\n", "").split(",")
        p = line[0]
        a = line[1]
        if p == a:
            numOfCases_equal += 1
        numOfCases_total += 1

    print("numOfCases_equal", numOfCases_equal)
    print("numOfCases_total", numOfCases_total)
    print(numOfCases_equal/numOfCases_total*100, "%")
