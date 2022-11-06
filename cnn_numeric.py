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


# 초기설정--------------------------------------------------------------------------------------------

NUMOFPERSON = 10
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
PREPROCESSED = False


drivePath = "./Study_Authentication/"
inputPath = drivePath + "dataInput"
npPath = drivePath + 'npData_p' + \
    str(NUMOFPERSON) + "_winSize" + str(WINDOWSIZE) + "/"


for gpu in tf.config.experimental.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(gpu, True)

# @title
# 서브루틴--------------------------------------------------------------------------------------------
tf.random.set_seed(SEED)


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

# @title
# 전처리---------------------------------------------------------------------------------------------

if(not PREPROCESSED):
    files = os.listdir(inputPath)[:NUMOFPERSON]
    categories = cleaningFilename(files, ".csv")

    for pidx, fname in enumerate(files):
        X = []
        Y = []
        print(fname)
        fdata_x = []
        fdata_y = []
        fpath = inputPath + "/" + fname
        samplelen = 0
        with open(fpath, "r") as file:
            lines = file.readlines()
            start = int(WINDOWSIZE + 1) + SKIPIDLE_S
            end = len(lines) - SKIPIDLE_E
            for row, line in enumerate(tqdm(lines[start:end], desc=str(pidx+1)+"/"+str(NUMOFPERSON))):
                sample = []
                samplelen += 1
                slicefrom = int(row-WINDOWSIZE)+1
                sliceat = row
                #print("Read", pidx+1,":", row, "/", len(lines))
                for i in range(slicefrom, sliceat+1):
                    line = lines[i].split(",")
                    data = list(map(float, line))
                    #data_acc = math.sqrt(data[1]**2 + data[2]**2 + data[3]**2)
                    #data_gyro = math.sqrt(data[4]**2 + data[5]**2 + data[6]**2)
                    # sample.append(data_acc)
                    # sample.append(data_gyro)
                    for d in data[1:]:
                        sample.append(d)
                fdata_x.append(sample)

        shape_x = np.array(fdata_x).shape
        # print(shape_x)

        for t in range(samplelen):
            label = [0 for i in range(NUMOFPERSON)]
            label[pidx] = 1
            fdata_y.append(label)
        shape_y = np.array(fdata_y).shape
        # print(shape_y)

        # np 데이터 저장
        X += fdata_x
        Y += fdata_y

        X = np.array(X)
        Y = np.array(Y)
        X = X.reshape(X.shape[0], X.shape[1], 1)

        X_train, X_test, y_train, y_test = train_test_split(
            X, Y, shuffle=SHUFFLEDATA)
        xy = (X_train, X_test, y_train, y_test)

        if not os.path.exists(npPath):
            os.mkdir(npPath)
        np.save(npPath + "/image_data_" + str(fname) + ".npy", xy)


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
# 모델설정--------------------------------------------------------------------------------------------

model = Sequential()
model.add(Convolution1D(filters=FILTERS, kernel_size=KERNELSIZE,
          input_shape=X_train.shape[1:], activation='relu'))
model.add(MaxPool1D(POOLSIZE))
model.add(Dropout(0.3))
model.add(Convolution1D(filters=FILTERS,
          kernel_size=KERNELSIZE,  activation='relu'))
model.add(MaxPool1D(POOLSIZE))
model.add(Convolution1D(filters=FILTERS,
          kernel_size=KERNELSIZE,  activation='relu'))
model.add(MaxPool1D(POOLSIZE))
model.add(Dropout(0.3))
model.add(Convolution1D(filters=FILTERS,
          kernel_size=KERNELSIZE,  activation='relu'))
model.add(MaxPool1D(POOLSIZE))
model.add(Flatten())
model.add(Dense(NUMOFPERSON, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])

model.summary()

# @title
# 모델저장--------------------------------------------------------------------------------------------

model_dir = drivePath + '/model'
if not os.path.exists(model_dir):
    os.mkdir(model_dir)
model_path = model_dir + '/img_classification.model' + \
    str(NUMOFPERSON)+"_WSIZE"+str(WINDOWSIZE)+"_SEED"+str(SEED)

if os.path.isfile(model_path):
    os.remove(model_path)

# @title
# 모델학습--------------------------------------------------------------------------------------------

checkpoint = ModelCheckpoint(
    filepath=model_path, monitor='val_loss', verbose=1, save_best_only=True)
early_stopping = EarlyStopping(monitor='val_loss', patience=PATIENCE)
history = model.fit(X_train, y_train, batch_size=BATCHSIZE, epochs=50, validation_data=(
    X_test, y_test), callbacks=[checkpoint, early_stopping])


print("정확도 : %.4f" % (model.evaluate(X_test, y_test)[1]))

# @title
# 결과저장--------------------------------------------------------------------------------------------
res_x = model.predict(X_test)
with open(drivePath + "/result/res_P"+str(NUMOFPERSON)+"_WSIZE"+str(WINDOWSIZE)+"_SEED"+str(SEED)+".csv", "w")as file:
    file.write("predict, actual\n")
    for idx, p in enumerate(res_x):
        # file.write(str(p))
        file.write(str(np.argmax(p)))
        file.write(","+str(np.argmax(y_test[idx])))
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
