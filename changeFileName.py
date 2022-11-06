import os

import os
import random
import string

fpath = "./Study_Authentication/dataInput/"


def randomString(nLength):
    rand_str = ""  # 문자열
    for i in range(nLength):
        rand_str += str(random.choice(string.ascii_uppercase))
    return rand_str


def changeFilenameRandom(path):
    i = 1
    for filename in os.listdir(path):
        cName = randomString(10)
        print(path+filename, '=>', path+cName+str(i)+'.csv')
        os.rename(path+filename, path+cName+str(i)+'.csv')
        i += 1


changeFilenameRandom(fpath)
