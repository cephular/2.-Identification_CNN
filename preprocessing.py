import numpy as np
import os
import matplotlib.pyplot as plt
import scipy
from scipy import signal


#-------------------------------------------------초기설정-------------------------------------------------#


fileNum = 3
rawdata_path = "C:/Users/cephu/Documents/GitHub/IoTLab_data/raw"
output_path = "./Study_Authentication/dataOutput/"
numOfTrain = 12
window_size = 2

numOfFeature = 6
numOfFeature *= (window_size*2+1)
numOfOutput = 5

#-------------------------------------------------서브루틴-------------------------------------------------#

# 전처리 (req = casting(), return_column_list())


def preprocessing(fileNum, r_path, o_path):
    # 사람 탐색
    for directory_name in os.listdir(r_path)[2:]:
        print(directory_name)
        if fileNum == 0:
            break
        fileNum -= 1
        # 관성센서 탐색
        for data_name in os.listdir(r_path+"/"+directory_name):
            if data_name == "l_shimmer.csv":
                # =time + acc x,y,z + gyro x,y,z / size=7
                data = casting(f_path=r_path+"/"+directory_name+"/"+data_name)
                data = linear_interpolation(data)
                save_data(o_path+"/"+directory_name+".csv", data)
                visualizing(data)


# 데이터 형변환 (req = return_column_list())
def casting(f_path):
    columnInfo = None
    data = [[], [], [], [], [], [], []]
    with open(f_path, 'r') as file:
        # 한 라인씩 읽기
        for lineNum, line in enumerate(file.readlines()):
            if lineNum == 1:
                # =time + acc x,y,z + gyro x,y,z / size=7
                columnInfo = return_column_list(line)

            if lineNum >= 3:
                line = line.split("\t")
                for cidx in columnInfo:
                    data[cidx].append(float(line[cidx]))
    return data


# 컬럼 인덱스 정보 반환
def return_column_list(colLine):
    timestamp = []
    accIdxs = []
    gyroIdxs = []
    for cIdx, column in enumerate(colLine.split("\t")):
        if "Timestamp" in column:
            timestamp.append(cIdx)
        if "Accel" in column:
            accIdxs.append(cIdx)
        if "Gyro" in column:
            gyroIdxs.append(cIdx)
    return timestamp + accIdxs + gyroIdxs


# 선형보간법
def linear_interpolation(data):
    interData = [[], [], [], [], [], [], []]
    firstTick = data[0][0]
    indicator = 0
    for lineIdx, dtime in enumerate(data[0]):
        if not(lineIdx+1 >= len(data[0])):
            print("INTERPOLATION: ", lineIdx, "/", len(data[0]))
            dtime = int(dtime-firstTick)
            ntime = int(data[0][lineIdx+1]-firstTick)

            if (lineIdx == 0) or (dtime == indicator) or (indicator > dtime and indicator < ntime):
                interData[0].append(indicator)

                if (lineIdx == 0) or (dtime == indicator):
                    for cIdx in range(1, len(data)):
                        interData[cIdx].append(data[cIdx][lineIdx])
                else:
                    for cIdx in range(1, len(data)):
                        y = None
                        y0 = data[cIdx][lineIdx]
                        y1 = data[cIdx][lineIdx+1]
                        x = indicator
                        x0 = dtime
                        x1 = ntime
                        y = y0 + (y1-y0)*((x-x0)/(x1-x0))
                        interData[cIdx].append(y)
                indicator += 20
                continue
    data = None
    return interData

# 데이터 저장


def save_data(f_path, data):
    data = data.copy()
    with open(f_path, 'w') as file:
        for lineIdx in range(len(data[0])):
            inline = []
            for cIdx in range(len(data)):
                inline.append(str(data[cIdx][lineIdx]))
            file.write(",".join(inline)+"\n")

# 데이터 시각화


def visualizing(data):
    data = np.array(data)
    # plt.plot(data[1])
    # plt.plot(data[2])
    # plt.plot(data[3])

    momentum = (data[2]**2+data[3]**2)**(1/2)
    plt.plot(momentum)
    plt.show()


def visualizing_anlz(o_path):
    data = [[], [], [], [], [], [], []]
    data_f = [[], [], [], [], [], [], []]
    o_path += "m_2.csv"

    with open(o_path, 'r') as file:
        for line in file.readlines():
            line = line.split(",")
            for idx, d in enumerate(line):
                data[idx].append(float(d))
                data[idx].append(float(d))

    data = np.array(data)

    # 노이즈 리무버
    N = 2  # Filter order
    Wn = 0.1  # Cutoff frequency
    B, A = signal.butter(N, Wn, output='ba')
    for idx, d in enumerate(data):
        data_f[idx] = signal.filtfilt(B, A, data[idx])

    # 시각화
    momentum = (data[1]**2+data[2]**2+data[3]**2)**(1/2)-9.8
    #momentum = (data[3]**2+data[4]**2+data[5]**2)**(1/2)
    #momentum = data[2]
    N = 5
    Wn = 0.02
    B, A = signal.butter(N, Wn, output='ba')
    momentum_f = signal.filtfilt(B, A, momentum)

    # peak
    momentum_p, d = scipy.signal.find_peaks(momentum_f, height=None, threshold=None, distance=None,
                                            prominence=None, width=None, wlen=None, rel_height=0.5, plateau_size=None)
    tp = []
    for idx in momentum_p:
        tp.append(momentum_f[idx])

    plt.plot(momentum_f, label="Filtered_ACC_X")
    #plt.plot(momentum_f, label="ACC_Filtered_X")
    #plt.plot(momentum_p, tp, 'ro', label="peak")
    plt.xlabel("Sample ID (1 samples/20ms)")
    plt.ylabel("Acceleration_X")
    plt.legend()
    plt.show()

    # 업샘플링 및 보간
    momentum = momentum_f[1918:1968]
    mom_tp = momentum.copy()
    cnt = 0
    for idx in range(len(momentum_f)):
        if idx == 50:
            break
        print(idx)
        if cnt == 0:
            mom_tp[idx] = -1
        else:
            momentum[idx] = -1
        cnt += 1
        if cnt == 3:
            cnt = 0

    plt.plot(momentum, 'bo', label="ACC_X")
    plt.plot(mom_tp, 'rx', label="interpolated")
    plt.xlabel("Sample ID (1 samples/20ms)")
    plt.ylabel("Acceleration_X")
    plt.legend()
    plt.show()


#-------------------------------------------------메인루틴-------------------------------------------------#

if __name__ == "__main__":
    #preprocessing(fileNum, rawdata_path, output_path)
    visualizing_anlz(output_path)
