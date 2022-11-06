# @title
# 초기설정--------------------------------------------------------------------------------------------

NUMOFPERSON = 2
WINDOWSIZE = 60 * 1  # 1sample/20ms
SEED = 1

drivePath = "./Study_Authentication/"

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
