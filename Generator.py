#coding=utf-8

import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import CommonOperation
import time

def normalizeCol1Z(array):
    return (array-np.mean(array))/np.std(array)
def normalizeCol1(rawdata):
    max1 = np.max(rawdata)
    min1 = np.min(rawdata)
    ret=(rawdata - min1) / (max1 - min1)
    pct = np.percentile(ret, (25, 50, 75), interpolation='midpoint')
    factor = -1.0 if (np.mean(pct) > 0.5) else 1.0
    return ret* factor
def normalizeCol2(rawdata):
    rawdata[:, 2] = normalizeCol1(rawdata[:,2])
    return rawdata
def gamma_transY(src, batch_size=10080,gamma=None):
    if(gamma==None):
        gamma=random.random() * 1.5 + 0.6
    ret = src.copy()
    min = np.min(ret[:, 2])
    max = np.max(ret[:, 2])
    ret[:, 2] = (((ret[:, 2] - min) / (max - min)) ** gamma) * (max - min) + min
    return ret
def slice_transX(src,batch_size=10080):
    ret = src.copy()
    N=ret.shape[1]
    while (ret.shape[0] > batch_size * 0.95):
        row = random.randint(1, ret.shape[0] - 2)
        e = ret[row][N-1] - ret[row - 1][N-1] - ret[row + 1][N-1]
        if (e == 1):
            continue
        ret = np.delete(ret, row, 0)
    return ret
def array_smooth(src,smooth_window=10):
    ret = np.zeros(len(src))
    L = src.shape[0]
    dist = np.array([1.0 / (1 + np.abs(j)) ** 0.3 for j in range(-smooth_window, smooth_window + 2)])
    for i in range(0, L - 1):
        lb_off = int(max(0, i - smooth_window)) - i
        ub_off = int(min(L - 1, i + smooth_window)) - i
        subarr = src[lb_off + i:ub_off + i]
        subdist = dist[smooth_window + lb_off:smooth_window + ub_off]
        ret[i] = np.mean(subarr / subdist)
    return ret
def smooth_flattenY(src,batch_size=10080, smooth_window=10):
    ret = src.copy()
    ret[:,2]=array_smooth(src[:,2],smooth_window)
    return ret
def stretch_transX(src,batch_size=10080):
    ret = src.copy()
    while (ret.shape[0] < batch_size * 1.05):
        row = random.randint(1, ret.shape[0] - 2)
        ret = np.insert(ret, row, values=ret[row + random.randint(-1, 1)], axis=0)
    return ret
def mixNoise(src, noiseAbs=0.0001):
        ret = src.copy()
        max = np.max(ret[:, 2])
        maxN = max * noiseAbs
        noise = np.random.random([1, ret.shape[0]]) * maxN
        ret[:, 2] = ret[:, 2] + noise
        max=np.max(ret[:,2])
        min=np.min(ret[:,2])
        ret[:,2]=(ret[:,2]-min)/(max-min)
        return ret
def complexOper(src, extraCount):
    operations = {"sli": slice_transX, "fla": smooth_flattenY, "str": stretch_transX, "gam": gamma_transY}
    operList = [operations[each] for each in list(operations)]
    generated=src.copy()
    for j in range(extraCount):
        func = operList[random.randint(0, len(operList) - 1)]
        generated = func(generated)
    return generated
def shifted_data(rawdata, offset=None, shift_rate=0.0001,batch_size=10080):
    raw_length=len(rawdata)
    if(offset==None):
        offset=random.randint(0,rawdata.shape[0]-1)
    else:
        offset += random.randint(-int(30), int(30))
        #offset += random.randint(-int(batch_size * shift_rate), int(batch_size * shift_rate))
    if (offset < 0):
        offset = 0
    end = offset + batch_size
    if (end >= raw_length):
        ret= np.vstack([rawdata[offset:raw_length,:],rawdata[0:end-raw_length,:]])
    else:
        ret= rawdata[offset:end, :]
    while(ret.shape[0]<batch_size):
        ret=np.vstack([ret,rawdata[:batch_size-ret.shape[0]]])
    return ret

def writeCsv(matrix, filePath):
    # 这里只需要score,value,anomaly
    with open(filePath, "wt") as of:
        of.write("timestamp,score,value,anomaly\n")
        for i in range(matrix.shape[0]):
            of.write(str(i))
            of.write(",")
            of.write(str(matrix[i][1]))
            of.write(",")
            of.write(str(matrix[i][2]))
            of.write(",")
            of.write(str(matrix[i][matrix.shape[1]-1]))
            of.write("\n")
        of.close()
def readRaw(src_dir,part_str='.dashboard2.',removeValueCol=True,normalizeValue=True):
    alldata={}
    for file in os.listdir(src_dir):
        if not (part_str in os.path.join(src_dir, file)): continue
        rawdata = np.array(pd.read_csv(os.path.join(src_dir, file)).fillna(0))
        if(removeValueCol):
            rawdata=np.delete(rawdata,2,1)
        if(normalizeValue):
            rawdata=normalizeCol2(rawdata)
        print("\033[1;32m {0} \033[0m: shape:{1}".format(file, rawdata.shape))
        alldata[file]=rawdata
    return alldata
def generate_data(src_dir, dst_dir, part_str='.dashboard2.', batch_size=10080):
    CommonOperation.checkDirectory(dst_dir)
    operations = {"sli": slice_transX, "fla": smooth_flattenY, "str": stretch_transX, "gam": gamma_transY}
    alldata=readRaw(src_dir,part_str)
    candidate_serials=[]
    for file in alldata:
        rawdata=alldata[file]
        raw_length = rawdata.shape[0]
        marker = file.split(".")[-2][0]  # a/b/c
        def process2csv(rawdata,offset, marker,batch_size):
            # raw data
            generated=shifted_data(rawdata,offset,0,batch_size)
            candidate_serials.append([marker,generated])
            writeCsv(generated, os.path.join(dst_dir, marker + ".csv"))
            # basic oper
            for prefix, func in operations.items():
                generated = mixNoise(func(shifted_data(rawdata,offset)))
                candidate_serials.append([marker + "_" + prefix,generated])
                writeCsv(generated, os.path.join(dst_dir, marker + "_" + prefix + ".csv"))
                print("\t" + prefix + " operation done")
            # mixed operation data
            generated=mixNoise(complexOper(shifted_data(rawdata,offset),6))
            candidate_serials.append([marker + "_mix",generated])
            writeCsv(generated, os.path.join(dst_dir, marker + "_mix.csv"))
            print("\tcomplex operation done")
            print(marker + " generate done")
        random.seed(time.time())
        for batch_index in range(3):
            process2csv(rawdata,random.randint(0,raw_length-batch_size), marker + str(batch_index),batch_size)
    return candidate_serials
def combineData(src_dir, dst_dir, part_str='.dashboard2.', batch_size=10080,count=12):
    def combined(sd1,sd2):
        div=50
        while div>40 and div<60:
            div=random.randint(0,100)
        if(div<=50):
            ret = sd2.copy()
        else:
            ret = sd1.copy()
        ret[:,2]=sd1[:,2]*(div/100.0)+sd2[:,2]*(1-div/100.0)
        return normalizeCol2(complexOper(ret,4))
    alldata=readRaw(src_dir,part_str)
    alldata=[alldata[k] for k in alldata]
    def process2csv(rawdata1,rawdata2, count ,marker):
        generated=combined(shifted_data(rawdata1,None,batch_size=batch_size),shifted_data(rawdata2,None,batch_size=batch_size))
        for i in range(count):
            generated=combined(generated,shifted_data((rawdata2 if i%2==1 else rawdata1),None,batch_size=generated.shape[0]))
        writeCsv(generated, os.path.join(dst_dir,"combined_{0}.csv".format(marker)))
        print(marker + " generate done")
    for i in range(count):
        r1=random.randint(0,len(alldata)-1)
        r2=random.randint(0,len(alldata)-1)
        while(r2==r1):
            r2=random.randint(0,len(alldata)-1)
        process2csv(alldata[r1],alldata[r2],5, str(i))
def combineFromCandidates(candidate_serials,dst_dir,batch_size=10080,count=10):
    def combined(sd1,sd2):
        div=50
        while (div>40 and div<60) or (div<20) or (div>80):
            div=random.randint(0,100)
        if(div<=50):
            ret = sd2.copy()
        else:
            ret = sd1.copy()
        ret[:,2]=sd1[:,2]*(div/100.0)+sd2[:,2]*(1-div/100.0)
        return normalizeCol2(complexOper(ret,4))
    def process2csv(rawdata1,rawdata2, marker):
        generated=combined(shifted_data(rawdata1,None,batch_size=batch_size),shifted_data(rawdata2,None,batch_size=batch_size))
        writeCsv(generated, os.path.join(dst_dir,"tcombined_{0}.csv".format(marker)))
        print(marker + " generate done")
    for i in range(count):
        r1=random.randint(0,len(candidate_serials)-1)
        r2=random.randint(0,len(candidate_serials)-1)
        while(candidate_serials[r1][0][0]==candidate_serials[r2][0][0]):
            r2=random.randint(0,len(candidate_serials)-1)
        process2csv(candidate_serials[r1][1],candidate_serials[r2][1], "{0}".format(str(i)))
def view_data(src_dir,dst_dir,graph_size=6):
    import KPI_Loader
    kpi_dict = KPI_Loader.read_kpi(src_dir=os.path.join("data_generated", "10080"))
    kpi_dict = KPI_Loader.preprocess(kpi_dict)
    CommonOperation.checkDirectory(dst_dir)
    colorlist=['purple','b','g']
    for index,fileName in enumerate(kpi_dict):
        kpi=kpi_dict[fileName]
        if(index%graph_size==0):
            plt.figure(figsize=(20, 20))
        plt.subplot(graph_size, 1, (index%graph_size)+1)
        color=colorlist[index%len(colorlist)]
        #plt.plot(kpi.timestamp, kpi.score,c='y')
        plt.plot(kpi.timestamp, kpi.value,c=color)
        plt.scatter(kpi.loc[kpi.anomaly == 1, 'timestamp'], kpi.loc[kpi.anomaly == 1, 'value'],c='r')
        plt.title(fileName,fontsize=30,color=color)
        if(index%graph_size==graph_size-1):
            plt.savefig(os.path.join(dst_dir,fileName+".png"))
            plt.close()
if __name__ =="__main__":
    dst_dir=os.path.join("data_generated","10080")
    #candidates=generate_data(src_dir="data_original",dst_dir=dst_dir,batch_size=10080)
    #combineData(src_dir="data_original",dst_dir=dst_dir,batch_size=10080,count=12)
    #combineFromCandidates(candidates,dst_dir=dst_dir,batch_size=10080,count=24)
    view_data(src_dir=dst_dir,dst_dir=os.path.join("data_generated","view","10080"),graph_size=12)