import random
import os
import numpy as np
import pandas as pd


def data_frame_value_normalize(kpi_data,direct_normalize):
    v=kpi_data.value.values
    minv,maxv=np.percentile(v,(1,99))
    v=(v-minv)/(maxv-minv)
    pct=np.percentile(v, (25, 50, 75))
    factor=-1 if (direct_normalize and np.mean(pct)>0.5) else 1
    kpi_data['value']=v*factor
    return kpi_data

def read_kpi(src_dir,sampleSize=None,needed=[],align=True,direct_normalize=True,firstStr="combine"):
    kpi = {}
    allfiles=[os.path.join(src_dir,each) for each in os.listdir(src_dir) ]
    neededfiles=[os.path.join(src_dir,each) for each in os.listdir(src_dir) if (each.replace(".csv","") in needed)]

    candicatedfiles=[os.path.join(src_dir,each) for each in os.listdir(src_dir) if not(each.replace(".csv","") in needed)]
    if(len([each for each in candicatedfiles if firstStr in each])>0):
        candicatedfiles=[each for each in candicatedfiles if firstStr in each]
    if(sampleSize!=None and sampleSize>=0):
        files=random.sample(candicatedfiles,min(sampleSize,len(candicatedfiles)))
        files.extend(neededfiles)
    else:
        files=allfiles
    files=sorted(files)
    print("found {0} files. {1} needed. try load {2} kpi data".format(len(allfiles),len(neededfiles),len(files)),flush=True)
    for eachFile in files:
        kpi_data = pd.read_csv(eachFile).fillna(0)
        kpi_data = data_frame_value_normalize(kpi_data,direct_normalize)
        # 筛掉点数过少的数据后右对齐所有的数据
        kpi[os.path.split(eachFile)[-1].replace('.csv', '')] = kpi_data
    if(align):
        min_length = min([len(t) for t in kpi.values()])
        for k in kpi.keys():
            kpi[k] = kpi[k].iloc[:min_length]
        print('we get {0} time series and the minimum length is {1}'.format(len(kpi), min_length),flush=True)
    else:
        print('we get {0} time series and the average length is {1}'.format(len(kpi), np.average([len(t) for t in kpi.values()])),flush=True)
    return kpi

def preprocess(kpi, use_norm=True, use_binary=False):
    # 根据 use_* 处理数据
    if use_norm:
        def maxminNorm():
            for k, v in kpi.items():
                maxx = v.value.quantile(0.999)
                minn = v.value.quantile(0.001)
                kpi[k].value = v.value.apply(lambda x: 0.5 if maxx == minn else 1 if x > maxx else 0 if x < minn else (x - minn) / (maxx - minn))
        def zscoreNorm():
            for k, v in kpi.items():

                m=np.mean(v.value.values)
                d=np.std(v.value.values)
                kpi[k].value = v.value.apply(lambda x: 0.5 if d==0 else (x-m)/d)

                #m=np.mean(v.score.values)
                #d=np.std(v.score.values)
                #kpi[k].score = v.score.apply(lambda x: 0.5 if d==0 else (x-m)/d)

        #zscoreNorm()
        maxminNorm()
    if use_binary:
        for k, v in kpi.items():
            kpi[k].score = kpi[k].anomaly
    print('finish pre-process data',flush=True)
    return kpi


