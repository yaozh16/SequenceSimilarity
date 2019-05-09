import os
import pickle
import numpy as np
from matplotlib import pyplot as plt

def checkDirectory(dirPath):
    if(not os.path.exists(dirPath) and dirPath!=""):
        pa,ch=os.path.split(dirPath)
        checkDirectory(pa)
        os.mkdir(dirPath)

def checkFileDirectory(filePath):
    checkDirectory(os.path.split(filePath)[0])

def pickleLoad(filePath,defaultObject):
    if(not os.path.exists(filePath)):
        print("pickle file not found. return with default value.",flush=True)
        return defaultObject
    with open(filePath,"rb") as f:
        ret=pickle.load(f)
        f.close()
        print("pickle file loaded from {0}".format(filePath),flush=True)
        return ret

def pickleDump(filePath,object):
    checkFileDirectory(filePath)
    with open(filePath,"wb") as f:
        pickle.dump(object,f)
        f.close()
    print("pickle file dumped to {0}".format(filePath),flush=True)

def writeObjectList(objectList, outputpath):
    checkFileDirectory(outputpath)
    with open(outputpath,'w') as f:
        for each in objectList:
            f.write(each+"\n")
        f.close()

def normMaxMin(src):
    minn,maxn=np.percentile(src,[0.01,0.99])
    return (src-minn)/(maxn-minn) if maxn>minn else np.array([0.5]*len(src))

def normZScore(src):
    d=np.std(src)
    return (src-np.mean(src))/d if d>0 else np.array([0.5]*len(src))

def array_smooth(src,smooth_window=10):
    ret = np.zeros(len(src))
    L = src.shape[0]
    dist = np.array([1/(1 + np.abs(j)) ** 0.3 for j in range(-smooth_window, smooth_window + 2)])
    for i in range(0, L - 1):
        lb_off = int(max(0, i - smooth_window)) - i
        ub_off = int(min(L - 1, i + smooth_window)) - i
        subarr = src[lb_off + i:ub_off + i]
        subdist = dist[smooth_window + lb_off:smooth_window + ub_off]
        ret[i] = np.mean(subarr * subdist)/np.mean(subdist)
    return ret

def plotSerials(serialList,nameList,targetList,plotList,lowerbound=None,upperbound=None,markX=None,show=True,offsets={} ):
    plt.figure(figsize=(20,10))
    N=len(plotList)
    colorList=['r','b','y','g','purple','k','c']
    objcolorList=['r','b','y','g',"#900302",'cyan','k']
    for i,k in enumerate(plotList):
        #plt.subplot(N,1,i+1)
        if i in offsets:
            offset=offsets[i]
        else:
            offset=0
        curve=serialList[k][lowerbound:upperbound]
        plt.plot(range(offset,len(curve)+offset),curve,color=colorList[i%len(colorList)],linestyle="-",label="[{0}]{1}".format(targetList[k],nameList[k]))
        #plt.plot(range(offset,len(curve)+offset),curve,color=colorList[targetList[k]%len(colorList)],linestyle=":")
        if(markX!=None):
            r=range(offset,len(curve)+offset)
            plt.scatter([e for e in  r if e in markX],[curve[e] for e in  r if e in markX],color='c')
            r=range(offset,len(curve)+offset,5)
            plt.scatter([e for e in  r if e in markX],[curve[e] for e in  r if e in markX],color='y')
        plt.legend()
    if(show):
        plt.savefig("test.png")
        plt.show("test")

if __name__ =="__main__":
    pass