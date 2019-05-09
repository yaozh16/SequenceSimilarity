#coding=utf-8
import pandas as pd
import os
from os.path import join
import numpy as np
import Similarity
from matplotlib import pyplot as plt
import CommonOperation
import cv2
import dtaidistance
import Similarity

def loadTest(testFile):
    testData=np.array(pd.read_csv(testFile))
    targetMap=[int(e.replace("'","").replace("b","")) for e in  testData[:,-1]]
    serialMatrix=testData[:,:-1]
    serialMatrix=np.array(serialMatrix,dtype=np.float)
    N=len(testData)
    for i in range(N):
        serialMatrix[i]=CommonOperation.array_smooth(serialMatrix[i])
        serialMatrix[i]=CommonOperation.normZScore(serialMatrix[i])
        pass
    print("load {0} serial signals(each length:{1})".format(serialMatrix.shape[0],serialMatrix.shape[1]))
    print("{0} classes found:".format(len(set(targetMap))))
    for cls in set(targetMap):
        print("\t\033[1;{0}m{1}\033[0m: contains {2}".format(cls+31,cls,len([e for e in targetMap if e==cls])))
    return targetMap,serialMatrix
    pass
def loadTestKPI(testFile):
    kpi={}
    targetMap,serialMatrix=loadTest(testFile)
    for i in range(len(targetMap)):
        kpi["s{0}".format(i)]=serialMatrix[i].copy()
    return kpi
def testMethods(testFile, methodList, output_dir):
    CommonOperation.checkDirectory(output_dir)
    index2Class,serialMatrix=loadTest(testFile)
    N=len(index2Class)
    assert N==serialMatrix.shape[0]
    clsSizes=dict([[cls,len([e for e in index2Class if e==cls])] for cls in set(index2Class)])
    clsMaxScores=dict([[cls,  np.sum([N-e  for e in range(clsSizes[cls]) ])  ] for cls in set(index2Class)])
    clsMinScores=dict([[cls,  np.sum([e+1  for e in range(clsSizes[cls]-1) ])+ N  ] for cls in set(index2Class)])
    method2ScoreRate={}
    method2ErrorRate={}
    for method in methodList:
        print("method:{0}".format(method))
        scoreValues=[]
        scoreRates=[]
        errorCounts=[]
        errorRates=[]
        matrix_dump_file=join(output_dir, "similarity_matrix", "{0}".format(method))
        if(os.path.exists(matrix_dump_file)):
            simiarityMatrix=CommonOperation.pickleLoad(matrix_dump_file,{})
        else:
            simiarityMatrix=np.zeros([N,N],dtype=np.float)
            for i in range(N):
                cls=index2Class[i]
                print("{0}(id:\033[1;{2}m{1}\033[0m)".format(cls,i,cls+31),end=",")
                for j in range(i,N):
                    s=Similarity.Similarity(serialMatrix[i],serialMatrix[j])
                    simiarity_ij=s.use_method(method)
                    simiarityMatrix[i][j]=simiarity_ij
                    simiarityMatrix[j][i]=simiarity_ij
            print()
            CommonOperation.pickleDump(matrix_dump_file, simiarityMatrix)
        for i in range(N):
            cls=index2Class[i]
            similarRank=list(np.argsort(simiarityMatrix[i]))
            similarRank.reverse()
            scoreValues.append( np.sum( [ (N-j) for j,r in enumerate(similarRank) if index2Class[r]==cls ] ) )
            scoreRates.append( 100.0*(scoreValues[-1]-clsMinScores[cls])/(clsMaxScores[cls]-clsMinScores[cls]) )
            errorCounts.append( clsSizes[cls] - len( [r for r in similarRank[:clsSizes[cls]] if index2Class[r]==cls] )  )
            errorRates.append( 100.0*errorCounts[-1]/clsSizes[cls]  )
            #for j in range(N):
            #    print("{0}(id:\033[1;{2}m{1}\033[0m,sim:{3})".format(index2Class[similarRank[j]],similarRank[j],index2Class[similarRank[j]]+31,simiarityMatrix[i][similarRank[j]]),end=",")
            print("statistic:score:{0}({1}%),errorCount:{2}/({3}%)".format(scoreValues[-1],
                                                                           scoreRates[-1],
                                                                           errorCounts[-1],
                                                                           errorRates[-1]))
        #print("final statistic(average):\n\tscoreRate:{0}\n\terrorRate:{1}".format(np.average(scoreRates), np.average(errorRates)))
        method2ScoreRate[method]=scoreRates
        method2ErrorRate[method]=errorRates

        plt.figure(figsize=(40,10))
        plt.bar(range(N),scoreRates)
        plt.xlabel("index")  #设置x轴标签
        plt.ylabel("score rate")  #设置y轴标签
        plt.title("{0} score rate ".format(method))
        plt.ylim(50,100)
        plt.savefig(join(output_dir,"{0}_score_rate.png".format(method)))
        plt.close()

        plt.figure(figsize=(40,10))
        plt.bar(range(N),errorRates)
        plt.xlabel("index")  #设置x轴标签
        plt.ylabel("error rate")  #设置y轴标签
        plt.title("{0} error rate ".format(method))
        plt.ylim(0,50)
        plt.savefig(join(output_dir,"{0}_error_rate.png".format(method)))
        plt.close()

        print("statistic(average):\n|algorithm|scoreRate|errorRate|\n|:-:| :-: | :-: |")
        print("|{0}|{1}|{2}|".format(method, "%.3f"%np.average(scoreRates),"%.3f"%np.average(errorRates)))


    methodList=sorted(methodList,key=lambda method:np.average(method2ErrorRate[method]))
    print(methodList)
    print("final statistic(average):\n|algorithm|scoreRate(%)| errorRate(%)|\n|:-:| :-: | :-: |")
    for method in methodList:
        print("|{0}|{1}|{2}|".format(method,"%.2f"% np.average(method2ScoreRate[method]),"%.2f"%np.average(method2ErrorRate[method])))
    CommonOperation.pickleDump(join(output_dir,"method2ScoreRate"),method2ScoreRate)
    CommonOperation.pickleDump(join(output_dir,"method2ErrorRate"),method2ErrorRate)

def applyMethodTest():
    testFile=join("Test","pattern_Test.csv")
    output_dir=join("Test","output")
    methodList=['dtw_s20', 'dtw_s10', 'dtw_s5', 'dtw_s3', 'mlc', 'dtw_m_l', 'dtw_r', 'mcc', 'mpc', 'epc', 'ecc', 'elc', 'dtw_m_p']
    testMethods(testFile, methodList, output_dir)

def plotSerials(plotlist,methods):
    testFile = join("Test", "pattern_Test.csv")

    kpi = loadTestKPI(testFile)



    targetMap, serialMatrix = loadTest(testFile)

    s1=serialMatrix[plotlist[0]]
    s2=serialMatrix[plotlist[1]]

    SL = [serialMatrix[i] for i in plotlist]
    NL=['serial{0}'.format(i) for i in plotlist]
    TL=[targetMap[i] for i in plotlist]
    offsets={}

    dtw_ss=[e for e in methods if "dtw_s" in e]
    for m in dtw_ss:
        s=Similarity.Similarity(s1,s2)
        NL.append('serial{0} shifted by {1}({2})'.format(plotlist[-1],m,s.use_method(m)))
        TL.append(targetMap[plotlist[-1]])
        SL.append(s2+s.bestShiftY)

    dtw_ss = [e for e in methods if e.startswith('m')]
    for m in dtw_ss:
        s = Similarity.Similarity(s1, s2)
        TL.append(targetMap[plotlist[-1]])
        NL.append('serial{0} shifted by {1} :({2})'.format(plotlist[-1], m,s.use_method(m)))
        if(s.bestShiftX>0):
            offsets[len(SL)]=s.bestShiftX
            SL.append(s2[:-s.bestShiftX])
        else:
            SL.append( s2[-s.bestShiftX:])
    CommonOperation.plotSerials(SL,
                                NL,
                                TL,
                                plotList=range(len(SL)), lowerbound=0,offsets=offsets)


def viewScoreError(methods):
    output_dir=join("Test","output")
    method2ScoreRate=CommonOperation.pickleLoad(join(output_dir, "method2ScoreRate"),{})
    method2ErrorRate=CommonOperation.pickleLoad(join(output_dir, "method2ErrorRate"), {})

    CommonOperation.plotSerials([ method2ScoreRate[e] for e in methods ],
                                methods,
                                range(len(methods)),
                                plotList=range(len(methods)),
                                lowerbound=None,
                                upperbound=None,
                                markX=range(len(method2ScoreRate[methods[0]])))


def viewSimiarityMatrix(methods, curveIndex):
    output_dir=join("Test","output")
    matrix_dump_files =dict([[method,join(output_dir, "similarity_matrix", "{0}".format(method))] for method in methods])
    matrixs=dict([  [method,CommonOperation.pickleLoad(matrix_dump_files[method],{})] for method in methods])


    CommonOperation.plotSerials([matrixs[method][curveIndex] for method in methods],
                                methods,
                                range(len(methods)),
                                range(len(methods)),
                                markX=range(len(matrixs[methods[0]][curveIndex]))
                                )
    pass

if __name__=="__main__":
    #applyMethodTest()


    #viewScoreError(["dtw_s10","mcc"])
    #viewSimiarityMatrix(["dtw_s10","mcc"],20)
    #plotSerials([20,39],methods=["dtw_s10","mcc"])
    plotSerials([20,39,19,11,13],methods=[])

    #viewScoreError(["dtw_s10", "dtw_s20"])