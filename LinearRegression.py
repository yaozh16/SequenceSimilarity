import dtaidistance
import numpy as np
import time
from matplotlib import pyplot as plt
from dtaidistance import dtw_visualisation as dtwvis
import math
import CommonOperation
import os
import random
import SimilarityAnalyzer
import KPI_Loader

from sklearn import linear_model

def time_series_dtw_mapping_path(s1,s2,max_shift_rate=0.005,image_output=None):
    T0=time.time()
    print("find warping paths...",end=" ")
    d, paths = dtaidistance.dtw.warping_paths_fast(s1, s2,window=int(max_shift_rate*max(len(s1),len(s2))),max_step=2 ,penalty=0.1,psi=500)
    print("done(use time: {0})".format(time.time()-T0))
    T0=time.time()
    print("\tfind best path...",end=" ")
    best_path = dtaidistance.dtw.best_path(paths)
    print("done(use time: {0})".format(time.time()-T0))
    if(image_output):
        dtwvis.plot_warpingpaths(s1, s2, paths, best_path, filename=image_output)
    del paths
    return (best_path)

def relax_map_path(s1,s2,path,relax_rate=0.1):
    relaxed_path = path[int(len(path) * relax_rate):int(-len(path) * relax_rate)]
    mapped_x = [s1[p[0]] for p in relaxed_path]
    mapped_y = [s2[p[1]] for p in relaxed_path]

    '''plt.scatter(mapped_x, mapped_y, c='r')
    plt.show()'''
    return mapped_x,mapped_y
def applyRegression(m1,m2):
    reg=linear_model.LinearRegression()
    data_fit = [[x ** 0.5, x ** 0.8, x, x ** 1.2, x ** 2, x ** 3] for x in m1]
    reg.fit(data_fit, m2)
    score = reg.score(data_fit, m2)
    return score
def scatter_plot(k1, k2List, s1, s2List, errs, output_path):
    L=len(s2List)
    c=2
    r=L
    plt.figure(figsize=(30*c, 10*r))
    for i,s2 in enumerate(s2List):
        k2=k2List[i]
        #best_path=time_series_dtw_mapping_path(s1,s2,0.4,k1+"_"+k2+".png")
        best_path=time_series_dtw_mapping_path(s1,s2,0.1,None)
        m1,m2=relax_map_path(s1,s2,best_path,0.1)

        plt.subplot(r,c,2*i+1)
        if(errs[i]<3):
            plt.plot(range(len(s2)),s2, c='r',marker='.')
        else:
            plt.plot(range(len(s2)),s2, c='g',marker='.')
        plt.title("%s(%.2f)" % (k2List[i], errs[i]))

        plt.subplot(r,c,2*i+2)
        del best_path
        if(errs[i]<3):
            plt.scatter(m1, m2, c='r',marker='.')
        else:
            plt.scatter(m1, m2, c='g',marker='.')
        plt.title("best(%.2f)" % (applyRegression(m1,m2)))
        del m1,m2
        plt.legend('r')
    plt.savefig(output_path)
    #plt.show()
    plt.close()



def test(similarity_dict_path,srcdir):

    similarity_dict=CommonOperation.pickleLoad(similarity_dict_path,{})
    ks=list(similarity_dict.keys())
    rand_k=ks[random.randint(0,len(ks)-1)]
    #rand_k=ks[0]
    del ks
    print("random select {0} ".format(rand_k))


    sampled_dict=[ [k,similarity_dict[rand_k][k]]  for k in list(similarity_dict[rand_k])]
    sampled_dict=sorted(sampled_dict,key=lambda x:x[1][0])
    sampled_dict=sampled_dict[0:9]
    del similarity_dict
    kpi=KPI_Loader.read_kpi(src_dir=srcdir,sampleSize=0,needed=[x[0] for x in sampled_dict])

    markers=[e[0] for e in sampled_dict]
    filtered_kpi_value=[kpi[k].value.values for k in markers]
    titles_errs=[e[1][0] for e in sampled_dict]

    del kpi
    scatter_plot(rand_k, markers, filtered_kpi_value[0], filtered_kpi_value,titles_errs,
                 output_path="test.png")
if __name__=="__main__":
    test(os.path.join("backup", "similarity_dict.dat"),os.path.join("data_generated", "10080"))


