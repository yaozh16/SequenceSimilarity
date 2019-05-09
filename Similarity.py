#coding=utf-8
"""
calculate the similarity of two time series with same length
"""
import numpy as np
import dtaidistance
from sklearn import linear_model
from dtaidistance import dtw
from dtaidistance import dtw_visualisation as dtwvis

class Similarity(object):
    """
    各个函数的返回值均为相似度 (或者距离的倒数) 越大越相似
    """

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def use_method(self, method, *args):
        if 'eu' in method:
            return self.euclid(args[0]) if len(args) > 0 else self.euclid()
        if 'epc' == method:
            return self.pearson_coefficient(self.x,self.y)
        if 'tcc' in method:
            return self.temporal_correlation_coefficient()
        if 'ecc' in method:
            return self.easy_cross_correlation(self.x,self.y)
        if 'mcc' in method:
            return self.max_cross_correlation()
        if 'mpc' in method:
            return self.max_pearson_coefficient()
        if 'dtw_r' in method:
            return self.dtw_raw()
        if 'dtw_s' in method:
            return self.dtw_shift(shiftcount=int(method.strip("dtw_s")))
        if 'dtw_m_l' in method:
            return self.dtw_map_l()
        if 'dtw_m_p' in method:
            return self.dtw_map_p()
        if 'elc' in method:
            X=[[e] for e in self.x]
            return self.easy_linear_correlation(X,self.y)
        if 'mlc' in method:
            return self.max_linear_correlation()
        else:
            raise NotImplementedError(method)

    def euclid(self, p=1):
        return 1 / (1 + (sum([abs(x - y) ** p for x, y in zip(self.x, self.y)]) / len(self.x)) ** (1 / p))


    # 可以作为距离的系数
    def temporal_correlation_coefficient(self):
        x_diff = [t2 - t1 for t1, t2 in zip(self.x, self.x[1:])]
        y_diff = [t2 - t1 for t1, t2 in zip(self.y, self.y[1:])]
        return sum([t1 * t2 for t1, t2 in zip(x_diff, y_diff)]) / (1 + sum([t * t for t in x_diff]) ** 0.5 * sum([t * t for t in y_diff]) ** 0.5)

    def dtw_raw(self, window=60, penalty=0.01, psi=60, max_step=1, d=1):
        cx = np.power(self.x, d)
        cy = np.power(self.y, d)
        dis = dtaidistance.dtw.distance_fast(cx, cy, window=window, penalty=penalty, psi=psi, max_step=max_step)
        return 1 / (1 + dis)
    def dtw_shift(self,shiftcount=5,shiftstep=0.04, window=60, penalty=0.01, psi=60, max_step=1, d=1):
        cx=np.power(self.x,d)
        minDis=np.inf
        for i in range(2*shiftcount):
            cy = np.power(self.y+shiftstep*(i-shiftcount), d)
            dis=dtaidistance.dtw.distance_fast(cx, cy,window=window,penalty=penalty,psi=psi,max_step=max_step)
            if(minDis>dis):
                minDis=dis
                self.bestShiftY=shiftstep*(i-shiftcount)
        return 1/(1+minDis)
    def dtw_map_l(self, window=60, penalty=0.01, psi=60, max_step=1, d=1):
        cx=np.power(self.x,d)
        cy = np.power(self.y, d)
        dis, paths = dtaidistance.dtw.warping_paths_fast(cx, cy, window=window,penalty=penalty,psi=psi,max_step=max_step)
        best_path = dtaidistance.dtw.best_path(paths)
        mapped_x = np.array([cx[p[0]] for p in best_path])
        mapped_y = np.array([cy[p[1]] for p in best_path])
        X=[[e] for e in mapped_x]
        return self.easy_linear_correlation(X,mapped_y)
    def dtw_map_p(self, window=60, penalty=0.01, psi=60, max_step=1, d=1):
        cx=np.power(self.x,d)
        cy = np.power(self.y, d)
        dis, paths = dtaidistance.dtw.warping_paths_fast(cx, cy, window=window,penalty=penalty,psi=psi,max_step=max_step)
        best_path = dtaidistance.dtw.best_path(paths)
        mapped_x = np.array([cx[p[0]] for p in best_path])
        mapped_y = np.array([cy[p[1]] for p in best_path])
        X=[[e] for e in mapped_x]
        return self.pearson_coefficient(X,mapped_y)

    def easy_cross_correlation(self,x,y):

        t = zip(x, y)
        L = np.array(list(t))
        s = np.sum(L[:, 0] * L[:, 1])
        L1 = np.sum(np.square(L[:, 0]))
        L2 = np.sum(np.square(L[:, 1]))
        if s == 0:
            return 0
        else:
            return s / (L1 ** 0.5) / (L2 ** 0.5)
    def max_cross_correlation(self):
        cor=[]
        for i in range(1, 60):
            cor.append(self.easy_cross_correlation(self.x[i:], self.y))
        for i in range(0, 60):
            cor.append(self.easy_cross_correlation(self.x, self.y[i:]))
        self.bestShiftX=59-np.argmax(cor)
        return max(cor)

    def pearson_coefficient(self,x,y):
        Z=list(zip(x,y))
        x=[e[0] for e in Z]
        y=[e[1] for e in Z]
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        dx=x-x_mean
        dy=y-y_mean
        x_b= np.sum(np.square(dx))**0.5
        y_b= np.sum(np.square(dy))**0.5
        cor=np.sum(dx*dy)/(1+x_b+y_b)
        return cor
    def max_pearson_coefficient(self):
        cor=[]
        for i in range(1, 60):
            cor.append(self.pearson_coefficient(self.x[i:], self.y))
        for i in range(0, 60):
            cor.append(self.pearson_coefficient(self.x, self.y[i:]))
        self.bestShiftX=60-np.argmax(cor)
        return max(cor)
    def easy_linear_correlation(self,X,y):
        Z=list(zip(X,y))
        X=[e[0] for e in Z]
        y=[e[1] for e in Z]
        reg=linear_model.LinearRegression()
        reg.fit(X,y)
        return reg.score(X,y)
    def max_linear_correlation(self):
        cor=[]
        for i in range(1, 60):
            X=[[e] for e in self.x[i:]]
            cor.append(self.easy_linear_correlation(X, self.y))
        X0=[[e] for e in self.x]
        for i in range(0, 60):
            cor.append(self.easy_linear_correlation(X0, self.y[i:]))
        self.bestShiftX=60-np.argmax(cor)
        return max(cor)


if __name__=="__main__":
    import CommonOperation
    import KPI_Loader
    import pandas as pd
    import Generator
    import os

    method = "dtw"
    subdir = "10080"
    outputdir = os.path.join("output", subdir, method)

    src_dir=os.path.join("data_generated", "10080")
    ss=["combined_1","combined_0",]

    kpi=KPI_Loader.read_kpi(src_dir=src_dir,sampleSize=0,needed=ss)
    kpi=KPI_Loader.preprocess(kpi)
    for k in kpi:
        n=Generator.smooth_flattenY(np.array(kpi[k]),smooth_window=10)
        n=pd.DataFrame(n,columns=["timestamp","score","value","anomaly"])
        kpi[k]=n

    import SimilarityCalculation
    sim_dict=SimilarityCalculation.calc_similarity(kpi,"dtw",default_value=1)
    import SimilarityAnalyzer

    sim_dicts=SimilarityAnalyzer.select_top_similar(sim_dict,len(ss))
    SimilarityAnalyzer.plot_similarity(kpi,sim_dicts,len(ss),os.path.join("Ignore","view"))


    #from dtaidistance import dtw_visualisation as dtwvis
    #from dtaidistance import dtw
    #d, paths = dtw.warping_paths_fast(s1, s2, window=1000,max_step=2,penalty=0.1,psi=500)
    #print(d)
    '''best_path = dtw.best_path(paths)
    dtwvis.plot_warpingpaths(s1, s2, paths, best_path,"out.png")
    import cv2
    img=cv2.imread("out.png")
    cv2.imshow("test",img)
    cv2.waitKey()'''


