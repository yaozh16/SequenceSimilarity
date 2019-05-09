#coding=utf-8

import time

from Similarity import Similarity
import numpy as np

def calc_similarity(kpi, similarity_method = 'ecc', weights = (0.5, 0.5),default_dict=dict(),default_value=0):
    # 两两算相似度, Similarity(score) * weights[0] + Similarity(value) * weights[1] 作为结果
    '''k1 = random.sample(list(kpi.keys()), 1)[0]
    v1 = kpi[k1]
    similarity_dict=dict()
    similarity_dict[k1]=dict([[tt, []] for tt in kpi.keys()])
    '''
    similarity_dict = default_dict if  isinstance(default_dict,dict) else dict()

    for k1, v1 in kpi.items():
        for k2, v2 in kpi.items():
            if(not k1 in similarity_dict.keys()):
                similarity_dict[k1]=dict()
            if(not k2 in similarity_dict.keys()):
                similarity_dict[k2]=dict()
            if k1 == k2:
                similarity_dict[k1][k1] = [default_value, default_value, default_value]
            elif k2 in similarity_dict.keys() and k1 in similarity_dict[k2]:
                continue
            else:
                T0=time.time()
                print([k1,k2],end=" ")
                s1 = Similarity(v1.value.values, v2.value.values).use_method(similarity_method)
                s2 = Similarity(v1.score.values, v2.score.values).use_method(similarity_method)
                print("[{1},{2}]:{0}".format(time.time()-T0,s1,s2),flush=True)
                similarity_dict[k1][k2] = [s1, s2, s1 * weights[0] + s2 * weights[1]]
                similarity_dict[k2][k1] = [s1, s2, s1 * weights[0] + s2 * weights[1]]
    print('finish calculate similarity. current dict size reaches {0} ({1})'.format(len(similarity_dict.keys()),time.asctime()),flush=True)
    return similarity_dict


if __name__=="__main__":
    pass

