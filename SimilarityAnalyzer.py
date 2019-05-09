import os

import matplotlib.pyplot as plt

import CommonOperation
import KPI_Loader

def select_top_similar(similarity_dict,similar_top = 5, similar_threshold = -0.1 , smaller_first=False,ignorePattern=None):
    # 对每一条时间序列，都选取出他的最相似 top k (top of value, score and both)
    sorted_value_dict = {}
    sorted_score_dict = {}
    sorted_dict = {}
    for k, v in similarity_dict.items():
        directionFactor=1 if  smaller_first else -1
        filtered=[t for t in v.items() if ignorePattern==None or (not (ignorePattern in t[0])) or (k in t[0])]
        sorted_value_s = sorted(filtered, key=lambda kv: directionFactor*kv[1][0])
        sorted_value_dict[k] = [(t[0], t[1][0]) for t in sorted_value_s[:similar_top] if t[1][0] > similar_threshold]
        sorted_score_s = sorted(filtered, key=lambda kv: directionFactor*kv[1][1])
        sorted_score_dict[k] = [(t[0], t[1][1]) for t in sorted_score_s[:similar_top] if t[1][1] > similar_threshold]
        sorted_s = sorted(filtered, key=lambda kv: directionFactor* kv[1][2])
        sorted_dict[k] = [(t[0], t[1][2]) for t in sorted_s[:similar_top] if t[1][2] > similar_threshold]
    return  sorted_value_dict,sorted_score_dict,sorted_dict

def plot_similarity(kpi, dict_list, similar_top, output_dir):
    print("plot similarity",flush=True)
    CommonOperation.checkDirectory(output_dir)
    # 绘制他们的原始数据以及异常点
    dict_name = ['value_similarity', 'anomaly_similarity', 'combine_similarity']
    for i in range(len(dict_list)):
        if (not os.path.exists(os.path.join(output_dir, dict_name[i]))):
            os.mkdir(os.path.join(output_dir, dict_name[i]))
        for k, v in dict_list[i].items():
            print(k,flush=True)
            graph_row=similar_top+1
            plt.figure(figsize=(20, 20))
            colorlist=['purple','b','g']
            plt.title(k)
            for index,similar_i in enumerate(v):
                color = colorlist[index % len(colorlist)]
                name=similar_i[0]
                current_data=kpi[name]
                plt.subplot(graph_row, 1, index+1)
                if (dict_name[i].startswith("value")):
                    plt.plot(current_data.timestamp,  current_data.value)
                    plt.scatter( current_data.loc[ current_data.anomaly == 1, 'timestamp'], current_data.loc[kpi[name].anomaly == 1, 'value'], c='r')
                elif (dict_name[i].startswith("anomaly")):
                    plt.plot( current_data.timestamp,  current_data.score)
                    plt.scatter( current_data.loc[ current_data.anomaly == 1, 'timestamp'], current_data.loc[kpi[name].anomaly == 1, 'score'], c='r')
                elif (dict_name[i].startswith("combine")):
                    plt.plot( current_data.timestamp,  current_data.score, color="g")
                    plt.plot( current_data.timestamp,  current_data.value, color="b")
                    plt.scatter( current_data.loc[ current_data.anomaly == 1, 'timestamp'], current_data.loc[kpi[name].anomaly == 1, 'score'], c='r')
                plt.title(similar_i[0] + ': ' + str(similar_i[1]),color=color)
            plt.savefig(os.path.join(output_dir, dict_name[i], k + '.png'))
            plt.close()

def plot_all(kpi,dict_list,filter_key,output_dir,graph_row=6):
    print("plot similarity", flush=True)
    CommonOperation.checkDirectory(output_dir)
    # 绘制他们的原始数据以及异常点
    dict_name = ['value_similarity', 'anomaly_similarity', 'combine_similarity']
    for i in range(len(dict_list)-1,0,-1):
        CommonOperation.checkDirectory(os.path.join(output_dir, dict_name[i]))
        for k, v in dict_list[i].items():
            if(not filter_key in  k):
                continue
            CommonOperation.checkDirectory(os.path.join(output_dir, dict_name[i],k))
            print("[{1}]key:{0}".format(k,dict_name[i]),flush=True)
            plt.figure(figsize=(40, 20))
            colorlist=['b','darkcyan','g','darkblue']
            g_index=0
            for index,similar_i in enumerate(v):
                if(index==0):
                    continue
                g_index=(index-1)%graph_row
                name=v[0][0]
                current_data=kpi[name]
                color='r'
                plt.subplot(graph_row, 2, 4*int(g_index/2) + 1 +(g_index%2))
                if (dict_name[i].startswith("value")):
                    plt.plot(current_data.timestamp,  current_data.value,label="value",color=color)
                    plt.scatter( current_data.loc[ current_data.anomaly == 1, 'timestamp'], current_data.loc[kpi[name].anomaly == 1, 'value'], c='r')
                elif (dict_name[i].startswith("anomaly")):
                    plt.plot( current_data.timestamp,  current_data.score,label="score",color=color)
                    plt.scatter( current_data.loc[ current_data.anomaly == 1, 'timestamp'], current_data.loc[kpi[name].anomaly == 1, 'score'], c='r')
                elif (dict_name[i].startswith("combine")):
                    plt.plot(current_data.timestamp,  current_data.value,label="value",color=colorlist[0])
                    #plt.plot( current_data.timestamp,  current_data.score,label="score",color=colorlist[1])
                    plt.scatter(current_data.loc[ current_data.anomaly == 1, 'timestamp'], current_data.loc[kpi[name].anomaly == 1, 'value'], c='r')
                plt.title(v[0][0] ,color=color)


                color=colorlist[index%len(colorlist)]
                name=similar_i[0]
                current_data=kpi[name]
                plt.subplot(graph_row, 2, 4*int(g_index/2) + 3 +(g_index%2))
                if (dict_name[i].startswith("value")):
                    plt.plot(current_data.timestamp,  current_data.value,color=color)
                    plt.scatter( current_data.loc[ current_data.anomaly == 1, 'timestamp'], current_data.loc[kpi[name].anomaly == 1, 'value'], c='r')
                elif (dict_name[i].startswith("anomaly")):
                    plt.plot( current_data.timestamp,  current_data.score,color=color)
                    plt.scatter( current_data.loc[ current_data.anomaly == 1, 'timestamp'], current_data.loc[kpi[name].anomaly == 1, 'score'], c='r')
                elif (dict_name[i].startswith("combine")):
                    plt.plot(current_data.timestamp,  current_data.value,label="value",color=colorlist[2])
                    #plt.plot( current_data.timestamp,  current_data.score,label="score",color=colorlist[3])
                    plt.scatter(current_data.loc[ current_data.anomaly == 1, 'timestamp'], current_data.loc[kpi[name].anomaly == 1, 'value'], c='r')
                plt.title(similar_i[0] + ': ' + str(similar_i[1]),color=color)


                if(g_index+1==graph_row):
                    plt.savefig(os.path.join(output_dir, dict_name[i], k , '{0}_{1}.png'.format(int((index-1)/graph_row)*graph_row,index-1)))
                    plt.close()
                    plt.figure(figsize=(40, 20))
            if (g_index+1!=graph_row):
                plt.savefig(os.path.join(output_dir, dict_name[i], k, '{0}_{1}.png'.format(int((index-1)/graph_row)*graph_row,index-1)))
                plt.close()

if __name__=="__main__":
    similarity_dict=CommonOperation.pickleLoad(os.path.join("output", "10080", "pcc", "similarity_dict.dat"),{})


    similar_top=-1

    sorted_value_dict, sorted_score_dict, sorted_dict=select_top_similar(similarity_dict, similar_top,ignorePattern="combine")

    method = "pcc"
    subdir = "10080"
    outputdir = os.path.join("output", subdir, method)
    kpi=KPI_Loader.read_kpi(src_dir=os.path.join("data_generated", "10080"))
    kpi=KPI_Loader.preprocess(kpi)

    plot_all(kpi,[sorted_value_dict, sorted_score_dict, sorted_dict],"combined",outputdir)