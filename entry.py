import os

import CommonOperation
import KPI_Loader
import SimilarityCalculation
import SimilarityAnalyzer

def sample_larger(enlarge=10,method="dtw",subdir="10080",doAnalyze=True):
    srcdir = os.path.join("data_generated", subdir)
    outputdir = os.path.join("output", subdir, method)
    similarity_dict_path = os.path.join(outputdir, "similarity_dict.dat")
    similar_top = 10

    default_similarity_dict = CommonOperation.pickleLoad(similarity_dict_path, dict())

    kpi = KPI_Loader.read_kpi(srcdir, enlarge, needed=default_similarity_dict.keys())
    kpi = KPI_Loader.preprocess(kpi)

    similarity_dict = SimilarityCalculation.calc_similarity(kpi, method, (0.5, 0.5), default_similarity_dict,default_value=1)
    CommonOperation.pickleDump(similarity_dict_path, similarity_dict)
    CommonOperation.writeObjectList(similarity_dict.keys(), os.path.join(outputdir, "USEDFILES.txt"))

    if (doAnalyze):
        sorted_value_dict, sorted_score_dict, sorted_dict = SimilarityAnalyzer.select_top_similar(similarity_dict,
                                                                                                  similar_top=similar_top,
                                                                                                  smaller_first=False)

        sorted_lists = [sorted_value_dict, sorted_score_dict, sorted_dict]
        #print(similarity_dict)
        #print(sorted_lists)
        SimilarityAnalyzer.plot_similarity(kpi, sorted_lists, similar_top, outputdir)

if __name__ == "__main__":
    method="dtw"
    batch_size="10080"
    for i in range(50):
        sample_larger(2,method,batch_size,False)
    sample_larger(None,method,batch_size,True)