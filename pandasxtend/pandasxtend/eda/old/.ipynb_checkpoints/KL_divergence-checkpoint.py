import numpy as np
import pandas as pd

__all__ = [
    "KL_div"
]


def KL_div(BASE,COMP,feat,cut = 10,thre = 0.0000000000001):
    """
    input/
    BASE:BASE_DF,COMP:COMP_DF,feat:compare base with comp in feature 
    """
    #calc max min in the base comp data
    max_min = pd.concat([BASE[feat],COMP[feat]],axis = 0)
    
    sup = max_min.max()
    inf = max_min.min()
    
    #set the cut point
    q_list = []
    
    for i in range(0,cut + 1,1):
        q_point = (sup - inf) * i/cut + inf

    # first cut
    #try :
    #	for i in range(0,cut + 1,1):
    #		#make points of every percentage of feature
    #		q_point = (sup - inf) * i/cut + inf
    
    # second cut
    #except :
    #	print(" can not cut the points ... ")
    #	for i in range(0,4,1):
    #		#make points of every percentage of feature
    #		q_point = (sup - inf) * i/3 + inf
    
    q_list.extend([q_point])
    
    #calc distribution by base,comp dataset
    #base
    base = pd.cut(BASE[feat],bins = q_list)
    temp1 = base.value_counts().sort_index()/base.count()
    base_df = pd.DataFrame(index = temp1.index,data = temp1.values,columns = ["BASE"])
    
    #comp
    comp = pd.cut(COMP[feat],bins = q_list)
    temp2 = comp.value_counts().sort_index()/comp.count()
    comp_df = pd.DataFrame(index = temp2.index,data = temp2.values,columns = ["COMP"])
    
    dist_df = base_df.join(comp_df)
    
    #treatment 0 divide case
    dist_df += thre
    
    # return KL divergence
    return np.sum(dist_df["BASE"] * np.log(dist_df["BASE"]/dist_df["COMP"]))