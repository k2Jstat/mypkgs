class CATDAP02:
    def __init__(self,base_feature,X_bins = 5,y_type = "class",y_bins = 5,thresholds_category_count = 50,adj = 1e-10):
        """
        y_type select following type ["class","numeric"]
        
        """
        self.X_bins = X_bins
        self.y_type = y_type
        self.y_bins = y_bins
        self.thresholds_category_count = thresholds_category_count
        self.adj = adj
        self.base_feat = base_feature

    def fit(self,X,y):
        """
        
        """
        import numpy as np
        import pandas as pd
        from tqdm import tqdm
        
        X_bins_ = self.X_bins
        y_type_ = self.y_type
        y_bins_ = self.y_bins
        thresholds_category_count_ = self.thresholds_category_count
        adj = self.adj
        base_feat = self.base_feat

        AIC_LIST = []
        CROSS_DICTIONARY = {}
        HM_DIC = {}
        
        feature_list = list(X.columns)
        feature_list.remove(base_feat)
        #feature_list2 = list(feature_list).copy()
        #feature_list2 = list(feature_list).copy()

        # AICを計算　*******************
        #特徴量を順々に回す
        for feature in tqdm(feature_list):

            #分割集計
            if y_type_ == "class":
                DF = pd.DataFrame(data = {"base_feature":X[base_feat],"add_feature":X[feature],"label":y})
            elif y_type_ == "numeric":
                if y_bins_ == None:
                    print("If y_type is class ,set integer into y_bins")
                else :
                    #数値の分割
                    y_num = pd.qcut(y,y_bins_,duplicates = "drop")
                    #数値の大小でソートして、ラベルを大小順にわかるようにふよ
                    labels_cut = list(y_num.unique().sort_values().astype(str))
                    for i in range(0,len(labels_cut)):
                        #とりあえず99分割までは対応させる。
                        i0 = str((101 + i))[1:3]
                        y_num = y_num.astype(str)
                        y_num.replace({labels_cut[i] : i0 + "_" +  labels_cut[i]},inplace = True)
                    #DF = pd.DataFrame(data = {"feature":X[feature],"label":y_num}) 
                    DF = pd.DataFrame(data = {"base_feature":X[base_feat],"add_feature":X[feature],"label":y_num})


                    
            
            DF["label"] = DF["label"].astype(str)

            #欠損データ、非欠損データに分割
            DF_nmiss = DF[pd.isna(DF["base_feature"]) == False].copy()

            #変数が時間の場合、数値に修正してやる
            if str(DF_nmiss["base_feature"].dtype).find("[ns]") > -1:
                DF_nmiss["base_feature"] = DF_nmiss["base_feature"].dt.year * 10000 + DF_nmiss["base_feature"].dt.month * 100 + DF_nmiss["base_feature"].dt.day

            #カテゴリ変数はそのままカテゴリとして採用
            if DF_nmiss["base_feature"].dtype == "object":
                if len(DF_nmiss["base_feature"].unique()) >= thresholds_category_count_:
                    #DF_nmiss["category"] = "all"
                    DF_nmiss[base_feat] = "Integrated"
                else :
                    #DF_nmiss["category"] = DF_nmiss["feature"]
                    DF_nmiss[base_feat] = DF_nmiss["base_feature"]

            #数値データはビン化
            else :
                try :
                    #DF_nmiss["category"] = pd.qcut(DF_nmiss["feature"],
                    DF_nmiss[base_feat] = pd.qcut(DF_nmiss["base_feature"],X_bins_,duplicates = "drop")
                #基本10分割でセット。それで分割できなければもう生の値でいい
                except :
                    if len(DF_nmiss["base_feature"].unique()) >= thresholds_category_count_:
                        #DF_nmiss["category"] = "all"
                        DF_nmiss[base_feat] = "Integrated"
                    else :
                        #DF_nmiss["category"] = DF_nmiss["feature"]
                        DF_nmiss[base_feat] = DF_nmiss["base_feature"]
            #else :
            #    print("予想外の事態")

            #欠損先はカテゴリを欠損で埋める
            DF_miss = DF[pd.isna(DF["base_feature"]) == True].copy()
            DF_miss[base_feat] = "00: Deficiency"  

            #欠損先と入力先を縦積み
            DF2 = pd.concat([DF_nmiss,DF_miss],axis = 0)

            #feat2についても同様の処理　**********************************
            #欠損データ、非欠損データに分割
            DF2_miss = DF2[pd.isna(DF2["add_feature"]) == True].copy()
            DF2_nmiss = DF2[pd.isna(DF2["add_feature"]) == False].copy()

            #変数が時間の場合、数値に修正してやる
            if str(DF2_nmiss["add_feature"].dtype).find("[ns]") > -1:
                DF2_nmiss["add_feature"] = DF2_nmiss["add_feature"].dt.year * 10000 + DF2_nmiss["add_feature"].dt.month * 100 + DF2_nmiss["add_feature"].dt.day

            #カテゴリ変数はそのままカテゴリとして採用
            if DF2_nmiss["add_feature"].dtype == "object":
                if len(DF2_nmiss["add_feature"].unique()) >= thresholds_category_count_:
                    #DF2_nmiss["category2"] = "all"
                    DF2_nmiss[feature] = "Integrated"
                else :
                    #DF2_nmiss["category2"] = DF2_nmiss["feature2"]
                    DF2_nmiss[feature] = DF2_nmiss["add_feature"]

            #数値データはビン化
            else :
                try :
                    #DF2_nmiss["category2"] = pd.qcut(DF2_nmiss["feature2"],X_bins,duplicates = "drop")
                    DF2_nmiss[feature] = pd.qcut(DF2_nmiss["add_feature"],X_bins_,duplicates = "drop")
                #基本10分割でセット。それで分割できなければ生の値でいい
                except :
                    if len(DF2_nmiss["add_feature"].unique()) >= thresholds_category_count_:
                        #DF2_nmiss["category2"] = "all"
                        DF2_nmiss[feature] = "Integrated"
                    else :
                        #DF2_nmiss["category2"] = DF_nmiss["feature2"]
                        DF2_nmiss[feature] = DF_nmiss["add_feature"]

            #欠損先はカテゴリを欠損で埋める
            #DF2_miss["category2"] = "欠損"  
            DF2_miss[feature] = "00: Deficiency"  


            #欠損先と入力先を縦積み
            DF3 = pd.concat([DF2_nmiss,DF2_miss],axis = 0)     

            #独立と仮定しない場合のAIC
            #AICを計算するためのクロス表を作成

            #全体件数での対数尤度。log内が0だと計算できないので0.00000000001を足してやる
            #_term2 = (cross_base.sum().sum()) * np.log(cross_base.sum().sum() + adj)
            #AIC0 =====================================================================================
            cross_base = pd.crosstab(DF3[base_feat],DF3["label"],margins=None,dropna = False)
            cross_base = cross_base.fillna(0)
            n_all = cross_base.sum().sum()

            n_feat_b , n_lab_b = cross_base.shape

            temp0 = 0

            for i in range(n_feat_b):
            #print(i)
                for j in range(n_lab_b):
                    #print(j)
                    n_base = cross_base.iloc[i,j]
                    #立て
                    n_sum_col = cross_base.iloc[i,:].sum()
                    n_sum_row = cross_base.iloc[:,j].sum()  

                    temp0 += n_base * np.log((n_base * n_all + adj)/(n_sum_col * n_sum_row + adj))

            AIC0 = -2 * temp0 + 2 * (n_feat_b - 1) * (n_lab_b - 1)

            #AIC1 =====================================================================================
            #cross_comp = pd.crosstab([DF3["category"],DF3["category2"]],DF3["label"],margins=None,dropna = False)
            cross_comp = pd.crosstab([DF3[base_feat],DF3[feature]],DF3["label"],margins=None,dropna = False)
            cross_comp = cross_comp.fillna(0)

            n_feat_c , n_lab_c = cross_comp.shape

            #独立と仮定した場合のAIC
            temp1 = 0

            for i in range(n_feat_c):
                #print(i)
                for j in range(n_lab_c):
                    #print(j)
                    n_base = cross_comp.iloc[i,j]
                    #立て
                    n_sum_col = cross_comp.iloc[i,:].sum()
                    n_sum_row = cross_comp.iloc[:,j].sum()

                    temp1 += n_base * np.log((n_base * n_all + adj)/(n_sum_row * n_sum_col + adj))
                    #print(temp1)

            AIC1 = -2 * temp1 + 2 * (n_lab_c - 1) * (n_feat_c - 1)

            #CATのAIC ==================================================================================
            AIC_CAT = AIC1 - AIC0 

            #情報の入力率
            rate_fill = len(DF_nmiss)/(len(DF_nmiss) + len(DF_miss))
            rate_fill2 = len(DF2_nmiss)/(len(DF2_nmiss) + len(DF2_miss))

            #クロス表
            #_cross2 = pd.crosstab([DF3["category"],DF3["category2"]],DF3["label"],margins=True,dropna = False)
            _cross2 = pd.crosstab([DF3[base_feat],DF3[feature]],DF3["label"],margins=True,dropna = False)

            #各クラスの構成比を計算
            #loop = len(_cross2.columns) - 1
            #for i in range(0,loop):
            #    _cross2["ratio_" + str(_cross2.columns[i])] = _cross2.iloc[:,i]/_cross2["All"]

            #計算したAICをリストに保存
            _OUT = [base_feat,feature,AIC_CAT,rate_fill * rate_fill2]
            AIC_LIST.append(_OUT)
            CROSS_DICTIONARY[base_feat + " " +feature] = _cross2

            tt = DF3.copy()
            tt["label"] = tt["label"].astype(int)

            hm = tt.pivot_table(
                index = [base_feat],
                columns = [feature],
                values = "label",
                aggfunc = "mean",
                dropna = False
            ).fillna(0)
            HM_DIC[base_feat + " " +feature] = hm


        #資料を保存
        AIC_LIST2 = pd.DataFrame(AIC_LIST)
        AIC_LIST2.columns = ["base_feature","add_feature","AIC","RATIO_FILL"]
        AIC_LIST3 = AIC_LIST2.sort_values(["AIC"])
        
        self.AIC_LIST = AIC_LIST3
        self.HM_DIC = HM_DIC
        self.CROSS_DICTIONARY = CROSS_DICTIONARY
        self.base_feat = base_feat

    def CrossTable_toFile(self,crosstab_file_name = "CrossTab_AIC",heatmap_file_name = "HeatMap_AIC",path_output = "./",format_ = "csv"):
        """
        input/file_name,path_output
        example/CATDAP().CrossTable_toExcel(file_name = "temp.xls",path_output = "../output/")
        """
        AIC_LIST_cross = self.AIC_LIST
        HM_DIC_cross = self.HM_DIC
        cross_dic = self.CROSS_DICTIONARY
        base_feat = self.base_feat

        import pandas as pd
        from tqdm import tqdm

        #heatmatp作成
        base_cr = pd.DataFrame()
        base_hm = pd.DataFrame()

        for i in tqdm(range(len(AIC_LIST_cross))):
            #AICのクロス表
            feature_set = ["base_feature","add_feature"]
            yyy = pd.DataFrame(data = [AIC_LIST_cross.iloc[i,][feature_set]])
            yyy.set_index(feature_set,inplace = True)
            yyy["AIC"] = AIC_LIST_cross.iloc[i,]["AIC"]
            yyy["RATIO_FILL"] = AIC_LIST_cross.iloc[i,]["RATIO_FILL"]

            #chose_key = yyy.index[0][0] + " " + yyy.index[0][1]
            #
            #xxx = cross_dic[AIC_LIST_cross.iloc[i,]]
            chose_key = AIC_LIST_cross[(AIC_LIST_cross["base_feature"] == yyy.index[0][0]) & 
                                       (AIC_LIST_cross["add_feature"] == yyy.index[0][1])][feature_set].copy()
            #chose_key = AIC_LIST_cross[(AIC_LIST_cross["feature"] == yyy.index[0][0]) & (AIC_LIST_cross["feature2"] == yyy.index[0][1])].copy()
            #select_ = (chose_key["feature"] + " " + chose_key["feature2"]).astype(str)
            select_ = (chose_key["base_feature"] + " " + chose_key["add_feature"]).astype(str)
            sel_list = list(select_)
            xxx = cross_dic[sel_list[0]].copy()
            #columns_flg = ["flg_" + str(i) for i in range()]

            #各クラスの構成比を計算
            loop = len(xxx.columns) - 1
            for i in range(0,loop):
                xxx["Ratio_" + str(xxx.columns[i])] = xxx.iloc[:,i]/xxx["All"]

            #xxx.columns = ["flg_0","flg_1","All"]

            temp = pd.concat([yyy,xxx],axis = 0)#.fillna(" ")
            #temp = temp[["AIC","RATIO_FILL","flg_0","flg_1","All"]]

            base_cr = pd.concat([base_cr,temp],axis = 0)

            # *** 
            xxx2 = HM_DIC_cross[sel_list[0]]

            xxx2.columns = xxx2.columns.astype(str)
            xxx2.index = xxx2.index.astype(str)

            xxxT = xxx2.T.reset_index().T.copy()

            columns_bin = ["bin_" + str(i) for i in range(1,xxxT.shape[1] + 1)]
            xxxT.columns = columns_bin 

            temp_hm = pd.concat([yyy,xxxT],axis = 0)#.fillna(" ")
            base_hm = pd.concat([base_hm,temp_hm],axis = 0)


            base_cr.fillna("",inplace = True)
            base_hm.fillna("",inplace = True)
            
            #吐き出し
            if format_ == "csv":
                #base_cr.to_csv(path_output + file_name + "." + format_)
                base_cr.to_csv(path_output + crosstab_file_name + "_" + str(base_feat) + "." + format_)
                base_hm.to_csv(path_output + heatmap_file_name + "_" + str(base_feat) + "." + format_)

            elif format_ == "xls":
                base_cr.to_excel(path_output + crosstab_file_name + "_" + str(base_feat) + "." + format_)
                base_hm.to_excel(path_output + heatmap_file_name + "_" + str(base_feat) + "." + format_)
