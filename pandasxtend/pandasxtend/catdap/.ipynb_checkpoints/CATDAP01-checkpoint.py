import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

__all__ = [
    "CATDAP01",
]

#対象データの作成
#目的変数がカテゴリの場合はそのまま使用
#対象データの作成
#目的変数がカテゴリの場合はそのまま使用
class CATDAP01():
    def __init__(self,y_type = "class",X_bins = 5,y_bins = 5,X_bin_closed = "left",adj_parametor = 1e-08):
        self.y_type = y_type
        self.X_bins = X_bins
        self.y_bins = y_bins
        self.X_bin_closed = X_bin_closed
        self.adj_parametor = adj_parametor

    def fit(self,X,y):
        feature_list = X.columns
        y_type = self.y_type
        X_bins = self.X_bins
        X_bin_closed = self.X_bin_closed
        y_bins = self.y_bins
        adj_parametor = self.adj_parametor
    
        def calc_aic_feat(feature):
            def make_DF(feature):
                #def make_DF(feature):
                if y_type == "class":
                    DF = pd.DataFrame(data = {"feature":X[feature],"label":y})

                #目的変数が数値の場合はの場合はそのまま使用
                elif y_type == "numeric":
                    if y_bins == None:
                        print("If y_type is class ,set integer into y_bins")
                    else :
                        #数値の分割
                        y_num = pd.qcut(y,y_bins,duplicates = "drop")
                        #数値の大小でソートして、ラベルを大小順にわかるようにふよ
                        y_cut = list(y_num.unique().sort_values().astype(str))
                        cardinality_y = len(y_cut)

                        for i in range(cardinality_y):
                            #とりあえず99分割までは対応させる。
                            i0 = str(i + 1).zfill(len(str(cardinality_y)))
                            y_num = pd.Series(y_num.astype(str))
                            y_num.replace({y_cut[i] : i0 + "_" + y_cut[i]},inplace = True)

                        DF = pd.DataFrame(data = {"feature":X[feature],"label":y_num})
                        DF["label"] = DF["label"].astype(str)

                return DF

            def preprocess(DF):
                #欠損データ、非欠損データに分割
                DF_miss = DF[pd.isna(DF["feature"]) == True].copy()
                DF_nmiss = DF[pd.isna(DF["feature"]) == False].copy()

                #変数が時間の場合、数値に修正する
                if str(DF_nmiss["feature"].dtype).find("[ns]") > -1:
                    DF_nmiss["feature"] = DF_nmiss["feature"].dt.year * 10000 + DF_nmiss["feature"].dt.month * 100 + DF_nmiss["feature"].dt.day

                #カテゴリ変数はそのままカテゴリとして採用
                if DF_nmiss["feature"].dtype == "object":
                    DF_nmiss["category"] = DF_nmiss["feature"]
                    cat_list = list(DF_nmiss["category"].unique())
                    cat_list.sort()

                    cardinality_ = len(cat_list)
                    cat_dict = dict(zip(
                        [cat_list[i] for i in range(cardinality_)], 
                        [str(i+1).zfill(len(str(cardinality_))) + ": " + str(cat_list[i]) for i in range(cardinality_)])
                    )
                    DF_nmiss["category"].replace(cat_dict,inplace = True)

                else :
                    try :
                        qcut_ = int(100 / X_bins)
                        p_list = [p for p in range(0,100,qcut_)]
                        p_list.extend([100])

                        q_list = [np.percentile(DF_nmiss["feature"],q) for q in p_list]

                        if X_bin_closed == "left":
                            q_list[-1] = q_list[-1] + 1
                            DF_nmiss["category"] = pd.cut(DF_nmiss["feature"],bins = q_list,right = False)
                        elif X_bin_closed == "right":
                            q_list[0] = q_list[0] - 1
                            DF_nmiss["category"] = pd.cut(DF_nmiss["feature"],bins = q_list,right = True)

                    #基本50分割でセット。それで分割できなければ生の値でいい
                    except :
                        DF_nmiss["category"] = DF_nmiss["feature"]

                    cat_list = list(DF_nmiss["category"].unique())
                    cat_list.sort()

                    cardinality_ = len(cat_list)
                    #ラベルふり
                    cat_dict = dict(zip(
                        [cat_list[i] for i in range(cardinality_)], 
                        [str(i+1).zfill(len(str(cardinality_))) + ": " + str(cat_list[i]) for i in range(cardinality_)])
                    )
                    #cat_transform_list = dict(zip(cat_list,q_list))

                    DF_nmiss["category"].replace(cat_dict,inplace = True)

                #欠損先はカテゴリを欠損で埋める
                DF_miss["category"] = str(0).zfill(len(str(cardinality_))) + ": missing"

                #欠損先と入力先を縦積み
                DF2 = pd.concat([DF_nmiss,DF_miss],axis = 0)

                missing_rate = len(DF_miss)/(len(DF_nmiss) + len(DF_miss))
                return DF2,missing_rate


            def make_cross_table(DF2):

                #AICを計算するためのクロス表を作成
                _cross = pd.crosstab(DF2["category"],DF2["label"],margins=None,dropna = False)
                #該当先がいない場合、NAになるので0埋め
                _cross = _cross.fillna(0)

                n_feature , n_label = _cross.shape
                #全体件数での対数尤度。log内が0だと計算できないので閾値を足してやる

                return _cross,n_feature,n_label

            def calc_term2(cross_table):
                term2 = (cross_table.sum().sum()) * np.log(cross_table.sum().sum() + adj_parametor)
                return term2

            def calc_term1_0(cross_table,n_feature,n_label):
                #AIC0 =====================================================================================
                #列単位での和
                i_list = [(cross_table.iloc[i,:].sum()) * np.log(cross_table.iloc[i,:].sum() + adj_parametor) for i in range(0,n_feature)]

                #行単位での和
                j_list = [(cross_table.iloc[:,j].sum()) * np.log(cross_table.iloc[:,j].sum() + adj_parametor) for j in range(0,n_label)]

                term1_0 = sum(i_list) + sum(j_list)
                return term1_0


            def calc_term1_1(cross_table,n_feature,n_label):
                #AIC1 =====================================================================================
                yudo_matrix = np.zeros([n_feature ,n_label])

                for i in range(0,n_feature):
                    for j in range(0,n_label):
                        yudo_matrix[i,j] = (cross_table.iloc[i,j]) * np.log(cross_table.iloc[i,j] + adj_parametor)

                term1_1 = yudo_matrix.sum()

                return term1_1

            def calc_aic0(term1_0,term2,n_feature,n_label):
                AIC0 = -2 * (term1_0 - 2 * term2) + 2 * (n_feature + n_label - 2)
                return AIC0

            def calc_aic1(term1_1,term2,n_feature,n_label):
                AIC1 = -2 * (term1_1 - term2) + 2 * (n_feature * n_label - 1)
                return AIC1

            def calc_aic_cat(AIC1,AIC0):
                #CATのAIC ==================================================================================
                AIC_CAT = AIC1 - AIC0
                return AIC_CAT

            def make_output_cross_table(DF2):
                output_cross_table = pd.crosstab(DF2["category"],DF2["label"],margins=True,dropna = False)
                output_cross_table = output_cross_table.fillna(0)
                return output_cross_table

            def make_aic_table(output_cross_table,feature,missing_rate,AIC_CAT):
                #各クラスの構成比を計算
                cardinality_label = len(output_cross_table.columns) - 1

                ratio_label = [col for col in output_cross_table.columns if col != "All"]
                ratio_table = pd.concat(
                    [round(output_cross_table.loc[:,label_]/output_cross_table.loc[:,"All"],5) for label_ in ratio_label],
                    axis = 1)
                ratio_table.columns = ["Ratio_" + str(label_) for label_ in ratio_label]

                output_cross_table2 = output_cross_table.join(ratio_table)

                base = pd.DataFrame(index = [i for i in range(output_cross_table2.shape[0])])
                base["feature_name"] = feature
                base["missing_rate"] = round(missing_rate,5)
                base["AIC"] = round(AIC_CAT,5)

                AIC_table = pd.concat([base,output_cross_table2.reset_index()],axis = 1)

                return AIC_table

            DF = make_DF(feature)
            #print("make df end")
            DF2,missing_rate = preprocess(DF)
            #print("preprocess end")
            cross_table,n_feature,n_label = make_cross_table(DF2)

            term2 = calc_term2(cross_table)
            term1_0 = calc_term1_0(cross_table,n_feature,n_label)
            term1_1 = calc_term1_1(cross_table,n_feature,n_label)

            AIC0 = calc_aic0(term1_0,term2,n_feature,n_label)
            AIC1 = calc_aic1(term1_1,term2,n_feature,n_label)
            #print(AIC0,term1_0,term2)
            #print(AIC1,term1_1,term2)
            AIC_CAT = calc_aic_cat(AIC1,AIC0)

            output_cross_table = make_output_cross_table(DF2)
            AIC_table = make_aic_table(output_cross_table,feature,missing_rate,AIC_CAT)

            return AIC_table

        aic_chunk = joblib.Parallel(n_jobs = -1)(joblib.delayed(calc_aic_feat)(feat) for feat in feature_list)
        AIC_table = pd.concat(aic_chunk,axis = 0)
        AIC_table.sort_values(["AIC","feature_name","category"],ascending = [True,True,True],inplace = True)
        AIC_table.reset_index(drop = True,inplace = True)
        AIC_summary = AIC_table[["feature_name","AIC"]].drop_duplicates()
        AIC_summary.reset_index(drop = True,inplace = True)

        self.AIC_table = AIC_table
        self.AIC_summary = AIC_summary
        print("AIC calculation is finished")

    def to_csv(self,output_dir = "./",file_name = "temp"):
        AIC_table = self.AIC_table
        AIC_summary = self.AIC_summary 
        AIC_table.to_csv(output_dir +"CATDAP01_table_{}.csv".format(file_name))
        AIC_summary.to_csv(output_dir +"CATDAP01_summary_{}.csv".format(file_name))
        
    def plot(self,yoko = 4):
        AIC_table = self.AIC_table
        AIC_summary = self.AIC_summary 

        feature_list = AIC_summary["feature_name"].copy()
        num_feautre = len(feature_list)
        
        if (num_feautre % yoko) != 0:
            tate = (num_feautre // yoko) + 1
        else :
            tate = (num_feautre // yoko)

        fig, ax = plt.subplots(tate,yoko,tight_layout = True,figsize = (yoko * 5,tate * 5))

        col_ratio = [col for col in AIC_table.columns if str(col).find("Ratio") > -1]
        
        for num,col_ in enumerate(feature_list):
            temp = AIC_table[(AIC_table["feature_name"] == col_) & (AIC_table["category"] != "All")].set_index("category")[col_ratio]
            AIC_value = AIC_summary[AIC_summary["feature_name"] == col_]["AIC"].values[0]
            
            try:
                i,j = num // yoko,num % yoko  
                if tate == 1:
                    temp.plot(kind = "bar",stacked = True,title = col_ + " AIC:" + str(AIC_value),ax=ax[j],cmap = "coolwarm",ec = "black")
                    #ax[j].grid(axis = "y")
                    
                elif yoko == 1:
                    temp.plot(kind = "bar",stacked = True,title = col_ + " AIC:" + str(AIC_value),ax=ax[i],cmap = "coolwarm",ec = "black")
                    #ax[i].grid(axis = "y")
                    
                else :
                    temp.plot(kind = "bar",stacked = True,title = col_ + " AIC:" + str(AIC_value),ax=ax[i][j],cmap = "coolwarm",ec = "black")
                    #ax[i][j].grid(axis = "y")

            except :
                None

        plt.show()

        
        
        

        