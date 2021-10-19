import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import itertools
import seaborn as sns
from tqdm import tqdm

__all__ = [
    "CATDAP02",
]


class CATDAP02():
    def __init__(self,y_type = "class",X_bins = 5,num_max_select_feature = 2,y_bins = 5,X_bin_closed = "left",adj_parametor = 1e-08):
        self.y_type = y_type
        self.X_bins = X_bins
        self.X_bin_closed = X_bin_closed 
        self.y_bins = y_bins
        self.adj_parametor = adj_parametor
        self.num_max_select_feature = num_max_select_feature
    
    def fit(self,X,y):
        y_type = self.y_type 
        X_bins = self.X_bins
        X_bin_closed = self.X_bin_closed 
        y_bins = self.y_bins
        num_max_select_feature = self.num_max_select_feature
        adj_parametor = self.adj_parametor
        
        feature_list = X.columns
        
        def make_DF(feature_select):
            #def make_DF(feature):
            if y_type == "class":
                DF = pd.DataFrame()
                col_feat = ["feature_" + str(i + 1) for i in range(len(feature_select))]
                for i,col in enumerate(col_feat):
                    DF[col] = X[feature_select[i]]

                DF["label"] = y

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

                    DF = pd.DataFrame()
                    col_feat = ["feature_" + str(i + 1) for i in range(num_select_feature)]
                    for i,col in enumerate(col_feat):
                        DF[col] = X[feature_select[i]]

                    DF["label"] = y_num
                    DF["label"] = DF["label"].astype(str)

            return DF

        def preprocess(DF,feature):
            #欠損データ、非欠損データに分割
            DF_miss = DF[pd.isna(DF[feature]) == True].copy()
            DF_nmiss = DF[pd.isna(DF[feature]) == False].copy()

            idx_start = feature.find("_")
            idx_ = feature[idx_start + 1:]

            #変数が時間の場合、数値に修正する
            if str(DF_nmiss[feature].dtype).find("[ns]") > -1:
                DF_nmiss[feature] = DF_nmiss[feature].dt.year * 10000 + DF_nmiss[feature].dt.month * 100 + DF_nmiss[feature].dt.day

            #カテゴリ変数はそのままカテゴリとして採用
            if DF_nmiss[feature].dtype == "object":
                #print("object")
                DF_nmiss["category_" + idx_] = DF_nmiss[feature]
                cat_list = list(DF_nmiss["category_" + idx_].unique())
                cat_list.sort()

                cardinality_ = len(cat_list)
                cat_dict = dict(zip(
                    [cat_list[i] for i in range(cardinality_)], 
                    [str(i+1).zfill(len(str(cardinality_))) + ": " + str(cat_list[i]) for i in range(cardinality_)])
                )
                DF_nmiss["category_" + idx_].replace(cat_dict,inplace = True)

            else :
                #print("numeric")
                try :
                    qcut_ = int(100 / X_bins)
                    p_list = [p for p in range(0,100,qcut_)]
                    p_list.extend([100])

                    q_list = [np.percentile(DF_nmiss[feature],q) for q in p_list]

                    if X_bin_closed == "left":
                        #print("left")
                        q_list[-1] = q_list[-1] + 1
                        DF_nmiss["category_" + idx_ ] = pd.cut(DF_nmiss[feature],bins = q_list,right = False)
                    elif X_bin_closed == "right":
                        #print("right")
                        q_list[0] = q_list[0] - 1
                        DF_nmiss["category_" + idx_] = pd.cut(DF_nmiss[feature],bins = q_list,right = True)

                #基本50分割でセット。それで分割できなければ生の値でいい
                except :
                    DF_nmiss["category_" + idx_ ] = DF_nmiss[feature]

                cat_list = list(DF_nmiss["category_" + idx_ ].unique())
                cat_list.sort()

                cardinality_ = len(cat_list)
                #ラベルふり
                cat_dict = dict(zip(
                    [cat_list[i] for i in range(cardinality_)], 
                    [str(i+1).zfill(len(str(cardinality_))) + ": " + str(cat_list[i]) for i in range(cardinality_)])
                )
                #cat_transform_list = dict(zip(cat_list,q_list))

                DF_nmiss["category_" + idx_].replace(cat_dict,inplace = True)

            #欠損先はカテゴリを欠損で埋める
            DF_miss["category_" + idx_ ] = str(0).zfill(len(str(cardinality_))) + ": missing"

            #欠損先と入力先を縦積み
            DF2 = pd.concat([DF_nmiss,DF_miss],axis = 0)

            missing_rate = len(DF_miss)/(len(DF_nmiss) + len(DF_miss))

            return DF2,missing_rate

        def preprocessing_loop(DF):
            calc_feature_list = ["feature_"+str(i + 1) for i in range(num_select_feature)]
            DF_chunk = [preprocess(DF,feature_) for feature_ in calc_feature_list]
            DF2 = DF.join(pd.concat([DF_chunk[i][0].iloc[:,-1] for i in range(len(calc_feature_list))],axis = 1))
            missing_rate_dict = dict(zip(
                [feat_ for feat_ in calc_feature_list],
                [DF_chunk[i][1] for i in range(len(calc_feature_list))]
            ))

            col_list_category = [col for col in DF2.columns if col.find("category_") > -1]

            return DF2,missing_rate_dict,col_list_category

        def make_cross_table(DF2,col_list_category):
            #AICを計算するためのクロス表を作成
            _cross = pd.crosstab(
                [DF2[col_] for col_ in col_list_category],
                DF2["label"],margins=None,dropna = False
            )
            #該当先がいない場合、NAになるので0埋め
            _cross = _cross.fillna(0)

            n_feature , n_label = _cross.shape
            #全体件数での対数尤度。log内が0だと計算できないので閾値を足してやる

            return _cross,n_feature,n_label


        #def calc_term2(cross_table):
        #    term2 = (cross_table.sum().sum()) * np.log(cross_table.sum().sum() + adj_parametor)
        #    return term2

        #def calc_term2(cross_table):
        #    term2 = (cross_table.sum().sum()) * np.log(cross_table.sum().sum() + adj_parametor)
        #    return term2

        #def calc_term1_0(cross_table,n_feature,n_label):
            #AIC0 =====================================================================================
            #列単位での和
        #    i_list = [(cross_table.iloc[i,:].sum()) * np.log(cross_table.iloc[i,:].sum() + adj_parametor) for i in range(0,n_feature)]

            #行単位での和
        #    j_list = [(cross_table.iloc[:,j].sum()) * np.log(cross_table.iloc[:,j].sum() + adj_parametor) for j in range(0,n_label)]

        #    term1_0 = sum(i_list) + sum(j_list)
        #    return term1_0


        def calc_term1_1(cross_table,n_feature,n_label):
            #AIC1 =====================================================================================
            yudo_matrix = np.zeros([n_feature ,n_label])

            n_all = cross_table.sum().sum()
            for i in range(n_feature):
                #print(i)
                for j in range(n_label):
                    #立て
                    n_sum_col = cross_table.iloc[i,:].sum()
                    n_sum_row = cross_table.iloc[:,j].sum()

                    #-2 * (cross_table.iloc[2,1] * np.log((n_all * cross_table.iloc[2,1])/(cross_table.iloc[:,1].sum() * cross_table.iloc[2,:].sum())))

                    #yudo_matrix[i,j] = cross_table.iloc[i,j] * np.log(
                    #    (n_all * cross_table.iloc[i,j] + adj_parametor)/(n_sum_row * n_sum_col + adj_parametor)
                    #)
                    yudo_matrix[i,j] = cross_table.iloc[i,j] * np.log(
                        (n_all * cross_table.iloc[i,j] + adj_parametor)/(n_sum_row * n_sum_col + adj_parametor)
                    )

            term1_1 = yudo_matrix.sum()

            return term1_1

        #def calc_aic0(term1_0,term2,n_feature,n_label):
        #    AIC0 = -2 * (term1_0 - 2 * term2) + 2 * (n_feature + n_label - 2)
        #    return AIC0

        #def calc_aic1(term1_1,term2,n_feature,n_label):
        def calc_aic1(term1_1,n_feature,n_label):
            #AIC1 = -2 * (term1_1 - term2) + 2 * ((n_label - 1) * (n_feature - 1)  - 1)
            AIC1 = -2 * (term1_1) + 2 * (n_label - 1) * (n_feature  - 1)
            return AIC1


        #def calc_aic_cat(AIC1,AIC0):
            #CATのAIC ==================================================================================
        #    AIC_CAT = AIC1 - AIC0
        #    return AIC_CAT

        def make_output_cross_table(DF2,col_list_category):
            output_cross_table = pd.crosstab(
                [DF2[col_] for col_ in col_list_category],
                DF2["label"],margins=True,dropna = False
            )
            #該当先がいない場合、NAになるので0埋め
            output_cross_table = output_cross_table.fillna(0)
            return output_cross_table

        def make_aic_table(output_cross_table,feature_select,missing_rate_dict,AIC_CAT):
            cardinality_label = len(output_cross_table.columns) - 1
            ratio_label = [col for col in output_cross_table.columns if col != "All"]
            ratio_table = pd.concat(
                [round(output_cross_table.loc[:,label_]/output_cross_table.loc[:,"All"],5) for label_ in ratio_label],
                axis = 1)
            ratio_table.columns = ["Ratio_" + str(label_) for label_ in ratio_label]

            output_cross_table2 = output_cross_table.join(ratio_table)    
            base = pd.DataFrame(index = [i for i in range(output_cross_table2.shape[0])])
            for i in range(len(feature_select)):
                base["feature_name_" + str(i + 1)] = feature_select[i]
            
            for i in range(len(feature_select)):
                base["missing_rate_of_feature_" + str(i + 1)] = missing_rate_dict["feature_" + str(i + 1)]

            base["AIC"] = round(AIC_CAT,5)
            AIC_table = pd.concat([base,output_cross_table2.reset_index()],axis = 1)
            return AIC_table

        def calc_aic_feature_select(feature_select):
            DF = make_DF(feature_select)
            DF2,missing_rate_dict,col_list_category_ = preprocessing_loop(DF)

            cross_table,n_feature,n_label = make_cross_table(DF2,col_list_category_)
            #term2 = calc_term2(cross_table)
            #term1_0 = calc_term1_0(cross_table,n_feature,n_label)
            term1_1 = calc_term1_1(cross_table,n_feature,n_label)

            #AIC0 = calc_aic0(term1_0,term2,n_feature,n_label)
            AIC1 = calc_aic1(term1_1,n_feature,n_label)
            #AIC_CAT = calc_aic_cat(AIC1,AIC0)

            output_cross_table = make_output_cross_table(DF2,col_list_category_)
            #AIC_table = make_aic_table(output_cross_table,feature_select,missing_rate_dict,AIC_CAT)
            AIC_table = make_aic_table(output_cross_table,feature_select,missing_rate_dict,AIC1)

            return AIC_table

        #col_aic_table_all = ["feature_name_" + str(i+1) for i in range(num_max_select_feature)]
        #col_aic_table_all.extend()
        #col_aic_table_all.extend(["category_" + str(i+1) for i in range(num_max_select_feature)])
        
        AIC_table_all = pd.DataFrame()
        AIC_summary_all = pd.DataFrame()
        
        for num_select_feature in tqdm(range(1,num_max_select_feature+1)):
            set_feature_list = [feature for feature in itertools.combinations(feature_list, num_select_feature)]

            aic_chunk = joblib.Parallel(n_jobs = -1)(joblib.delayed(calc_aic_feature_select)(feature_select) for feature_select in set_feature_list)
            AIC_table = pd.concat(aic_chunk,axis = 0)

            col_feature_sort = ["feature_name_" + str(i+1) for i in range(num_select_feature)]
            col_category_sort = ["category_" + str(i+1) for i in range(num_select_feature)]
            
            col_sort = ["AIC"].copy()
            col_sort.extend(col_feature_sort)
            col_sort.extend(col_category_sort)

            AIC_table.sort_values(col_sort,ascending = True,inplace = True)
            AIC_table.reset_index(drop = True,inplace = True)
            col_table = AIC_table.columns

            col_summary = col_feature_sort.copy()
            col_summary.extend(["AIC"])

            AIC_summary = AIC_table[col_summary].drop_duplicates()
            AIC_summary.reset_index(drop = True,inplace = True)

            AIC_table_all = pd.concat([AIC_table_all,AIC_table],axis = 0)
            AIC_summary_all = pd.concat([AIC_summary_all,AIC_summary],axis = 0)        

        AIC_table_all = AIC_table_all[col_table].sort_values(col_sort)
        AIC_summary_all = AIC_summary_all[col_summary].sort_values("AIC")
        AIC_table_all.reset_index(drop = True,inplace = True)
        AIC_summary_all.reset_index(drop = True,inplace = True)
        
        self.AIC_table_all = AIC_table_all
        self.AIC_summary_all = AIC_summary_all
        print("AIC calculation is finished")
        
        
    def to_csv(self,output_dir = "./",file_name = "temp"):
        AIC_table_all = self.AIC_table_all
        AIC_summary_all = self.AIC_summary_all 
        AIC_table_all.to_csv(output_dir +"CATDAP02_table_{}.csv".format(file_name))
        AIC_summary_all.to_csv(output_dir +"CATDAP02_summary_{}.csv".format(file_name))
        
        
    def heatmap(self,target_,yoko = 4,num_set_feature = 20):
        AIC_table_all = self.AIC_table_all
        AIC_summary_all = self.AIC_summary_all 
        num_max_select_feature = self.num_max_select_feature
        
        if num_max_select_feature == 2:
            #num_set_feature = set_feature.shape[0]
            col_set_feature = [col for col in AIC_summary_all.columns if col.find("feature_") > -1]
            set_feature = AIC_summary_all[col_set_feature]

            if num_set_feature > AIC_summary_all.shape[0]:
                num_set_feature = AIC_summary_all.shape[0]
            
            #プロットの設定
            if (num_set_feature % yoko) != 0:
                tate = (num_set_feature // yoko) + 1
            else :
                tate = (num_set_feature // yoko)

            fig, ax = plt.subplots(tate,yoko,tight_layout = True,figsize = (yoko * 5,tate * 5))       

            
            
            for num in range(num_set_feature):
                #print(num)
                feature_select = AIC_summary_all[col_set_feature].iloc[num,:]
                AIC_value = AIC_summary_all.iloc[num,:]["AIC"]        

                #集計対象変数の組み合わせのテーブルを抽出
                temp = AIC_table_all.copy()
                for col in col_set_feature:
                    temp = temp[temp[col] == feature_select[col]].copy()

                #クロス集計用にをリネーム
                col_category = [col for col in temp.columns if str(col).find("category_") > -1]
                rename_dict = dict(zip(col_category,feature_select.values))

                #集計テーブルで必要な列をターゲットに抽出
                col_temp2 = col_category.copy()
                col_temp2.extend([target_])
                temp2 = temp[temp["category_1"] != "All"].copy()
                temp2 = temp2[col_temp2].copy()
                temp2.rename(columns = rename_dict,inplace = True)

                title_list = list(rename_dict.values())

                tbl_plt = temp2.pivot_table(
                    index = rename_dict[col_category[0]],
                    columns = rename_dict[col_category[1]],
                    values = target_
                )

                try:
                    i,j = num // yoko,num % yoko  
                    if tate == 1:
                        ax[j].set_title(target_ + " by " + str(title_list[0]) + "&" + str(title_list[1]) + " AIC:" + str(AIC_value))
                        sns.heatmap(tbl_plt,annot = True,ax = ax[j],cmap='RdBu')

                    elif yoko == 1:
                        ax[i].set_title(target_ + " by " + str(title_list[0]) + "&" + str(title_list[1]) + " AIC:" + str(AIC_value))
                        sns.heatmap(tbl_plt,annot = True,ax = ax[i],cmap='RdBu')

                    else :
                        #temp.plot(kind = "bar",stacked = True,title = col_ + " AIC:" + str(AIC_value),ax=ax[i][j],cmap = "coolwarm",ec = "black")
                        ax[i][j].set_title(target_ + " by " + str(title_list[0]) + "&" + str(title_list[1]) + " AIC:" + str(AIC_value))
                        sns.heatmap(tbl_plt,annot = True,ax = ax[i][j],cmap='RdBu')

                except :
                    None

            plt.show()

        
        
        