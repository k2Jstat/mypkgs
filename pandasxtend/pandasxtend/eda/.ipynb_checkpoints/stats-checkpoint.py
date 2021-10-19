import numpy as np
import pandas as pd
import joblib

__all__ = [
    "KL_Divergence",
    "Summary",
]

class KL_Divergence:
    def __init__(self,df,binary_columns,base_label = True,bins_ = None,adj_parametor = 1e-10):
        if bins_ == None:
            #sturges
            bins_ = int(round(np.log2(df.shape[0]) + 1,0)) 
            
        self.df = df
        self.binary_columns = binary_columns
        self.base_label = base_label
        self.param_adj = adj_parametor
        self.bins_ = bins_

    def run(self):
        df = self.df
        binary_columns = self.binary_columns
        param_adj = self.param_adj
        base_label = self.base_label
        bins_ = self.bins_
        
        def KL(feat):
            temp = pd.DataFrame()
            try:
                unq_list = np.unique(df[binary_columns])
                min_,max_ = df[feat].agg(["min","max"])

                q_list = [round((min_ - max_) * i/bins_ + max_,0) for i in range(0,bins_ + 1,1)]
                q_list.sort()

                df[feat + "_bins"] = pd.cut(df[feat],q_list,right = False)

                comp_df = pd.crosstab(df[feat + "_bins"],df[binary_columns])

                comp = set(unq_list) - set([base_label])
                comp_label = comp.pop()

                comp_sum = comp_df.sum(axis = 0)
                comp_df = (comp_df / comp_sum).round(5)

                comp_df2 = comp_df + param_adj
                kl_value = np.sum(comp_df2[base_label] * np.log(comp_df2[base_label]/comp_df2[comp_label]))

            except :
                kl_value = np.nan
                comp_df = pd.DataFrame()

            temp["columns"] = [feat]
            temp["base"] = [base_label]
            temp["KL_Divergence"] = [round(kl_value,5)]

            return temp,comp_df

        columns_numeric = [col for col in df.columns if df[col].dtype != "O"]
        job_df = joblib.Parallel(n_jobs=-1, verbose=1)(joblib.delayed(KL)(feat_) for feat_ in columns_numeric)

        def make_kl_df(i):
            return job_df[i][0]

        def make_tbl_df(i):
            temp_tbl = job_df[i][1]
            if temp_tbl.shape[0] != 0:
                temp_tbl_base = pd.DataFrame(index = [i for i in range(temp_tbl.shape[0])])
                temp_tbl_base["columns"] = temp_tbl.index.name[:-5]
                temp_tbl.index.name = "bins"
                return pd.concat([temp_tbl_base,temp_tbl.reset_index()],axis = 1)

        loop_ = range(len(job_df))
        kl_chunk = joblib.Parallel(n_jobs=-1, verbose=1)(joblib.delayed(make_kl_df)(i) for i in loop_)
        kl_df = pd.concat(kl_chunk,axis = 0).reset_index(drop = True)

        tbl_chunk = joblib.Parallel(n_jobs=-1, verbose=1)(joblib.delayed(make_tbl_df)(i) for i in loop_)
        tbl_df = pd.concat(tbl_chunk,axis = 0).reset_index(drop = True)

        self.kl_df = kl_df
        self.tbl_df = tbl_df

        return kl_df
    
    def to_csv(self,dir_output = "./",file_name = "temp"):
        kl_df = self.kl_df
        kl_df.to_csv(dir_output + "KL_divergence_" + str(file_name) + ".csv")
        tbl_df = self.tbl_df
        tbl_df.to_csv(dir_output + "CrossTable_bins_" + str(file_name) + ".csv")

        

class Summary:
    def __init__(self,df):
        self.df = df

    def run(self):
        df_input = self.df
        def feat_summary(feat):
            temp = pd.DataFrame()
            cnt_ = df_input[feat].count()
            unq_ = df_input[feat].nunique(dropna=False)
            miss_ = df_input[feat].isnull().sum()
            vc_ = df_input[feat].value_counts(dropna = False)
            f1,v1 = vc_.index[0],vc_.values[0]
            h1 = df_input[feat].iloc[0]
            t1 = df_input[feat].iloc[-1]

            if df_input[feat].dtype == "O":
                type_ = "strings"

            else :
                type_ = "numeric"
                min_ = df_input[feat].min()
                max_ = df_input[feat].max()
                mean_ = df_input[feat].mean()
                std_ = df_input[feat].std()
                p1,p5,p25,p50,p75,p95,p99 = np.percentile(df_input[feat],[1,5,25,50,75,95,99])

            temp["column_name"] = [feat]
            temp["format"] = [type_]
            temp["count"] = [cnt_]
            temp["unique_count"] = [unq_]
            temp["missing_count"] = [miss_]

            if df_input[feat].dtype != "O":
                temp["min"] = [min_]
                temp["p01"] = [p1]
                temp["p05"] = [p5]
                temp["p25"] = [p25]
                temp["p50"] = [p50]
                temp["p75"] = [p75]
                temp["p95"] = [p95]
                temp["p99"] = [p99]
                temp["max"] = [max_]
                temp["mean"] = [mean_]
                temp["std"] = [std_]

            temp["mode_value"] = [f1]    
            temp["mode_counts"] = [v1]
            temp["head1"] = [h1]
            temp["tail1"] = [t1]

            return temp.T

        job_df = joblib.Parallel(n_jobs=-1, verbose=1)(joblib.delayed(feat_summary)(feat_) for feat_ in df_input.columns)
        job_df2 = pd.concat(job_df,axis = 1).T.reset_index(drop = True)

        self.summary = job_df2
        return job_df2
    
    def to_csv(self,dir_output = "./",file_name = "temp"):
        df_output = self.summary
        df_output.to_csv(dir_output + "stats_summary_" + str(file_name) + ".csv")
