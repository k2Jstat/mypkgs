import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder

__all__ = [
    "CategoricalEncoder",
]

class CategoricalEncoder():
    def __init__(self,X_train,y_train = None,X_test = None,y_test = None,random_state_ = 71):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
    
        feature_cat = [col for col in X_train.columns if X_train[col].dtypes == "O"]
        self.feature_cat = feature_cat
        self.random_state_ = random_state_
    
    def LabelEncoding(self,feature_label = "LE"):
        X_train = self.X_train
        X_test = self.X_test
        feature_cat = self.feature_cat
        
        le_df = pd.DataFrame()
        
        le_train = pd.DataFrame(index = X_train.index)
        if X_test is not None:
            le_test = pd.DataFrame(index = X_test.index)
            
        for col in feature_cat:
            le = LabelEncoder()
            temp = le.fit_transform(X_train[col])
            le_train[col + "_" + feature_label] = temp
            le_df["transform_" + str(col)] = le.classes_

            if X_test is not None:
                le_test[col + "_" + feature_label] = le.transform(X_test[col])
                
        self.le_train = le_train
        self.le_df = le_df
        
        if X_test is not None:
            self.le_test = le_test

    def FrequencyEncoding(self,feature_label = "FE"):
        X_train = self.X_train
        X_test = self.X_test
        feature_cat = self.feature_cat
        
        fe_df = pd.DataFrame()
        
        fe_train = X_train[feature_cat].copy()
        fe_df = pd.DataFrame()

        if X_test is not None:
            fe_test = X_test[feature_cat].copy()

        for col in feature_cat:
            vc = X_train[col].value_counts()

            fe_train[col + "_" + feature_label] = fe_train[col].map(vc)
            if X_test is not None:
                fe_test[col + "_" + feature_label] = fe_test[col].map(vc)

            temp = pd.DataFrame()
            temp[col] = vc.index
            temp[col + "_" + feature_label] = vc.values
            fe_df = pd.concat([fe_df,temp],axis = 1) 


        col_feat_fe = [col for col in fe_train.columns if col.find("_" + str(feature_label)) > -1]
        fe_train = fe_train[col_feat_fe].copy()
        if X_test is not None:
            fe_test = fe_test[col_feat_fe].copy()

        self.fe_train = fe_train
        self.fe_df = fe_df
        if X_test is not None:
            self.fe_test = fe_test

    def TargetEncodingCV(self,feature_label = "TE",target_stat = "mean",min_samples_leaf = 1,smoothing = 1,noise_level = 0,
                         kf = KFold(n_splits = 4,shuffle=True,random_state = 71)):
        X_train = self.X_train
        X_test = self.X_test
        y_train = self.y_train
        y_test = self.y_test
        feature_cat = self.feature_cat
        random_state_ = self.random_state_
        
        te_train = X_train[feature_cat].copy()
        te_train["target"] = y_train.copy()
        if X_test is not None:
            te_test = X_test[feature_cat].copy()
            te_test["target"] = y_test.copy()
        
        te_df = pd.DataFrame()

        np.random.seed(random_state_)
        for col in feature_cat:
            temp = np.repeat(np.nan,te_train.shape[0])
            for i,(idx1,idx2) in enumerate(kf.split(te_train)):
                #全体平均
                prior = te_train["target"].mean()
                #カテゴリ別平均
                target_encode = te_train.iloc[idx1].groupby(col)["target"].agg(target_stat)
                lambda_ = 1 + np.exp(-(target_encode - min_samples_leaf)/ smoothing)
                target_encode_smoothed = lambda_ * target_encode + (1 - lambda_) * prior        
                temp[idx2] = te_train[col].iloc[idx2].map(target_encode_smoothed).fillna(prior)
                #add noise
                temp[idx2] = temp[idx2] * (1 + noise_level * np.random.randn(len(temp[idx2])))
                col_name_te =col + "_{}_{}_{}".format(feature_label,target_stat,i+1) 
                    
                if X_test is not None:
                    te_test[col_name_te] = te_test[col].map(target_encode).fillna(prior)
                    #add noise
                    te_test[col_name_te] = te_test[col_name_te] * (1 + noise_level * np.random.randn(len(te_test)))

                te_df[col] = target_encode.index
                te_df[col_name_te] = target_encode.values

            col_train_te = col + "_{}_{}".format(feature_label,target_stat)
            te_train.loc[:,col_train_te] = temp
            col_te_cv = [col for col in te_df.columns if col.find(col_train_te) > -1]
            
            if X_test is not None:
                te_test.loc[:,col + "_{}_{}".format(feature_label,target_stat)] = te_test[col_te_cv].mean(axis = 1)
            
            te_df.loc[:,col + "_{}_{}".format(feature_label,target_stat)] = te_df[col_te_cv].mean(axis = 1)

        col_feat_te = [col for col in te_train.columns if col.find("_{}_{}".format(feature_label,target_stat)) > -1]
        te_train = te_train[col_feat_te].copy()
        if X_test is not None:
            te_test = te_test[col_feat_te].copy()

        self.te_train = te_train
        self.te_df = te_df
        if X_test is not None:
            self.te_test = te_test
            