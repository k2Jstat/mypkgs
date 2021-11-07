import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold,train_test_split
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import seaborn as sns

__all__ = [
    "LGBCV",
]


class LGBCV():
    def __init__(self,objective_ = "regression",metric_ = "rmse",lgb_params_ = None,
                 num_iteration_ = 100,num_early_stopping_ = 10,seed_ = 71,
                 kf = KFold(n_splits = 3,random_state = 71,shuffle=True)):

        if lgb_params_ == None:
            lgb_params_ = {
                "objective":objective_, 
                "metric":metric_,
                "verbosity":-1,
                "learning_rate":0.1,
                #"bagging_seed":seed_
            }
            
        lgb_params_["bagging_seed"] = seed_
        
        self.lgb_params_ = lgb_params_
        self.kf = kf
        self.num_iteration = num_iteration_
        self.num_early_stopping = num_early_stopping_
        self.seed_ = seed_
            
    def fit(self,X,y,feature_category = None,pred_label = "lgb",importance_type = "gain",valid_size = 0.1):
        lgb_params_ = self.lgb_params_
        kf = self.kf
        num_iteration_ = self.num_iteration
        num_early_stopping_ = self.num_early_stopping
        seed_ = self.seed_
        
        metric_ = lgb_params_["metric"]        
        
        model_dict = {}
        oof_df_all = pd.DataFrame()
        imp_df_all = pd.DataFrame(index = X.columns)
        cv_df_all = pd.DataFrame()

        if feature_category == None:
            col_le = [col for col in X.columns if col.find('_LE') > -1]
            feature_category = col_le

        if len(feature_category) > 0:
            X[feature_category] = X[feature_category].astype("category")

        for n_fold, (train_idx, test_idx) in enumerate(kf.split(X,y)):
        #def lgb_learn(n_fold,train_idx,test_idx):
            n_fold_ = n_fold + 1
            X_, y_ = X.iloc[train_idx].copy(), y.iloc[train_idx].copy()
            X_test, y_test = X.iloc[test_idx].copy(), y.iloc[test_idx].copy()
            
            X_train,X_valid,y_train,y_valid = train_test_split(X_,y_,test_size = valid_size,random_state = seed_)
            #dtrain = lgb.Dataset(data = X_train,label = y_train,categorical_feature=feature_category)
            #dvalid = lgb.Dataset(data = X_valid,label = y_valid,categorical_feature=feature_category)
            #dtest = lgb.Dataset(data = X_test,label = y_test,categorical_feature=feature_category)
            dtrain = lgb.Dataset(data = X_train,label = y_train)
            dvalid = lgb.Dataset(data = X_valid,label = y_valid)
            dtest = lgb.Dataset(data = X_test,label = y_test)

            result  = lgb.train(
                lgb_params_,
                dtrain,
                valid_sets = [dtrain,dvalid],
                #categorical_feature=feature_category,
                num_boost_round = num_iteration_,
                early_stopping_rounds=num_early_stopping_,
                verbose_eval=1000
            )

            #oof_predの作成
            pred_train = result.predict(X_train,num_iteration=result.best_iteration)
            pred_valid = result.predict(X_valid,num_iteration=result.best_iteration)
            pred_test = result.predict(X_test,num_iteration=result.best_iteration)

            oof_df = pd.DataFrame(index = X_test.index)
            oof_df["cv_{}".format(pred_label)] = (n_fold + 1)
            oof_df["oof_pred_{}".format(pred_label)] = pred_test

            oof_df_all = pd.concat([oof_df_all,oof_df],axis = 0)

            #importanceの作成
            imp_df = pd.DataFrame(index = result.feature_name())
            imp_df["cv{}_{}".format(n_fold + 1,importance_type)] = result.feature_importance(importance_type=importance_type)

            imp_df_all = pd.concat([imp_df_all,imp_df],axis = 1)

            #cvの作成
            if metric_ == "rmse":
                cv_train = mean_squared_error(y_train,pred_train) ** 1/2
                cv_valid = mean_squared_error(y_valid,pred_valid) ** 1/2
                cv_test = mean_squared_error(y_test,pred_test) ** 1/2

            cv_df = pd.DataFrame(index = ["cv{}".format(n_fold + 1)])
            cv_df["train_size"] = X_train.shape[0]
            cv_df["valid_size"] = X_valid.shape[0]
            cv_df["test_size"] = X_test.shape[0]
            cv_df["train_{}".format(metric_)] = cv_train
            cv_df["valid_{}".format(metric_)] = cv_valid
            cv_df["test_{}".format(metric_)] = cv_test

            cv_df_all = pd.concat([cv_df_all,cv_df],axis = 0)

            print('Fold %2d %s : %.6f' % (n_fold + 1,metric_,cv_test))

            model_dict["cv{}".format(n_fold + 1)] = result

        #cvの作成
        cv_mean = pd.DataFrame([cv_df_all.mean(axis = 0)],index = ["mean"])
        cv_std = pd.DataFrame([cv_df_all.std(axis = 0)],index = ["std"])
        cv_df_all = pd.concat([cv_df_all,cv_mean,cv_std],axis = 0)

        #cv_df_all["train_size"] = cv_df_all["train_size"].astype(int)
        #cv_df_all["valid_size"] = cv_df_all["valid_size"].astype(int)
        #cv_df_all["test_size"] = cv_df_all["test_size"].astype(int)

        mean_imp = imp_df_all.mean(axis = 1)
        std_imp = imp_df_all.std(axis = 1)
        imp_df_all["mean_{}".format(importance_type)] = mean_imp
        imp_df_all["std_{}".format(importance_type)] = std_imp
        imp_df_all.sort_values("mean_{}".format(importance_type),inplace = True,ascending = False)

        score = cv_df_all.loc["mean","test_{}".format(metric_)]

        oof_df_all.sort_index(inplace = True)
        self.oof_pred_df = oof_df_all
        self.cv_df = cv_df_all
        self.importance_df = imp_df_all
        self.model_dict = model_dict

        return score

    def plot_importance(self,num_top = 50,figsize_ = (8,10)):
        importance_df = self.importance_df

        col_top = importance_df.index[:num_top]
        temp2 = importance_df.iloc[:,:-2]
        plot_df = pd.DataFrame()

        for col in temp2.columns:
            temp3 = temp2[temp2.index.isin(col_top)][[col]]
            temp3.columns = ["importance"]
            plot_df = pd.concat([plot_df,temp3],axis = 0)

        plot_df.reset_index(inplace = True)
        plot_df.rename(columns = {"index":"feature_name"},inplace = True)
        plt.figure(figsize = figsize_)
        sns.barplot(x = plot_df["importance"],y = plot_df["feature_name"])
        plt.tight_layout()
        plt.show()

        
        

        