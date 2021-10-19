import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

__all__ = [
    "AllPlot"
]
class AllPlot():
    def __init__(self,df,yoko=3,bins_ = None,adj_parametor = 1e-10):
        col_numeric = [col_ for col_ in df.columns if (df[col_].dtype != "O") & (df[col_].dtype.str.find("[ns]") == -1)]
        col_strings = [col_ for col_ in df.columns if (df[col_].dtype == "O")]
        col_datetime = [col_ for col_ in df.columns if (df[col_].dtype.str.find("[ns]") > -1)]
        if bins_ == None:
            #sturges
            bins_ = int(round(np.log2(df.shape[0]) + 1,0)) 
            
        self.df = df
        self.col_numeric = col_numeric
        self.col_strings = col_strings
        self.col_datetime = col_datetime
        self.yoko = yoko
        self.bins_ = bins_
        self.adj_parametor = adj_parametor
        
    def Histgram(self):
        df = self.df
        col_numeric = self.col_numeric
        yoko = self.yoko
        bins_ = self.bins_

        num_cols = len(col_numeric)
        if (num_cols % yoko) != 0:
            tate = (num_cols // yoko) + 1
        else :
            tate = (num_cols // yoko)

        fig, ax = plt.subplots(tate,yoko,tight_layout = True,figsize = (yoko * 5,tate * 3))

        for num,col_ in enumerate(col_numeric):
            try:
                i,j = num // yoko,num % yoko  
                if tate == 1:
                    #ax[j].grid(axis = "y")
                    ax[j].hist(df[col_],bins = bins_,ec = "black")   
                    ax[j].set_title(col_)
                    
                elif yoko == 1:
                    #ax[i].grid(axis = "y")
                    ax[i].hist(df[col_],bins = bins_,ec = "black")
                    ax[i].set_title(col_)
            
                else :
                    #ax[i][j].grid(axis = "y")
                    ax[i][j].hist(df[col_],bins = bins_,ec = "black")
                    ax[i][j].set_title(col_)

            except :
                None

        plt.show()

    def LinearityCheckBinaryTarget(self,target_):
        df = self.df
        col_numeric = self.col_numeric
        yoko = self.yoko
        bins_ = self.bins_
        adj_parametor = self.adj_parametor
        
        cardinality_ = len(df[target_].unique())
        if cardinality_ == 2:

            num_cols = len(col_numeric)
            if (num_cols % yoko) != 0:
                tate = (num_cols // yoko) + 1
            else :
                tate = (num_cols // yoko)

            fig, ax = plt.subplots(tate,yoko,tight_layout = True,figsize = (yoko * 5,tate * 4))

            for num,col_ in enumerate(col_numeric):
                try:
                    temp = df[[target_,col_]].copy()
                    temp["bins_" + col_] = pd.cut(temp[col_],bins_) 

                    ln_tbl = temp.pivot_table(
                        index = "bins_" + col_,
                        values = target_,
                        aggfunc = "mean"
                    )

                    ln_tbl["odds"] = (ln_tbl[target_] + adj_parametor)/(1 - ln_tbl[target_] + adj_parametor)
                    ln_tbl["ln_odds"] = np.log(ln_tbl["odds"])

                    ln_odds_f = ln_tbl["ln_odds"][0]
                    ln_odds_l = ln_tbl["ln_odds"][-1]

                    list_linear = [(ln_odds_l - ln_odds_f)/(bins_ - 1) * i + ln_odds_f for i in range(bins_)]

                    ln_tbl["ln_odds_linear"] = list_linear
                    idx_list = [str(idx_) for idx_ in ln_tbl.index]
                    ln_tbl.index = idx_list

                    i,j = num // yoko,num % yoko  
                    if tate == 1:
                        ax[j].grid(True)
                        ax[j].plot(ln_tbl[["ln_odds","ln_odds_linear"]],marker = "o")
                        ax[j].legend(["ln_odds","ln_odds_linear"])
                        ax[j].tick_params(axis='x', labelrotation=90)
                        ax[j].set_title("ln_odds_" + target_ + " by " + col_)  

                    elif yoko == 1:
                        ax[i].grid(True)
                        ax[i].xticks(rotation=90)
                        ax[i].plot(ln_tbl[["ln_odds","ln_odds_linear"]],marker = "o")
                        ax[i].tick_params(axis='x', labelrotation=90)
                        ax[i].legend(["ln_odds","ln_odds_linear"])

                    else :
                        ax[i][j].grid(True)
                        ax[i][j].plot(ln_tbl[["ln_odds","ln_odds_linear"]],marker = "o")
                        ax[i][j].legend(["ln_odds","ln_odds_linear"])
                        ax[i][j].tick_params(axis='x', labelrotation=90)
                        ax[i][j].set_title("ln_odds_" + target_ + " by " + col_)  

                except :
                    None

            plt.show()
        
        else :
            warnings.warn('set a numeric binary variable in target')
        
    def CountByCategory(self):
        df = self.df
        col_strings = self.col_strings
        yoko = self.yoko

        num_cols = len(col_strings)
        if (num_cols % yoko) != 0:
            tate = (num_cols // yoko) + 1
        else :
            tate = (num_cols // yoko)

        fig, ax = plt.subplots(tate,yoko,tight_layout = True,figsize = (yoko * 5,tate * 3))

        for num,col_ in enumerate(col_strings):
            try:
                i,j = num // yoko,num % yoko  
                vc = df[col_].value_counts(dropna = False)
                temp = pd.DataFrame(index = vc.index)
                temp["count"] = vc.values
                
                if tate == 1:
                    #ax[j].grid(axis = "y")
                    ax[j].bar(x = temp.index,height = temp["count"],ec = "black")
                    ax[j].set_title(col_)
                
                else :
                    #ax[i][j].grid(axis = "y")
                    ax[i][j].bar(x = temp.index,height = temp["count"],ec = "black")
                    ax[i][j].set_title(col_)
                
            except :
                None
                
        plt.show()       
        
    def Boxplot(self):
        df = self.df
        col_numeric = self.col_numeric
        col_strings = self.col_strings

        num_cols_numeric = len(col_numeric)
        num_cols_strings = len(col_strings)

        tate = num_cols_numeric
        yoko = num_cols_strings

        fig, ax = plt.subplots(tate,yoko,tight_layout = True,figsize = (yoko * 5,tate * 3))

        for i,col_num in enumerate(col_numeric):
            for j,col_str in enumerate(col_strings):          
                try:
                    if tate == 1:
                        ax[j].grid(axis = "y")
                        sns.boxplot(x = col_str,y = col_num,data = df,ax = ax[j])
                        ax[j].set_title(col_num + " by " + col_str)

                    elif yoko == 1:
                        ax[i].grid(axis = "y")
                        sns.boxplot(x = col_str,y = col_num,data = df,ax = ax[i])
                        ax[i].set_title(col_num + " by " + col_str)

                    else :
                        ax[i][j].grid(axis = "y")
                        sns.boxplot(x = col_str,y = col_num,data = df,ax = ax[i][j])
                        ax[i][j].set_title(col_num + " by " + col_str)

                except :
                    None

        plt.show()      
        
        
    def CountByDatetime(self):        
        df = self.df
        temp = df.copy()

        col_datetime = self.col_datetime
        num_cols = len(col_datetime)
        tate = num_cols

        yoko = 4

        fig, ax = plt.subplots(tate,yoko,tight_layout = True,figsize = (yoko * 5,tate * 3))

        for num,col_ in enumerate(col_datetime):
            try:
                i = num 
                temp[col_ + "_year"] = temp[col_].dt.year
                vc = temp[col_ + "_year"].value_counts(dropna = False)
                temp_y = pd.DataFrame(index = vc.index.astype(str))
                temp_y[col_ + "_year_count"] = vc.values
                temp_y.sort_index(inplace = True)

                temp[col_ + "_month"] = temp[col_].dt.month
                vc = temp[col_ + "_month"].value_counts(dropna = False)
                temp_m = pd.DataFrame(index = vc.index.astype(str))
                temp_m[col_ + "_month_count"] = vc.values
                temp_m.sort_index(inplace = True)

                temp[col_ + "_day"] = temp[col_].dt.day
                vc = temp[col_ + "_day"].value_counts(dropna = False)
                temp_d = pd.DataFrame(index = vc.index.astype(str))
                temp_d[col_ + "_day_count"] = vc.values
                temp_d.sort_index(inplace = True)

                temp[col_ + "_weekday"] = temp[col_].dt.weekday
                vc = temp[col_ + "_weekday"].value_counts(dropna = False)
                vc.index = vc.index.map({0:"0:Monday",1:"1:Tuesday",2:"2:Wednesday",3:"3:Thursday",4:"4:Friday",5:"5:Saturday",6:"6:Sunday"})
                temp_w = pd.DataFrame(index = vc.index.astype(str))
                temp_w[col_ + "_weekday_count"] = vc.values
                temp_w.sort_index(inplace = True)

                if tate == 1:
                    #ax[0].grid(axis = "y")
                    ax[0].bar(x = temp_y.index,height = temp_y[col_ + "_year_count"])
                    ax[0].set_title(col_ + "_year_count")
                    #ax[1].grid(axis = "y")
                    ax[1].bar(x = temp_m.index,height = temp_m[col_ + "_month_count"])
                    ax[1].set_title(col_ + "_month_count")
                    #ax[2].grid(axis = "y")
                    ax[2].bar(x = temp_d.index,height = temp_d[col_ + "_day_count"])
                    ax[2].set_title(col_ + "_day_count")
                    #ax[3].grid(axis = "y")
                    ax[3].bar(x = temp_w.index,height = temp_w[col_ + "_weekday_count"])
                    ax[3].set_title(col_ + "_weekday_count")       

                else :
                    #ax[i][0].grid(axis = "y")
                    ax[i][0].bar(x = temp_y.index,height = temp_y[col_ + "_year_count"])
                    ax[i][0].set_title(col_ + "_year_count")
                    #ax[i][1].grid(axis = "y")
                    ax[i][1].bar(x = temp_m.index,height = temp_m[col_ + "_month_count"])
                    ax[i][1].set_title(col_ + "_month_count")
                    #ax[i][2].grid(axis = "y")
                    ax[i][2].bar(x = temp_d.index,height = temp_d[col_ + "_day_count"])
                    ax[i][2].set_title(col_ + "_day_count")
                    #ax[i][3].grid(axis = "y")
                    ax[i][3].bar(x = temp_w.index,height = temp_w[col_ + "_weekday_count"])
                    ax[i][3].set_title(col_ + "_weekday_count")       

                plt.show()
            
            except :
                None
                

    def MeanByTimeseries(self):
        df = self.df
        col_datetime = self.col_datetime
        col_numeric = self.col_numeric

        num_cols = len(col_datetime)
        num_cols_num = len(col_numeric)

        tate = num_cols
        yoko = num_cols_num

        fig, ax = plt.subplots(tate,yoko,tight_layout = True,figsize = (yoko * 5,tate * 3))

        for i,col_ in enumerate(col_datetime):
            for j,var_ in enumerate(col_numeric):
                temp = df.groupby(col_)[var_].mean()
                try:
                    if tate == 1:
                        ax[j].plot(temp,marker = "o")
                        ax[j].set_title(var_ +" mean by " + col_)
                    else :    
                        ax[i][j].plot(temp,marker = "o")
                        ax[i][j].set_title(var_ +" mean by " + col_)

                except:
                    None

            plt.show()

                
