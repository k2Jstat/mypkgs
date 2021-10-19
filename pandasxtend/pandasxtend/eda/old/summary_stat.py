def summary_stat(df,df_name,top = 5,out_file = "./summary_stat_"):
    import pandas as pd
    #df = train
    #df_name = "train"
    #top = 5
    #out_file = "./summary_stat_"
    
    summary_dic = {}

    try :
        cnt_unq = pd.DataFrame(df[num_feat].nunique(),columns = ["unique"]).T
        cnt_na = pd.DataFrame(df[num_feat].isna().sum(),columns = ["cnt_na"]).T
        head3 = df[num_feat].head(top)
        head3.index = ["head_" + str(rank + 1) for rank in range(0,top)]
        med = pd.DataFrame(df[num_feat].median(),columns = ["median"]).T
        type_ = pd.DataFrame(df[num_feat].dtypes,columns = ["dtype"]).T

        summary = df[num_feat].describe(percentiles = {0.05,0.25,0.5,0.75,0.95})
        summary_num = pd.concat([summary,med]).T
        summary_num["cv"] = summary_num["std"] / summary_num["mean"]
        summary_num = pd.concat([summary_num.T,cnt_unq,cnt_na,type_,head3])

        summary_dic["summary_numeric"] = summary_num
        
    except: 
        None

    try :
        summary_cat = df[cat_feat].describe().iloc[:2,:]
        type_ = pd.DataFrame(df[cat_feat].dtypes,columns = ["dtype"]).T
        cat_ = pd.DataFrame()
        for feat in cat_feat:
            temp = df[feat].value_counts(dropna = False)

            top_df = pd.DataFrame(index = [feat])
            for rank in range(0,top):
                try :
                    top_df["TOP " + str(rank + 1) + " FEAT"] = [temp.index[rank]]
                    top_df["TOP " + str(rank + 1) + " COUNT"] = [temp.values[rank]]

                except:
                    top_df["TOP " + str(rank + 1) + " FEAT"] = " "
                    top_df["TOP " + str(rank + 1) + " COUNT"] = 0

            cat_ = pd.concat([cat_,top_df])

        summary_cat = pd.concat([summary_cat,type_,cat_.T],axis = 0)
        summary_dic["summary_category"] = summary_cat
    except: 
        None

    try :
        summary_ts = df[ts_feat].describe()
        type_ = pd.DataFrame(df[ts_feat].dtypes,columns = ["dtype"]).T
        summary_ts = pd.concat([summary_ts,type_],axis = 0)

        summary_dic["summary_time"] = summary_ts
    except:
        None
        
    out_list = list(summary_dic.keys())

    with pd.ExcelWriter(out_file + str(df_name) + ".xls") as writer:  
        for sheet_ in out_list:
            summary_dic[sheet_].T.to_excel(writer, sheet_name=sheet_)