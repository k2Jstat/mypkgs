def read_dir(dir_,fmt = ".csv"):
    import os 
    import pandas as pd
    from IPython.core.display import display
    
    file_list = os.listdir(dir_)
    df_dic = {}
    
    for file in file_list:
        if file.find(fmt) > -1:
            temp = pd.read_csv(dir_ + file)
            df_dic[file] = temp
            print(("file name : "+ file + ",  row: " + str(temp.shape[0]) + ",  col: " + str(temp.shape[1]) + "  ==================================================================================="))
            print("  ")
            display(temp.head(3))
            #display(temp.tail(3))
            print("  ")
            print("=================================================================================================================================================================")
            print("  ")
            print("  ")
            print("  ")
            del temp
            gc.collect()
        
    return df_dic