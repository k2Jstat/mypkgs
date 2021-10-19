def dist_by_cat(df,category = None,yoko = 6,tate = 6,yoko_block = 5,plot_type = "violin",bins_=30):
    """
    input   : pandas DataFrame
    output  : violin_plot 
              box_plot
              dist_plot
              
    example : dist_by_cat(train,"target")  
              dist_by_cat(train,plot_type = "dist")
    """

    import matplotlib.pyplot as plt
    import math
    import seaborn as sns
    import numpy as np
    import pandas as pd

    #set the numerical features
    num_feat = [f for f in df.columns if df[f].dtype != 'object']
    
    plt.figure(figsize = (yoko * yoko_block, tate * math.ceil(len(num_feat) / yoko_block)))
    
    # violin plot
    if plot_type == "violin":
        if category != None:
            print("★ by " + category)
        
        for i in range(len(num_feat)):   
            plt.subplot(math.ceil((len(num_feat))/ yoko_block) + 1, yoko_block, i+1)
            sns.violinplot(data = df,y = num_feat[i],x = category,inner="quartile",split = True)
        plt.show()

    # box plot
    elif plot_type == "box":
        if category != None:
            print("★ by " + category)
        
        for i in range(len(num_feat)):   
            plt.subplot(math.ceil((len(num_feat))/ yoko_block) + 1, yoko_block, i+1)
            sns.boxplot(data = df,y = num_feat[i],x = category)
        plt.show()

    # dist plot
    elif plot_type == "hist":
        if category != None:
            print("★ by " + category)
                
        for i in range(len(num_feat)):
            if category == None:
                plt.subplot(math.ceil(len(num_feat) / yoko_block), yoko_block, i+1)
                df.iloc[:,i].dropna().plot(bins=bins_,kind="hist",alpha = 0.5)
                plt.xlabel(num_feat[i])
            else :
                plt.subplot(math.ceil(len(num_feat) / yoko_block), yoko_block, i+1)
                category_list = list(np.sort(df[category].unique()))
                for j in category_list:
                    if len(df[df[category] == j].iloc[:,i].dropna()) > 0:
                        df[df[category] == j].iloc[:,i].dropna().plot(bins=bins_,kind="hist",alpha = 0.5)
                        plt.xlabel(num_feat[i])
                        plt.legend(category_list)
        plt.show()

        
    # dist plot
    elif plot_type == "dist":
        if category != None:
            print("★ by " + category)
                
        for i in range(len(num_feat)):
            if category == None:
                plt.subplot(math.ceil(len(num_feat) / yoko_block), yoko_block, i+1)
                sns.distplot(df.iloc[:,i].dropna(),hist = False)
                plt.xlabel(num_feat[i])

            else :
                plt.subplot(math.ceil(len(num_feat) / yoko_block), yoko_block, i+1)
                category_list = list(np.sort(df[category].unique()))
                for j in category_list:
                    if len(df[df[category] == j].iloc[:,i].dropna()) > 0:
                        sns.distplot(df[df[category] == j].iloc[:,i].dropna(),hist = False,label = j)
        plt.show()
        

    else :
        print("★ choose the following plot_type")
        print("★ 'violin' or 'box' or or 'hist' or 'dist' ")