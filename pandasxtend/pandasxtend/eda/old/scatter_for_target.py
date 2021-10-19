def scatter_for_target(df,target,hue = None,yoko = 5,tate = 6,yoko_block = 6,marker_size = 30):
    """
    input   : pandas DataFrame
    output  : scatter plot

    example : scatter_for_target(train,"target")
    """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import math
    
    # make the numeric feature list
    num_feat = [f for f in df.columns if df[f].dtype != 'object']
    
    # make plot box
    plt.figure(figsize = (yoko * yoko_block, tate * math.ceil(len(num_feat) / yoko_block)))
    print("★ " + target)
    #
    for i in range( len( num_feat ) ): 
        #print(i)
        plt.subplot( math.ceil( ( len(num_feat) )/ yoko_block) + 1, yoko_block, i+1)
        
        if hue == None:
            plt.scatter(x = num_feat[i],
                        y = target,
                        data = df,
                        s = marker_size,
                        marker = "o",
                        cmap = plt.get_cmap("tab20")
                        # edgecolors = "black",
                        # linewidths = 0.2
                        )
            plt.xlabel(num_feat[i])
            plt.ylabel(target)

        else : 
            class_ = list(np.sort(df[hue].unique()))

            if len(class_) > 20:
                print("★ hue is over !!!")
                
            else:
                for j in class_:
                    plt.scatter(x = num_feat[i],
                                y = target,
                                data = df[df[hue] == j],
                                s = marker_size,
                                marker = "o",
                                cmap = plt.get_cmap("tab20")
                                # edgecolors = "black",
                                # linewidths = 0.2
                                )
                plt.xlabel(num_feat[i])
                plt.ylabel(target)
                plt.legend(class_)
    plt.show()
