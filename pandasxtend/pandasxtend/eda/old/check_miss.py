#欠損値の先はどんな先がをチェック
def check_miss(df,yoko = 8,tate = 5,yoko_block = 3):
    """
    input   : pandas DataFrame
    output  : missing rate plot

    example : check_miss(train) 

    version :  0.4
    updated : 23-11-2018
    """

    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    temp1 = pd.DataFrame(df.count())
    temp1.columns = ["fill"]
    temp2 = pd.DataFrame(len(df) - df.count())
    temp2.columns = ["miss"]

    temp3 = temp1.join(temp2)
    temp3["all"] = temp3["fill"] + temp3["miss"]

    temp3["miss_rate"] = (temp3["miss"]/temp3["all"])

    temp4 = temp3.reset_index()[["index","miss_rate"]]
    temp4.columns = ["feat_name","miss_rate"]
    temp5 = temp4.sort_values(by = "miss_rate",ascending = False)
    
    plt.figure(figsize = (yoko, tate))
    plt.xlim([0,1])
    sns.barplot(x="miss_rate", y="feat_name", data=temp5)
    plt.show()

    return df[df.isnull().any(axis=1)]