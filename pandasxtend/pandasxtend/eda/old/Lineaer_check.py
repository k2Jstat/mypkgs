def linearity_check(df,feat,label,cut_length = 10):
        import numpy as np
    
    bun = df.copy()
    k = int(round(100/cut_length,0))
    
    #print(k)
    per = [np.nanpercentile(bun[feat],k) for k in range(0,100 + k , k)]
    bins = list(np.unique(per))
    #labels = [str(100 + j)[1:3] + ": 上位" + str(k * j)  + "%" for j in range(1,len(bins))]
    #labels = [str(100 + j)[1:3] + bins[j] for j in range(1,len(bins))]
    labels = [str(100 + j)[1:3] + ": " + str(bins[j-1]) + " --< " + str(bins[j])for j in range(1,len(bins))]
    #下位が入らないので、補正
    bins[0] -= 0.001
    bun["cat_" + feat] = pd.cut(bun[feat],bins = bins,labels = labels)
    #bun["cat_" + feat] = pd.qcut(bun[feat],cut_length)
    
    var = "cat_" + feat

    odds_label = bun.groupby(var).mean()[[label]]
    odds_label.columns = [label]

    odds_label["ln_odds"] = np.log(odds_label[label] / (1 - odds_label[label]))

    start = odds_label["ln_odds"].iloc[0]
    goal = odds_label["ln_odds"].iloc[-1]
    dif = goal - start

    lin_data = [dif/(len(bins) - 2) * i + start for i in range(0,len(bins) - 1)]

    df_plot = pd.DataFrame(data = lin_data ,index = odds_label.index,columns = ["linearity"])
    temp_l = odds_label.join(df_plot)
    temp_l["linearity"].plot(kind = "line")
    temp_l["ln_odds"].plot(kind = "line",marker = "o",linestyle = " ",figsize = (10,5),markersize = 12)
    plt.title(feat + "　別対数オッズ")
    plt.xticks(rotation = 45)
    plt.show()