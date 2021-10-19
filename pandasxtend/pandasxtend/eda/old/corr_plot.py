#特徴量の相関をチェック
def corr_plot(df,features,tate = 10,yoko = 10,cor_type = "pearson"):
    plt.figure(figsize=(yoko, tate))
    sns.heatmap(df[features].corr(cor_type),annot = True,fmt = ".3f",cmap="RdBu_r")
    plt.show()