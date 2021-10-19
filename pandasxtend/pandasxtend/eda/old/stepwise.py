def STEPWISE(X,y,null_model = False):
#def STEPWISE(X,y,clf,null_model = False):
    #stepwise forward
    X_ = X.copy()
    y_ = y.copy()
    #feat_all = ["intercept"]

    if null_model == True:
        feat_all = ["intercept"]
        add_feat = X_.columns
        #feat_allはintercept開始の切片から
        feat_all.extend(add_feat)
        feat_best = []
        X_["intercept"] = 1
        #null_model=Falseを使うときにはfit_interceptをFalseにしておくこと

    else :
        feat_all = X_.columns
        feat_best = []

    AIC_best = 0
    
    clf_list = []
    
    for feat in feat_all:
        #現時点でbestの特徴量に新しく試す特徴量を追加
        feat_list = feat_best.copy()
        feat_list.extend([feat])
        
        #学習
        clf_ = LogisticRegression()
        clf_.fit(X_[feat_list],y_)
        
        #AICの計算 ========================================================
        pred = clf_.predict_proba(X_[feat_list])[:,1]

        yudo_df = pd.DataFrame(data = {"true":y_,"pred":pred})

        #対数尤度の計算
        yudo_df["yudo"] = yudo_df["true"] * np.log(yudo_df["pred"]) + (1 - yudo_df["true"]) * np.log(1-yudo_df["pred"])
        LL = yudo_df["yudo"].sum()
        AIC = -2 * (LL - (len(feat_list) + 1)) #切片分パラメータ分足す

        print(feat_list);print("AIC : " + str(AIC));print("   ")
        # ================================================================
        #AICが改善されてたら、AICのベストスコアと、その際の特徴量をセット
        if AIC_best == 0:
            AIC_best = AIC.copy()
            feat_best = feat_list.copy()
            clf_best = clf_

        elif AIC < AIC_best:
            AIC_best = AIC.copy()
            feat_best = feat_list.copy()
            clf_best = clf_
            
        del clf_
            
    print("    ")
    print("conclusion * * * ")
    print("best AIC : " + str(AIC_best))
    print("best feature set : " + str(feat_best))
    print("* * * * * * * * *")
    
    return AIC_best,feat_best,clf_best
