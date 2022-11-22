def test_sd(data_train, data_test, L, n_users, M_ts, dim, days, plot_all):
    flattened_obs, M_ts = get_flattened_obs(data_train, L=L)

    window = M_ts
    lst_U_sd = []
    for i in range(n_users):
        data = flattened_obs[:,i*window:(i+1)*window]
        U,_,_ = np.linalg.svd(data)
        U = U[:,:dim]
        lst_U_sd.append(U)
        
    model_sd = mSSA(rank = dim, normalize = False, L=L)
    model_sd.update_model(data_train)

    P_sd = flattened_obs
    P_sd_hat = []
    y_sd = []
    y_true = P_sd[-1,:]
    P_tilde_sd_hat = []
    imputation_model_score_sd = []
    for i in range(n_users):
        P_i_sd = P_sd[:,int(i*window):int((i+1)*window)]
        P_i_sd_hat = lst_U_sd[i].dot(lst_U_sd[i].T.dot(P_i_sd)); P_sd_hat.append(P_i_sd_hat)
        y_i_sd = P_i_sd_hat[-1,:]; y_sd.append(y_i_sd)
        y_i_true = P_i_sd[-1,:]
        P_i_tilde_sd_hat = P_i_sd_hat[:-1,:]; P_tilde_sd_hat.append(P_i_tilde_sd_hat)
        imputation_model_score_sd.append(r2_score(P_i_sd.flatten('F'),P_i_sd_hat.flatten('F'))) # verified same as imputation_model_score)
    imputation_model_score_sd = np.array(imputation_model_score_sd)
    P_sd_hat = np.hstack(P_sd_hat)
    y_sd = np.hstack(y_sd)
    P_tilde_sd_hat = np.hstack(P_tilde_sd_hat)
    print("imputation score:", imputation_model_score_sd.mean())

    # verify weights_admm using sklearn
    from sklearn.linear_model import LinearRegression
    reg = LinearRegression(fit_intercept=False).fit(P_tilde_sd_hat.T, y_sd)
    weights_sd = reg.coef_
    
    model_sd.ts_model.models[0].weights = weights_sd
    
    actual, predictions_sd = predict_one_day(data_test, model_sd)

    Y = actual[:,:]
    Y_h_sd = predictions_sd.T[:,:]
    mse_sd = np.sqrt(np.mean(np.square(Y[:7*days]-Y_h_sd[:7*days])))
    print ('Forecasting accuracy (RMSE) my:',mse_sd)
    
    if plot_all:
        npar = np.arange(0,n_users)
    else: npar = [1]
    for i in npar:
        plt.figure()
        plt.title('forecasting the next seven days for %s'%data_test.columns[i])
#         plt.plot(predictions[i,:24*days],label= 'mSSA',color='green')
#         plt.plot(predictions_my[i,:24*days],label= 'FedmSSA',color='orange')
        plt.plot(predictions_sd[i,:24*days],label= 'sd',color='pink')
        plt.plot(actual[:24*days,i],label = 'actual',color='blue')
        plt.legend()
    plt.show()
    
    return Y, Y_h_sd, weights_sd