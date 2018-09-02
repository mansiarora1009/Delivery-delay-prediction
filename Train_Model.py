#Import required packages

import pandas as pd
import numpy as np
import xgboost as xgb
import os
from xgboost import XGBRegressor

import pickle
import calendar
import datetime

from sklearn import cross_validation
from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

#Change working directory
path = os.getcwd()
os.chdir(path+'\Model and Scaler files')

#Define functions for feature engineering

#Actual cost function based on calculations provided by Amazon
def cost_fun(y_test, y_pred, y_actual):
    cost = 0
    min_cost = 0
    zero_cost = 0
    for index in range(0, y_actual.shape[0]):
        if y_pred.iloc[index][0] == 1:
            cost = cost + (30 - y_actual.iloc[index][0])*4
        else:
            cost = cost + 26*(y_actual.iloc[index][0])
        if y_test.iloc[index][0] == 1:
            min_cost = min_cost + (30 - y_actual.iloc[index][0])*4
            zero_cost = zero_cost + 26*(y_actual.iloc[index][0])
        else:
            min_cost = min_cost + 26*(y_actual.iloc[index][0])
            zero_cost = zero_cost + 26*(y_actual.iloc[index][0])
    return cost, min_cost, zero_cost

#Parameter selection based on correlation between themsevles and with the score_cont
def getmodelparams(data_clusters):

    cols = ['score_bin', 'score_cont', 'dayofweek0','dayofweek1','dayofweek2','dayofweek3','dayofweek4','dayofweek5','dayofweek6']
    dfs =  data_clusters.loc[:,['score_cont','tmp_mean','wind_mean','totalsvrprob_mean','wdir_mean','wvhgt_mean','apcp_mean','icea_mean','asnow_mean']]
    corr = dfs.corr()
    var_list = pd.DataFrame(list(abs(corr['score_cont']).nlargest(6))[1:])
    if sum(var_list[0:3][0])>=0.7:
        j=3
    elif sum(var_list[0:4][0])>=0.7:
        j=4
    else:
        j=5
    var_list = pd.DataFrame(list(abs(corr['score_cont']).nlargest(j+1).index)[1:])
    var_list_basic = list(var_list.iloc[:,0].apply(lambda x: x[:-5]))
    var_list_final=[]
    var_list_final.append(pd.DataFrame(var_list_basic).apply(lambda x: x+'_mean').values.T.tolist())
    var_list_final.append(pd.DataFrame(var_list_basic).apply(lambda x: x+'_mean7').values.T.tolist())
    var_list_final.append(pd.DataFrame(var_list_basic).apply(lambda x: x+'_stdev').values.T.tolist())
    #var_list_final.append(pd.DataFrame(var_list_basic).apply(lambda x: x+'_delta1').values.T.tolist())

    return cols + list(np.ravel(var_list_final))

#Scale the data for faster computation and to improve the performance
def scaledata(df):
    scaler = StandardScaler()
    scaler2 = scaler.fit(df)
    return  scaler2


#Read the data
parent_data = pd.read_csv('training_data_subset.csv')

#Define the clusters
clusters = [1,2,3,4,5,6,7,8,9,10]

#Define the seasons
season = [1,2,3,4,5,6]

#Fill in dummy values for all seasons
parent_data['season1']=0
parent_data['season2']=0
parent_data['season3']=0
parent_data['season4']=0
parent_data['season5']=0
parent_data['season6']=0

#Populate season data based on month variables
parent_data.loc[(parent_data['month12']==1)|(parent_data['month1']==1),'season1']=1
parent_data.loc[(parent_data['month2']==1)|(parent_data['month3']==1),'season2']=1
parent_data.loc[(parent_data['month4']==1)|(parent_data['month5']==1),'season3']=1
parent_data.loc[(parent_data['month6']==1)|(parent_data['month7']==1),'season4']=1
parent_data.loc[(parent_data['month8']==1)|(parent_data['month9']==1),'season5']=1
parent_data.loc[(parent_data['month10']==1)|(parent_data['month11']==1),'season6']=1

#Create dataframes to store tuning parameters and intermediate results
result = pd.DataFrame(np.zeros((9*6*10*60, 7)))
vars_df = pd.DataFrame(np.zeros((60, 40)))

index2=0

#Parameter tuning and model generation
for i in clusters:
    for j in season:
        data_clusters = parent_data.loc[(parent_data['cluster'] == i) & ((parent_data['season' + str(j)] == 1))]    #Filter the data

        cols = getmodelparams(data_clusters)    #Get attributes for the model based on subset data

        lencols = len(cols)

        #Define the standard scaler, transform the training data
        data_clusters = data_clusters[cols]
        datax = data_clusters.iloc[:, 2: ]
        datax_right = pd.DataFrame(datax.iloc[:, 7: ])
        datax_left=datax.iloc[:,0:7]
        datax_left.reset_index(inplace=True)
        datax_left=datax_left.iloc[:,1:]
        datax_right.reset_index(inplace=True)
        datax_right=datax_right.iloc[:,1:]
        scaler = scaledata(datax_right)
        datax_right = pd.DataFrame(scaler.transform(datax_right))
        datax_temp = pd.concat((datax_left, datax_right), axis = 1, ignore_index = True)
        datax_temp.columns = cols[2: ]

        datay = data_clusters.iloc[:, 0:2]

        #Split the data into train and test - a high training samples is chosen because of the variance in the data.
        #Ideally results should be derived from an ensemble of models
        X_train, X_test, y_train, y_test =train_test_split(datax_temp, datay, test_size=0.08)
        y_test_cont = pd.DataFrame(y_test.iloc[:,1])

        index = 0

        threshold = [0.2,0.25, 0.3,0.35, 0.4,0.45, 0.5,0.55,0.6]

        #Optimize the depth and positive weight parameters
        for d in range(6, 10):          #Loop over range - max value has been chosen so as to ensure that we dont overfit the data
            for w in range(10, 30, 3):  #Loop over weight - max value has been chosen based on industry standards - wt = num. of negative samples/num. of positive samples
                xgbmodel = xgb.XGBClassifier(silent=False, max_depth= d, learning_rate=0.1, scale_pos_weight = w)
                xgbmodel.fit(X_train, np.ravel(y_train.iloc[:,0]), eval_metric = 'auc')
                y_pred = pd.DataFrame(pd.DataFrame(xgbmodel.predict_proba(X_test)).iloc[:, 1])
                temp = pd.DataFrame(np.where(y_pred.iloc[:, 0] > 0.15, 1, 0))

                #Store intermediate parameter tuning results
                result.iloc[index, 0] = d
                result.iloc[index, 1] = w
                result.iloc[index, 2] = 0.2
                result.iloc[index, 3], result.iloc[index, 4], result.iloc[index, 5] = cost_fun(y_test[['score_bin']], temp, y_test_cont)
                result.iloc[index, 6] = i
                index = index + 1

                #Print values to monitor progress
                print(i, d, w)

        cluster_result = result.loc[result[6] == i]
        cluster_result = cluster_result.sort_values(by = [3], ascending = True)

        #Select optimum tuning parameters
        opt_d = cluster_result.iloc[0, 0]
        opt_w = cluster_result.iloc[0, 1]
        optim_threshold = optim_threshold + [cluster_result.iloc[0, 2]]

        #Fit a model using optimum parameters, save it to dist using pickle
        xgbmodel = xgb.XGBClassifier(silent=False, max_depth= int(opt_d), learning_rate=0.1, scale_pos_weight = int(opt_w))
        xgbmodel.fit(X_train, np.ravel(y_train.iloc[:,0]), eval_metric = 'auc')
        #y_pred = pd.DataFrame(pd.DataFrame(xgbmodel.predict_proba(X_test)).iloc[:, 1])
        pickle.dump(xgbmodel, open("new_model_finetune_cluster_"+str(i)+"_season_"+str(j)+".dat", "wb"))

        #Save the standard scaler to disk using pickle
        joblib.dump(scaler, 'scaler_cluster_'+str(i)+'_season_'+str(j)+'.pkl')

        #Save the model parameters to disk, so that it can be re-read while testing the model
        vars_df.iloc[index2, 0] = i
        vars_df.iloc[index2, 1] = j
        vars_df.iloc[index2, 2] = lencols
        vars_df.iloc[index2, 3:lencols+1] = cols[2: ]
        vars_df.iloc[index2, -5] = cluster_result.iloc[0,0]
        vars_df.iloc[index2, -4] = cluster_result.iloc[0,1]
        vars_df.iloc[index2, -3] = cluster_result.iloc[0,3]
        vars_df.iloc[index2, -2] = cluster_result.iloc[0,4]
        vars_df.iloc[index2, -1] = cluster_result.iloc[0,5]
        index2 = index2 + 1

#Write model parameters to disk
vars_df.to_csv('model_season_parameters.csv')

#END

