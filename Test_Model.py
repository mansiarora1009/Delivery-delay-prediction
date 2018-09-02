#Import required packages
import pandas as pd
import numpy as np
import xgboost as xgb
import os
import pickle
import calendar
import datetime
from sklearn import cross_validation
from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler

#Change working directory
import os
path = os.getcwd()
os.chdir(path+'\Model and Scaler files')

#Load test data
test_data = pd.read_csv('Final_testing_data.csv')

#Load parameter file
cols = pd.read_csv('model_season_parameters.csv') #Read columns for each model - cluster-month combination
cols = cols.iloc[:,1:]
test_data['affected'] = 0

#Create variables for each season
test_data['season1']=0
test_data['season2']=0
test_data['season3']=0
test_data['season4']=0
test_data['season5']=0
test_data['season6']=0

test_data.loc[(test_data['month12']==1)|(test_data['month1']==1),'season1']=1
test_data.loc[(test_data['month2']==1)|(test_data['month3']==1),'season2']=1
test_data.loc[(test_data['month4']==1)|(test_data['month5']==1),'season3']=1
test_data.loc[(test_data['month6']==1)|(test_data['month7']==1),'season4']=1
test_data.loc[(test_data['month8']==1)|(test_data['month9']==1),'season5']=1
test_data.loc[(test_data['month10']==1)|(test_data['month11']==1),'season6']=1

#Loop through the test data, load the required model, predict the outcomes and write it to a data frame
for cluster in [1,2,3,4,5,6,7,8,9,10]:
    for s in [2]:

        model= pickle.load(open("new_model_finetune_cluster_"+str(cluster)+"_season_"+str(s)+".dat", "rb"))

        season = 'season'+str(s)

        cols2 = cols.loc[((cols['0']==cluster) & (cols['1']==s)),:]
        lencols = int(cols2.iloc[0,2])
        cols2 = cols2.iloc[:,3:lencols+1]
        cols2 = list(np.ravel(cols2.values.T.tolist()))
        scaler = joblib.load('scaler_cluster_'+str(cluster)+'_season_'+str(s)+'.pkl')

        if len(test_data.loc[(test_data['cluster']==cluster) & (test_data[season]==1),cols2])>0:
            datax = test_data.loc[(test_data['cluster']==cluster) & (test_data[season]==1),cols2]
            datax_right = pd.DataFrame(datax.iloc[:, 7: ])
            datax_left=datax.iloc[:,0:7]
            datax_left.reset_index(inplace=True)
            datax_left=datax_left.iloc[:,1:]
            datax_right.reset_index(inplace=True)
            datax_right=datax_right.iloc[:,1:]
            datax_right = pd.DataFrame(scaler.transform(datax_right))
            datax_temp = pd.concat((datax_left, datax_right), axis = 1, ignore_index = True)
            datax_temp.columns = cols2
            test_data.loc[(test_data['cluster']==cluster) & (test_data[season]==1),'affected'] = np.array(pd.DataFrame(model.predict_proba(datax_temp)).iloc[:, 1])

        else:
            pass

#Write the outcome to a CSV

#Threshold the data to get a binary output,
#threshold here is decided on based on the logic procided by amazon. Scores >=4 are 1,
#which is why a threshold slightly favouring the postive classes is selected (0.1)
test_data['affected_bin'] = 0
test_data.loc[test_data['affected']>0.1,'affected_bin'] = 1

final_test_file = test_data[['zip_code', 'date', 'affected_bin']]
final_test_file.columns = ['zip_code', 'date', 'affected']

final_test_file.to_csv('final_test.csv')


