#from google.colab import files
import io
import pandas as pd
import numpy as np
import seaborn as sns
import  matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import math
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import tensorflow as tf
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn import datasets, ensemble
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from scipy.stats import norm
from scipy import stats
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from xgboost.sklearn import XGBRegressor
import datetime
from sklearn.model_selection import GridSearchCV



train = pd.read_csv('train.csv', index_col= 'Id')
test = pd.read_csv('test.csv', index_col = 'Id')

train.describe()
train['SalePrice']

train_check = train.iloc[0:10,:]
train_null_check = train.isnull().sum()

#-------data glance
#check relationship to SalePrice
corrmat = train.corr()
f, ax = plt.subplots(figsize= (12,9))
sns.heatmap(corrmat, vmax=0.8, square=True)

sale_rel = pd.DataFrame({'correlation':corrmat['SalePrice']})
sale_rel = sale_rel.sort_values(by='correlation',ascending=False)

#append correlation >0.2 to list
feature_list=[]
for i in sale_rel.index:
    if sale_rel.loc[i,'correlation'] > 0.2:
        feature_list.append(i)
feature_list.remove('SalePrice')
#remove '2ndFlrSF' (from XGB9)
feature_list.remove('2ndFlrSF')

train_f = train[feature_list]
#check null values
null_list = train[feature_list].isnull().sum()
null_list = null_list[null_list != 0].index

num_list = [col for col in feature_list if train[col].dtype == 'float64' or 'int64']
cat_list = [col for col in feature_list if train[col].dtype == 'object']

#setting train set
train_set = train[feature_list]
train_set[null_list] #all columns with null values are numerical columns
#fillin null values
numerical_transfer = SimpleImputer(strategy='most_frequent')
train_set[null_list] = numerical_transfer.fit_transform(train_set[null_list])

#explore contents of train_set
var = 'GrLivArea'
data = pd.concat([train['SalePrice'], train_set[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000))

outliers = train.loc[(train['GrLivArea'] >4000) & (train['SalePrice'] < 200000) ]
train_set = train_set.drop(train_set.index[[outliers.index]] )

#Outliers checking
for i in train_set.columns:   
    var = i
    data = pd.concat([train['SalePrice'], train_set[var]], axis=1)
    data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000))

#Remove outliers - LotArea
train_set = train_set.drop(train_set.index[[train.loc[train['LotArea']>100000].index]] )

#Remove outliers - TotalBsmtSF
train_set = train_set.drop(train_set.index[[train.loc[train['TotalBsmtSF']>6000].index]] )
#Remove outliers - LotFrontage
train_set = train_set.drop(train_set.index[[train.loc[train['LotFrontage']>300].index]] )
#Remove outliers - 1stFlrSF
train_set = train_set.drop(train_set.index[[train.loc[train['1stFlrSF']>4000].index]] )
#Remove outliers - BsmtFinSF1
train_set = train_set.drop(train_set.index[[train.loc[train['BsmtFinSF1']>5000].index]] )

#Remove outliers - GarageCars
train_set = train_set.drop(train_set.index[[train.loc[train['GarageCars']==4].index]] )


#OneHotEncoder
OneHotList = ['FullBath','Fireplaces','HalfBath','BsmtFullBath']
#-FullBath
FullBath = train_set['FullBath'].copy()
FullBath = OneHotEncoder(drop='first').fit_transform(train_set['FullBath'].values.reshape(-1,1)).toarray()
FullBath = pd.DataFrame(FullBath, columns = ['FullBath_1','FullBath_2','FullBath_3'])
FullBath.index = train_set.index
#-Fireplaces
Fireplaces = train_set['Fireplaces'].copy()
Fireplaces = OneHotEncoder(drop='first').fit_transform(train_set['Fireplaces'].values.reshape(-1,1)).toarray()
Fireplaces = pd.DataFrame(Fireplaces, columns = ['Fireplaces_1','Fireplaces_2','Fireplaces_3'])
Fireplaces.index = train_set.index
#-HalfBath
HalfBath = train_set['HalfBath'].copy()
HalfBath = OneHotEncoder(drop='first').fit_transform(train_set['HalfBath'].values.reshape(-1,1)).toarray()
HalfBath = pd.DataFrame(HalfBath, columns = ['HalfBath_1','HalfBath_2'])
HalfBath.index = train_set.index
#-BsmtFullBath
BsmtFullBath = train_set['BsmtFullBath'].copy()
BsmtFullBath = OneHotEncoder(drop='first').fit_transform(train_set['BsmtFullBath'].values.reshape(-1,1)).toarray()
BsmtFullBath = pd.DataFrame(BsmtFullBath, columns = ['BsmtFullBath_1','BsmtFullBath_2','BsmtFullBath_3'])
BsmtFullBath.index = train_set.index
#concat to original data
concatList = [train_set, FullBath,Fireplaces,HalfBath,BsmtFullBath]  # List of your dataframes
train_set = pd.concat(concatList, axis=1)
train_set = train_set.drop(OneHotList, axis = 1)

#making train dataset with the same number of columns
train_set['FullBath_4'] = 0
train_set['Fireplaces_4'] = 0


#Split training and test
X = train_set
index_list = pd.Series(train_set.index)
y = train.loc[index_list]['SalePrice']

X_f = X[feature_list]

X_train, X_valid,  y_train, y_valid = train_test_split(train_set, y, test_size = 0.25, random_state=1) 


#RandomForestRegressor
RMSE = {} 
for i in range(100,1001, 50):
    model = RandomForestRegressor(n_estimators=i, random_state=10, criterion='mse')      
    model.fit(X_train, y_train)    
    y_predict = model.predict(X_valid)
    
    mse = mean_absolute_error(y_valid, y_predict)
    print(f'n_estimators={i}:','RMSE:', math.sqrt(mean_squared_error(y_valid, y_predict)))
    RMSE[f'n_estimators={i}'] = math.sqrt(mean_squared_error(y_valid, y_predict))

sorted_RMSE_RFR = sorted(RMSE.items(), key=lambda kv: kv[1]) 

model = RandomForestRegressor(n_estimators=650, random_state=10, criterion='mse')      
model.fit(X_train, y_train)    
y_predict = model.predict(X_valid)
rms = sqrt(mean_squared_error(y_valid, y_predict))

#predict test dataset
test = test[feature_list]

num_list_test = [col for col in feature_list if test[col].dtype == 'float64' or 'int64']
cat_list_test = [col for col in feature_list if test[col].dtype == 'object']

#check null values
null_list_test = test[feature_list].isnull().sum()
null_list_test = null_list_test[null_list_test != 0].index


numerical_transfer = SimpleImputer(strategy='most_frequent')
test[null_list_test] = numerical_transfer.fit_transform(test[null_list_test])


test_predict = model.predict(test)


# Save test predictions to file
output = pd.DataFrame({'Id': test.index,
                       'SalePrice': test_predict})
output.to_csv('submission_RF.csv', index=False)



#XGRegressor
xgb1 = XGBRegressor()
parameters = {'nthread':[4], #when use hyperthread, xgboost may become slower
              'objective':['reg:linear'],
              'booster':['gbtree'],
              'eval_metric':['rmse'],
              'learning_rate': [.03, 0.05, .07], #so called `eta` value
              'max_depth': [5, 6, 7],
              'min_child_weight': [4],
              'silent': [1],
              'subsample': [0.7],
              'colsample_bytree': [0.7],
              'n_estimators': [500,700,1000]}

xgb_grid = GridSearchCV(xgb1,
                        parameters,
                        cv = 2,
                        n_jobs = 5,
                        verbose=True)

xgb_grid.fit(X_train, y_train)

y_predict_xgb = xgb_grid.predict(X_valid)
rms = sqrt(mean_squared_error(y_valid, y_predict_xgb))

#predict test dataset
#fillin na values
test = test[feature_list]

num_list_test = [col for col in feature_list if test[col].dtype == 'float64' or 'int64']
cat_list_test = [col for col in feature_list if test[col].dtype == 'object']

#check null values
null_list_test = test[feature_list].isnull().sum()
null_list_test = null_list_test[null_list_test != 0].index


numerical_transfer = SimpleImputer(strategy='most_frequent')
test[null_list_test] = numerical_transfer.fit_transform(test[null_list_test])


#OneHotEncoder
OneHotList = ['FullBath','Fireplaces','HalfBath','BsmtFullBath']
#-FullBath
FullBath_t = test['FullBath'].copy()
FullBath_t = OneHotEncoder(drop='first').fit_transform(test['FullBath'].values.reshape(-1,1)).toarray()
FullBath_t = pd.DataFrame(FullBath_t, columns = ['FullBath_1','FullBath_2','FullBath_3','FullBath_4' ])
FullBath_t.index = test.index
#-Fireplaces
Fireplaces_t = test['Fireplaces'].copy()
Fireplaces_t = OneHotEncoder(drop='first').fit_transform(test['Fireplaces'].values.reshape(-1,1)).toarray()
Fireplaces_t = pd.DataFrame(Fireplaces_t, columns = ['Fireplaces_1','Fireplaces_2','Fireplaces_3','Fireplaces_4'])
Fireplaces_t.index = test.index
#-HalfBath
HalfBath_t = test['HalfBath'].copy()
HalfBath_t = OneHotEncoder(drop='first').fit_transform(test['HalfBath'].values.reshape(-1,1)).toarray()
HalfBath_t = pd.DataFrame(HalfBath_t, columns = ['HalfBath_1','HalfBath_2'])
HalfBath_t.index = test.index
#-BsmtFullBath
BsmtFullBath_t = test['BsmtFullBath'].copy()
BsmtFullBath_t = OneHotEncoder(drop='first').fit_transform(test['BsmtFullBath'].values.reshape(-1,1)).toarray()
BsmtFullBath_t = pd.DataFrame(BsmtFullBath_t, columns = ['BsmtFullBath_1','BsmtFullBath_2','BsmtFullBath_3'])
BsmtFullBath_t.index = test.index
#concat to original data
concatList = [test, FullBath_t,Fireplaces_t,HalfBath_t,BsmtFullBath_t]  # List of your dataframes
test = pd.concat(concatList, axis=1)
test = test.drop(OneHotList, axis = 1)


test = test.reindex(columns = train_set.columns)
test_predict_xgb_grid = xgb_grid.predict(test)


# Save test predictions to file
output = pd.DataFrame({'Id': test.index,
                       'SalePrice': test_predict_xgb_grid})
output.to_csv('submission_XGB_grid.csv', index=False)



xgb9 = XGBRegressor(booster='gbtree',
                    objective= 'reg:linear',
                    eval_metric='rmse',
                    gamma = 0.1,
                    min_child_weight= 3,
                    max_depth= 5,
                    subsample= 0.8,
                    colsample_bytree= 0.8,
                    tree_method= 'exact',
                    learning_rate=0.1,
                    n_estimators=200,
                    nthread=4,
                    scale_pos_weight=1,
                    reg_alpha=0.05,                           
                    seed=27)


xgb9.fit(X_train, y_train)

y_predict_xgb9 = xgb9.predict(X_valid)
rms = sqrt(mean_squared_error(y_valid, y_predict_xgb9))

fig, ax = plt.subplots(1, 1, figsize=(8, 13))
xgb.plot_importance(xgb9, max_num_features=20, height=0.5, ax=ax)


#predict test dataset
test = test[feature_list]

num_list_test = [col for col in feature_list if test[col].dtype == 'float64' or 'int64']
cat_list_test = [col for col in feature_list if test[col].dtype == 'object']

#check null values
null_list_test = test[feature_list].isnull().sum()
null_list_test = null_list_test[null_list_test != 0].index


numerical_transfer = SimpleImputer(strategy='most_frequent')
test[null_list_test] = numerical_transfer.fit_transform(test[null_list_test])


test_predict_xgb9 = xgb9.predict(test)


# Save test predictions to file
output = pd.DataFrame({'Id': test.index,
                       'SalePrice': test_predict_xgb9})
output.to_csv('submission_XGB9.csv', index=False)
