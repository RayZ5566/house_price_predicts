
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


#from google.colab import files

train = pd.read_csv('train.csv', index_col= 'Id')
test = pd.read_csv('test.csv', index_col = 'Id')

train.describe()
train['SalePrice']

train_check = train.iloc[0:10,:]
train_null_check = train.isnull().sum()

#train_select = train['LotArea','OverallQual','OverallCond','YearBuilt','SaleCondition']

cat_cols = [col for col in train.columns if
#            train[col].nunique() <= 10 and
            train[col].dtype == 'object' or
            col == 'YearBuilt' or
            col == 'YearRemodAdd' or
            col == 'GarageYrBlt' or
            col == 'YrSold']

num_cols = [col for col in train.columns if
            train[col].dtype in ['int64', 'float64'] and
            col != 'SalePrice' and
            col != 'YearBuilt' and
            col != 'YearRemodAdd' and
            col != 'GarageYrBlt' and
            col != 'YrSold']



my_cols = cat_cols + num_cols

X = train[my_cols].copy()
y = train['SalePrice']


# Preprocessing for numerical data
numerical_transfer = SimpleImputer(strategy='constant')

X[num_cols] = numerical_transfer.fit_transform(X[num_cols])
test[num_cols] = numerical_transfer.transform(test[num_cols])

# Preprocessing for categorical data
cat_transfer = SimpleImputer(strategy='most_frequent')
X[cat_cols] = cat_transfer.fit_transform(X[cat_cols])
test[cat_cols] = cat_transfer.transform(test[cat_cols])



#label_encoder
X_label = X.copy()
test_label = test.copy()
label_encoder = LabelEncoder()
for cols in cat_cols:
    X_label[cols] = LabelEncoder().fit_transform(X[cols])
    test_label[cols] = LabelEncoder().fit_transform(test[cols])


X_try = X_label[poly_col].sum(axis=1).values.reshape(-1,1)
test_try = test_label[poly_col].sum(axis=1).values.reshape(-1,1)
#Fitting Polynomial Regression to the dataset
X_train, X_valid, y_train, y_valid = train_test_split(X_try, y, test_size = 0.25, random_state = 1)
poly_col = ['Functional','MSZoning','LotArea','HouseStyle','OverallQual','YearBuilt','OverallCond','Utilities']
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
x_poly = poly_reg.fit_transform(X_train)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(x_poly, y_train)


y_predict_p = lin_reg_2.predict(poly_reg.transform(X_valid))
rms = sqrt(mean_squared_error(y_valid, y_predict_p))


test = test_label[poly_col]
test_predict = lin_reg_2.predict(poly_reg.transform(test_try))


# Save test predictions to file
output = pd.DataFrame({'Id': test.index,
                       'SalePrice': test_predict})
output.to_csv('submission_p.csv', index=False)


#RandomForest----------------------------------------------------#
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size = 0.25, random_state = 1)

# Preprocessing for numerical data
numerical_transfer = SimpleImputer(strategy='constant')

# Preprocessing for categorical data
categorical_transfer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
     ('onehot', OneHotEncoder(handle_unknown='ignore'))
     ])


preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transfer, num_cols),
        ('cat', categorical_transfer, cat_cols)
        ])

#model - RandomForestRegressor
RMSE = {} 
for i in range(100,1001, 50):
    model = RandomForestRegressor(n_estimators=i, random_state=1, criterion='mse')
    
    clf = Pipeline(steps=[('preprocessor', preprocessor),
                          ('model', model)
                          ])
    
    clf.fit(X_train[poly_col], y_train)
    
    y_predict = clf.predict(X_valid[poly_col])
    
    mse = mean_absolute_error(y_valid, y_predict)
    print(f'n_estimators={i}:','RMSE:', math.sqrt(mean_squared_error(y_valid, y_predict)))
    RMSE[f'n_estimators={i}'] = math.sqrt(mean_squared_error(y_valid, y_predict))

sorted_RMSE_RFR = sorted(RMSE.items(), key=lambda kv: kv[1])




#predict test_dataset

model = RandomForestRegressor(n_estimators=450, random_state=0, criterion='mse')
    
clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('model', model)
                      ])

clf.fit(X_train, y_train)

y_predict = clf.predict(X_valid)

mse = mean_absolute_error(y_valid, y_predict)
rmse = math.sqrt(mean_squared_error(y_valid, y_predict))
test = test[my_cols]
test_predict = clf.predict(test)


# Save test predictions to file
output = pd.DataFrame({'Id': test.index,
                       'SalePrice': test_predict})
output.to_csv('submission_r.csv', index=False)

#model - sequential---------------------------------------------------#
# Preprocessing for categorical data
cat_transfer = SimpleImputer(strategy='most_frequent')
X[cat_cols] = cat_transfer.fit_transform(X[cat_cols])
test[cat_cols] = cat_transfer.transform(test[cat_cols])



#label_encoder
X_label = X.copy()
test_label = test.copy()
label_encoder = LabelEncoder()
for cols in cat_cols:
    X_label[cols] = LabelEncoder().fit_transform(X[cols])
    test_label[cols] = LabelEncoder().fit_transform(test[cols])



X_train_s, X_valid_s, y_train_s, y_valid_s = train_test_split(X_label, y, test_size = 0.25, random_state = 1)


# Preprocessing for numerical data
numerical_transfer = SimpleImputer(strategy='constant')

X_train_s[num_cols] = numerical_transfer.fit_transform(X_train_s[num_cols])
X_valid_s[num_cols] = numerical_transfer.transform(X_valid_s[num_cols])
test[num_cols] = numerical_transfer.transform(test[num_cols])




network = Sequential()

network.add(Dense(units=256, activation='relu', kernel_initializer = 'normal', input_dim=32))
network.add(Dense(units=16, activation='relu', kernel_initializer = 'normal'))
network.add(Dense(units=16, activation='relu', kernel_initializer = 'normal'))
network.add(Dense(units=16, activation='relu', kernel_initializer = 'normal'))
network.add(Dense(units= 1, activation='sigmoid', kernel_initializer = 'normal'))
network.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])


network.fit(X_train_s[poly_col], y_train_s, batch_size = 5, epochs = 100)
y_predict_s = network.predict(X_valid_s[poly_col])






from sklearn.metrics import mean_squared_error
from math import sqrt
mse_p = mean_absolute_error(y_valid_s, y_predict_p )
rms = sqrt(mean_squared_error(y_valid_s, y_predict_p ))



#RandomForest

poly_col = ['Neighborhood','BldgType','MSSubClass','MSZoning','LotArea','HouseStyle','OverallQual','YearBuilt','OverallCond','Utilities','ExterQual','ExterCond','BsmtCond','BsmtFinType1','BsmtFinType2','HeatingQC',
            'KitchenQual','Functional','GarageQual','PoolQC','MiscVal']
            #'SaleType','SaleCondition','YearRemodAdd','MasVnrArea','TotalBsmtSF','1stFlrSF','2ndFlrSF','LowQualFinSF','FullBath','FireplaceQu','PoolQC']





X_train_s_r = X_train_s[poly_col]

RMSE = {} 
for i in range(100,1001, 50):
    model = RandomForestRegressor(n_estimators=i, random_state=10, criterion='mse')      
    model.fit(X_train_s_r, y_train)    
    y_predict = model.predict(X_valid_s[poly_col])
    
    mse = mean_absolute_error(y_valid_s, y_predict)
    print(f'n_estimators={i}:','RMSE:', math.sqrt(mean_squared_error(y_valid_s, y_predict)))
    RMSE[f'n_estimators={i}'] = math.sqrt(mean_squared_error(y_valid_s, y_predict))

sorted_RMSE_RFR = sorted(RMSE.items(), key=lambda kv: kv[1]) 

model = RandomForestRegressor(n_estimators=150, random_state=0, criterion='mse')      
model.fit(X_train_s_r, y_train)    
y_predict = model.predict(X_valid_s[poly_col])
rms = sqrt(mean_squared_error(y_valid_s, y_predict))

test = test_label[poly_col]
test_predict = model.predict(test)


# Save test predictions to file
output = pd.DataFrame({'Id': test.index,
                       'SalePrice': test_predict})
output.to_csv('submission_r.csv', index=False)

#predict test_dataset

test_predict_p = lin_reg_2.predict(poly_reg.transform(test_label[poly_col]))


# Save test predictions to file
output = pd.DataFrame({'Id': test.index,
                       'SalePrice': test_predict_p})
output.to_csv('submission_p.csv', index=False)


X_train, X_valid, y_train, y_valid = train_test_split(X_label, y, test_size = 0.25, random_state = 1)

numerical_transfer = SimpleImputer(strategy='constant')

X[num_cols] = numerical_transfer.fit_transform(X[num_cols])
X_valid = numerical_transfer.transform(X_valid[num_cols])
test[num_cols] = numerical_transfer.transform(test[num_cols])

chek = X_train.isnull().sum()

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
X_train = sc_x.fit_transform(X_train)
X_valid = sc_x.transform(X_valid)
test_label = sc_x.transform(test_label)

X_RMSE = {} 
for i in range(100,1001, 50):
    model = RandomForestRegressor(n_estimators=i, random_state=0, criterion='mse')      
    model.fit(X_train, y_train)    
    y_predict = model.predict(X_train)
    
    mse = mean_absolute_error(y_train, y_predict)
    print(f'n_estimators={i}:','RMSE:', math.sqrt(mean_squared_error(y_train, y_predict)))
    X_RMSE[f'n_estimators={i}'] = math.sqrt(mean_squared_error(y_train, y_predict))

sorted_RMSE_RFR = sorted(X_RMSE.items(), key=lambda kv: kv[1]) 

model = RandomForestRegressor(n_estimators=500, random_state=0, criterion='mse')      
model.fit(X_train, y_train)    
y_predict = model.predict(X_train)


test = test_label
test_predict = model.predict(test_label)


# Save test predictions to file
output = pd.DataFrame({'Id': test.index,
                       'SalePrice': test_predict})
output.to_csv('submission_r.csv', index=False)