import pandas as pd
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import matplotlib.pyplot as plt

###############################################################################
# Figure configuration
import matplotlib as mpl
plt.rcParams['figure.figsize'] = (16, 6)

###############################################################################
# Load and Define dataset:
start_date = datetime(2021,1,1)
end_date = datetime(2023,10,31)
DATA_raw = pd.read_csv('META_FeatureMart.csv', index_col = [0])
DATA_raw.describe()


#-------------------------------------------------------------------------------------------------------###
#-------------------------------------------------------------------------------------------------------###
#-------------------------------------------------------------------------------------------------------###



start_date = datetime(2021,1,1)
end_date = datetime(2023,10,31)
DATA_raw['DATE'] = pd.to_datetime(DATA_raw['DATE'])
mask = (DATA_raw['DATE'] >= start_date) & (DATA_raw['DATE'] <= end_date)
DATA_raw = DATA_raw.loc[mask]
DATA_raw.reset_index(drop = True, inplace = True)

#-------------------------------------------------------------------------------------------------------###
#-------------------------------------------------------------------------------------------------------###
#-------------------------------------------------------------------------------------------------------###


# copy the data
DATA = DATA_raw.copy() 
# NORMALIZATION
for column in DATA_raw.columns[1:]:
    if column != 'META_ADJ CLOSE':
        DATA[column] = (DATA_raw[column] -
                               DATA_raw[column].mean()) / DATA_raw[column].std() 
        
DATA.columns
columns_to_drop = ['DATE', 'META_OPEN', 'META_HIGH', 'META_LOW', 'META_CLOSE', 'META_ADJ CLOSE', 'META_DAILY_RETURN']

# Drop the specified columns
X = DATA.drop(columns = columns_to_drop).fillna(method='bfill')
# X = X.fillna(method='bfill')

y = DATA['META_ADJ CLOSE']

# define the model
min_cols = 3
model = xgb.XGBRegressor(objective ='reg:linear',  
                         n_estimators=20,
                          learning_rate = 0.1,
                          max_depth = min_cols, 
                          alpha = 10, 
                          eta=0.1,
                          subsample = 0.7,
                          colsample_bytree=0.8,
                          reg_lambda=1,
                          gamma=0)
# fit the model
model.fit(X, y)

df_feature_importance = pd.DataFrame(model.feature_importances_, index=X.columns, \
                                     columns=['feature_importance']).sort_values('feature_importance', ascending=False)


#-------------------------------------------------------------------------------------------------------###
#-------------------------------------------------------------------------------------------------------###
#-------------------------------------------------------------------------------------------------------###

df_feature_importance.plot(kind='bar');

#-------------------------------------------------------------------------------------------------------###
#-------------------------------------------------------------------------------------------------------###
#-------------------------------------------------------------------------------------------------------###




df_new_feature = df_feature_importance[df_feature_importance['feature_importance'] > 0]
df_new_feature.count()

index_values = df_new_feature.index
#=========================================================================
# exhaustively search for the optimal hyperparameters
#=========================================================================
from sklearn.model_selection import GridSearchCV
# set up our search grid
param_grid = {"max_depth":    [3, 5, 10],
              "n_estimators": [10, 50, 150],
              "learning_rate": [0.15, 0.3, 0.5]}

# try out every combination of the above values
regressor = xgb.XGBRegressor(eval_metric='rmse')
search = GridSearchCV(regressor, param_grid, cv=5).fit(X, y)

print("The best hyperparameters are ",search.best_params_)
regressor = xgb.XGBRegressor(learning_rate = search.best_params_["learning_rate"],
                           n_estimators  = search.best_params_["n_estimators"],
                           max_depth     = search.best_params_["max_depth"],)

regressor.fit(X, y)

#-------------------------------------------------------------------------------------------------------###
#-------------------------------------------------------------------------------------------------------###
#-------------------------------------------------------------------------------------------------------###

from xgboost import plot_importance
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
plt.rcParams.update({'font.size': 16})

fig, ax = plt.subplots(figsize=(12,6))
plot_importance(regressor, max_num_features=8, ax=ax)
plt.show();

#-------------------------------------------------------------------------------------------------------###
#-------------------------------------------------------------------------------------------------------###
#-------------------------------------------------------------------------------------------------------###

# List of columns to keep
columns_to_keep = index_values

# Drop columns except 'A' and 'B'
X = X[columns_to_keep]

#=========================================================================
# To use early_stopping_rounds: 
# "Validation metric needs to improve at least once in every 
# early_stopping_rounds round(s) to continue training."
#=========================================================================
# first perform a test/train split 
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y, test_size = 0.2, random_state=42 ,shuffle=True)
regressor.fit(X_train, y_train, early_stopping_rounds=6, eval_set=[(X_test, y_test)], verbose=False)

#=========================================================================
# use the model to predict the prices for the test data
#=========================================================================
y_hat = regressor.predict(X_test)

## There is no overfitting given the STD of RMSE is smaller than RMSE

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_hat)

#-------------------------------------------------------------------------------------------------------###
#-------------------------------------------------------------------------------------------------------###
#-------------------------------------------------------------------------------------------------------###


# Calculate Root Mean Squared Error (RMSE)
rmse = np.sqrt(mse)
print(rmse)

#-------------------------------------------------------------------------------------------------------###
#-------------------------------------------------------------------------------------------------------###
#-------------------------------------------------------------------------------------------------------###



#-------------------------------------------------------------------------------------------------------###
#-------------------------------------------------------------------------------------------------------###
#-------------------------------------------------------------------------------------------------------###
###############################################################################
# Cross Validation:
# Perform cross-validation
cv_scores = cross_val_score(regressor, X, y, cv=5, scoring='neg_mean_squared_error')

# Let's get the mean score
mean_cv_score = np.mean(cv_scores)

# Get feature importances
feature_importances = model.feature_importances_

# Print the results
print("Mean Cross-Validation Score:", mean_cv_score)
print("Feature Importances:", feature_importances)

# Evaluate Model:
# Calculate the mean squared error and convert it back to positive
mse_scores = -cv_scores
mean_mse = np.mean(mse_scores)
std_mse = np.std(mse_scores)

rmse = np.mean(np.sqrt(np.abs(cv_scores)))
std_rmse = np.std(np.sqrt(np.abs(cv_scores)))
print('---------------------------------------','\n')
print(f'MEAN of Mean Squared Error: {mean_mse}')
print('---------------------------------------','\n')
print(f'STD of RMSE Squared Error: {std_rmse}')
print('---------------------------------------','\n')
print(f'Root Mean Squared Error on Test Set: {rmse}')
print('---------------------------------------','\n')

#-------------------------------------------------------------------------------------------------------###
#-------------------------------------------------------------------------------------------------------###
#-------------------------------------------------------------------------------------------------------###



