import pandas as pd
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

###############################################################################
# Figure configuration
import matplotlib as mpl
plt.rcParams['figure.figsize'] = (16, 6)

###############################################################################
# Load and Define dataset:
start_date = datetime(2020,1,1)
end_date = datetime(2023,10,31)
DATA_raw = pd.read_csv('META_FeatureMart.csv', index_col = [0])


start_date = datetime(2021,1,1)
end_date = datetime(2023,10,31)
DATA_raw['DATE'] = pd.to_datetime(DATA_raw['DATE'])
mask = (DATA_raw['DATE'] >= start_date) & (DATA_raw['DATE'] <= end_date)
DATA_raw = DATA_raw.loc[mask]
DATA_raw.reset_index(drop = True, inplace = True)

# copy the data
DATA = DATA_raw.copy() 
###############################################################################
# Define X and Y:
DATA.columns
columns_to_drop = ['DATE', 'META_OPEN', 'META_HIGH', 'META_LOW', 'META_CLOSE', 'META_ADJ CLOSE', 'META_DAILY_RETURN']

# Drop the specified columns
X = DATA.drop(columns=columns_to_drop)
X = X.fillna(method='bfill')

y = DATA['META_ADJ CLOSE']

# Data Issues:
missing_values = X.isnull().sum()
X['DCPF1M'].fillna(method='ffill', inplace=True)
# NORMALIZATION
for column in DATA_raw.columns[1:]:
    if column != 'META_ADJ CLOSE':
        DATA[column] = (DATA_raw[column] -
                               DATA_raw[column].mean()) / DATA_raw[column].std()  
        


# define the model
model = RandomForestRegressor(bootstrap=True, criterion='squared_error', max_depth=None,\
           max_features=36, max_leaf_nodes=None,\
           min_impurity_decrease=0.0,\
           min_samples_leaf=1, min_samples_split=2,\
           min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=None,\
           oob_score=False, random_state=None, verbose=0, warm_start=False)

# fit the model
model.fit(X, y)

###############################################################################
# Feature Importance:
# 1. average feature importance
df_feature_importance = pd.DataFrame(model.feature_importances_, index=X.columns, \
                                     columns=['Feature Importance']).sort_values('Feature Importance', ascending=False)
print(df_feature_importance)
print(df_feature_importance.count())


# 2. all feature importance for each tree
# (1) bar chart

#-------------------------------------------------------------------------------------------------------###
#-------------------------------------------------------------------------------------------------------###
#-------------------------------------------------------------------------------------------------------###

df_feature_importance.plot(kind='bar');

#-------------------------------------------------------------------------------------------------------###
#-------------------------------------------------------------------------------------------------------###
#-------------------------------------------------------------------------------------------------------###

###############################################################################
# Feature Selection:
# Assuming feature_importance is your array of feature importance scores
feature_threshold = 0  # Adjust the threshold based on your preference
df_new_feature = df_feature_importance[df_feature_importance['Feature Importance'] > feature_threshold]

# Get the selected features
selected_features = df_new_feature.index
print(f'Selected Features: {selected_features.to_list()}')
X_selected = X[selected_features]

#=========================================================================
# exhaustively search for the optimal hyperparameters
#=========================================================================
from sklearn.model_selection import GridSearchCV
# set up our search grid
# Define the parameter grid to search
param_grid = {
    'n_estimators': [10, 50, 100],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

# Initialize the Random Forest Regressor
regressor = RandomForestRegressor(random_state=42)
# Initialize GridSearchCV
search = GridSearchCV(regressor, param_grid, cv=5).fit(X, y)

print("The best hyperparameters are ",search.best_params_)
regressor = RandomForestRegressor(n_estimators  = search.best_params_["n_estimators"],
                           max_depth     = search.best_params_["max_depth"],
                           min_samples_split = search.best_params_["min_samples_split"],
                           min_samples_leaf = search.best_params_["min_samples_leaf"],)

regressor.fit(X, y)

#=========================================================================
# To use early_stopping_rounds: 
# "Validation metric needs to improve at least once in every 
# early_stopping_rounds round(s) to continue training."
#=========================================================================
# first perform a test/train split 
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X_selected,y, test_size = 0.2)
regressor.fit(X_train, y_train)

#=========================================================================
# use the model to predict the prices for the test data
#=========================================================================
y_hat = regressor.predict(X_test)


#-------------------------------------------------------------------------------------------------------###
#-------------------------------------------------------------------------------------------------------###
#-------------------------------------------------------------------------------------------------------###

mse = mean_squared_error(y_test, y_hat)

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
