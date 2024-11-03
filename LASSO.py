import pandas as pd
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV

###############################################################################
# Figure configuration
import matplotlib as mpl
plt.rcParams['figure.figsize'] = (16, 6)

###############################################################################
# Load and Define dataset:
start_date = datetime(2023,1,1)
end_date = datetime(2023,10,31)
DATA_raw = pd.read_csv('META_FeatureMart.csv', index_col = [0])


#-------------------------------------------------------------------------------------------------------###
#-------------------------------------------------------------------------------------------------------###
#-------------------------------------------------------------------------------------------------------###
start_date = datetime(2023,1,1)
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
columns_to_drop = ['DATE', 'META_OPEN', 'META_HIGH', 'META_LOW', 'META_CLOSE', 'META_ADJ CLOSE', 'META_DAILY_RETURN','USRECD','USRECDM','USRECDP']

# Drop the specified columns
X = DATA.drop(columns = columns_to_drop)
X = X.fillna(method='bfill')

y = DATA['META_ADJ CLOSE']

# Data Issues:
missing_values = X.isnull().sum()
X['DCPF1M'].fillna(method='ffill', inplace=True)

#calling the model
lasso = Lasso()

# The parameter grid to search for the best alpha value
param_grid = {'alpha': np.logspace(-6, 6, 200)}
# Initialize GridSearchCV
search = GridSearchCV(lasso, param_grid, cv=5).fit(X, y)

print("The best hyperparameters are ", search.best_params_)

lasso_best = Lasso(alpha=search.best_params_['alpha'])
lasso_best.fit(X, y)

coefficients = lasso_best.coef_

# Create a DataFrame for easier visualization
feature_importance = pd.DataFrame({'Feature': X.columns, 'Coefficient': coefficients})

# Filter out features with zero coefficients
important_features = feature_importance[feature_importance['Coefficient'] != 0]

# Sort the features by the absolute value of their coefficients
important_features = important_features.reindex(important_features['Coefficient'].abs().sort_values(ascending=True).index)

#-------------------------------------------------------------------------------------------------------###
#-------------------------------------------------------------------------------------------------------###
#-------------------------------------------------------------------------------------------------------###

important_features.plot(kind='bar');

#-------------------------------------------------------------------------------------------------------###
#-------------------------------------------------------------------------------------------------------###
#-------------------------------------------------------------------------------------------------------###

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
lasso_best.fit(X_train, y_train)

y_hat = lasso_best.predict(X_test)

## There is no overfitting given the STD of RMSE is smaller than RMSE

#-------------------------------------------------------------------------------------------------------###
#-------------------------------------------------------------------------------------------------------###
#-------------------------------------------------------------------------------------------------------###
# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_hat)

# Calculate Root Mean Squared Error (RMSE)
rmse = np.sqrt(mse)
print('RMSE:',rmse)

#-------------------------------------------------------------------------------------------------------###
#-------------------------------------------------------------------------------------------------------###
#-------------------------------------------------------------------------------------------------------###

###############################################################################
# Cross Validation:
# Perform cross-validation
cv_scores = cross_val_score(lasso_best, X, y, cv=5, scoring='neg_mean_squared_error')

# Let's get the mean score
mean_cv_score = np.mean(cv_scores)

# Print the results
print("Mean Cross-Validation Score:", mean_cv_score)

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

