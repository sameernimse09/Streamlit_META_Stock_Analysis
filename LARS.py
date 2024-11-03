import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import io
# from google.colab import files
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
import scipy.stats as ss
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LassoLars

data = pd.read_csv('META_FeatureMart.csv', index_col = [0])

start_date = datetime(2023,1,1)
end_date = datetime(2023,10,31)

data['DATE'] = pd.to_datetime(data['DATE'])
filter = (data['DATE'] >= start_date) & (data['DATE'] <= end_date)
data = data.loc[filter]

# Copying the data into another variable
data2 = data.copy()
# Normalizing the data
for column in data.columns[1:]:
    if column != 'META_ADJ CLOSE':
        data2[column] = (data[column] - data[column].mean()) / data[column].std()

data2.columns

# Splitting the dataset into X and y
X = data2.drop(columns=['DATE', 'META_OPEN', 'META_HIGH', 'META_LOW', 'META_CLOSE', 'META_ADJ CLOSE', 'META_DAILY_RETURN']).fillna(method='bfill')  # Dropping non-numeric columns
X = sm.add_constant(X)
y = data2['META_ADJ CLOSE']

# Check for null values in each column
null_columns = np.isnan(X).any(axis=0)
null_columns

imputer = SimpleImputer(strategy='median')
X = imputer.fit_transform(X)

print('Shape of X:', X.shape)
print('Shape of y:', y.shape)

# Check for null values in each column
null_columns = np.isnan(X).any(axis=0)
null_columns


# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# ------------------------------------------------------------------------------
# Benchmark Model
# ------------------------------------------------------------------------------

benchmark_prep = sm.OLS(y, X).fit()
benchmark_prep.summary()

# Create a boolean mask with the same number of columns as X
boolean_mask_benchmark = np.abs(benchmark_prep.tvalues) >= 1.96

# Select columns from X based on the boolean mask
x_benchmark = X[:, boolean_mask_benchmark]

# Get the column names from the original DataFrame (data2)
column_names = data2.columns

# Extract selected features based on the benchmark model mask
selected_features_benchmark = [column_names[i] for i, mask in enumerate(boolean_mask_benchmark) if mask]

# Continue with the rest of your code
benchmark = sm.OLS(y, x_benchmark).fit()
print(benchmark.summary())
y_hat_benchmark1 = benchmark.predict(x_benchmark)
corr_benchmark1 = ss.pearsonr(y_hat_benchmark1, y)[0]
print('\nBenchmark: corr(Y, Y_pred) = ' + str(corr_benchmark1))
print('Hard Thresholding selected ' + str(np.sum(boolean_mask_benchmark)) + ' features in the benchmark model.')
print('Selected features in the benchmark model:', selected_features_benchmark)


from sklearn.linear_model import LassoLars
import statsmodels.api as sm
import numpy as np

# Assuming X and y are your features and target variable, and data2 is your DataFrame

# Step 1: Run LassoLars Regression and obtain coefficients
alpha = 0.5  # Adjust alpha as needed
lasso_lars_model = LassoLars(alpha=alpha, fit_intercept=False).fit(X, y)

# Create a boolean mask for selecting columns based on the coefficients
boolean_mask = np.abs(lasso_lars_model.coef_) >= 0.001

# Get the column names from the original DataFrame (data2)
column_names = data2.columns

# Select columns from X based on the boolean mask
x_selected = X[:, boolean_mask]

# Step 2: Run OLS with selected features
model_lasso_lars = sm.OLS(y, x_selected).fit()
print(model_lasso_lars.summary())
y_pred_lasso_lars = model_lasso_lars.predict(x_selected)
corr_lasso_lars = np.corrcoef(y_pred_lasso_lars, y)[0, 1]

# Print the selected features using the column_names list
selected_features = [column_names[i] for i, mask in enumerate(boolean_mask) if mask]
print('LassoLars: corr (Y, Y_pred) = ' + str(corr_lasso_lars))
print('LassoLars selected ' + str(len(selected_features)) + ' features: ', selected_features)


selected_features


from sklearn.linear_model import LassoLars
import statsmodels.api as sm
import numpy as np
import pandas as pd


# [Previous code to fit LassoLars and select features]

k = 25 

# Get the coefficients for the selected features
selected_coefficients = lasso_lars_model.coef_[boolean_mask]

# Create a DataFrame to associate coefficients with feature names
feature_importance_df = pd.DataFrame({'Feature Name': selected_features, 'Coefficient': selected_coefficients})

# Sort the features by absolute coefficient value in descending order
feature_importance_df = feature_importance_df.reindex(feature_importance_df['Coefficient'].abs().sort_values(ascending=False).index)

#-------------------------------------------------------------------------------------------------------###
#-------------------------------------------------------------------------------------------------------###
#-------------------------------------------------------------------------------------------------------###


feature_importance_df.sort_values(by='Coefficient', ascending=False, inplace=True)

# Plotting the feature importances
plt.figure(figsize=(10, 6))
plt.bar(feature_importance_df['Feature Name'], feature_importance_df['Coefficient'])
plt.xticks(rotation=90)
plt.xlabel('Features')
plt.ylabel('Absolute Coefficient Value')
plt.title('Feature Importance from Lars Regression')
plt.show()

#-------------------------------------------------------------------------------------------------------###
#-------------------------------------------------------------------------------------------------------###
#-------------------------------------------------------------------------------------------------------###

# Select the top k features
top_features = feature_importance_df['Feature Name'][:k]
print(top_features)


# Filter X_train and X_test to include only the top features
X_selected = X[:, [selected_features.index(feature) for feature in top_features]]
X_train_selected = X_train[:, [selected_features.index(feature) for feature in top_features]]
X_test_selected = X_test[:, [selected_features.index(feature) for feature in top_features]]


#####################################################
# Hyperparameter Tuning
#####################################################

# Define a range of alpha values to test
alphas = np.logspace(-6, 6, 200)
best_alpha = None
best_score = float('inf')

# Iterate over alpha values
for alpha in alphas:
    model = LassoLars(alpha=alpha, fit_intercept=False)
    scores = cross_val_score(model, X_train_selected, y_train, cv=5, scoring='neg_mean_squared_error')
    mean_score = -scores.mean()  # Convert to positive mean squared error

    if mean_score < best_score:
        best_score = mean_score
        best_alpha = alpha

print("Best alpha value:", best_alpha)

# Optionally, retrain LassoLars model using the best alpha value
lasso_lars_model_2 = LassoLars(alpha=best_alpha, fit_intercept=False).fit(X_train_selected, y_train)

y_hat = lasso_lars_model_2.predict(X_test_selected)
y_hat

#-------------------------------------------------------------------------------------------------------###
#-------------------------------------------------------------------------------------------------------###
#-------------------------------------------------------------------------------------------------------###

## Model Evaluation

rmse = mean_squared_error(y_test, y_hat, squared=False)
rmse

#-------------------------------------------------------------------------------------------------------###
#-------------------------------------------------------------------------------------------------------###
#-------------------------------------------------------------------------------------------------------###


#-------------------------------------------------------------------------------------------------------###
#-------------------------------------------------------------------------------------------------------###
#-------------------------------------------------------------------------------------------------------###


###############################################################################
# Cross Validation:
###############################################################################

# Perform cross-validation
cv_scores = cross_val_score(lasso_lars_model, X, y, cv=5, scoring='neg_mean_squared_error')

# Calculate the mean score
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
print(f'MEAN of Mean Squared Error: {mean_mse}')
print(f'STD of RMSE Squared Error: {std_rmse}')
print(f'Root Mean Squared Error on Test Set: {rmse}')

#-------------------------------------------------------------------------------------------------------###
#-------------------------------------------------------------------------------------------------------###
#-------------------------------------------------------------------------------------------------------###
