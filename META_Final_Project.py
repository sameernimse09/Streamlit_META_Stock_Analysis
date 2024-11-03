import streamlit as st
import scipy.stats as ss
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from datetime import datetime, timedelta
import yfinance as yf
from streamlit_extras.let_it_rain import rain
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LassoLars
import xgboost as xgb
import statsmodels.api as sm

# Create a Streamlit app
st.title("META Stock Model Selection")
# streamlit write italics
st.markdown("_META Team: Yash Gopalji Pankhania, Jared Videlefsky, Venkata Sai Charan Patnaik Amiti, Tanmay Dilip Zope, Sameer Sanjay Nimse, Raj Yuvraj Sarode_")
# st.divider()

st.toast("Warming up META...")
tab1, tab2, tab3, tab4, tab5, tab6, tab8, tab9, tab10 = st.tabs(["Home", 'Kalman Filter', 'Random Forest', 'Ridge Regression', 'LASSO', 'LARS', 'XGBoost', 'Trading Strategy', 'Profit Loss'])
with tab1:
    st.title("Home")

    st.subheader('Introduction')
    intro_text = """
### Meta: Building the Social Metaverse

Meta is a visionary company dedicated to constructing technologies that foster human connections, facilitate community discovery, and empower businesses to flourish. Their vision for the metaverse encompasses a wide array of cutting-edge technologies and products, spanning virtual reality (VR), augmented reality (AR), and the creation of digital environments that redefine how people interact, work, play, and socialize.

### Revenue Dominance

Meta's primary revenue stream is derived from advertising, with advertising being its most lucrative business segment by a significant margin. In the year 2020, a staggering 97.9 percent of Meta's global revenue was generated from advertising.

### Business Outlook for Meta

1. **Strong Q2 2023 Earnings**: Formerly known as Facebook, Meta reported robust earnings for the second quarter of 2023, breaking a streak of year-over-year declines. This remarkable performance was attributed to user growth across all regions, boasting daily active users of 2.064 billion in the U.S. and an astounding 3.07 billion users engaging with at least one Meta app on a daily basis. The company anticipates Q3 revenue to range from $32 billion to $34.5 billion, driven by enhanced monetization of Reels videos.

2. **Ongoing Challenges**: Meta faces a set of challenges, including regulatory concerns and intense competition. Despite its favorable earnings, Meta remains wary of the repercussions of Apple's privacy changes and persists in dealing with legal and regulatory hurdles. Additionally, the company is strategically focusing on artificial intelligence (AI), business messaging, and the metaverse as avenues for future growth.    
"""
    st.markdown(intro_text)

    st.header('META Competitors')

    comp_text = """

**Amazon**:  
Amazon competes with Meta in the stock market as they both vie for digital advertising revenue and online retail dominance. While Meta focuses on social media, Amazon's e-commerce and cloud computing services intersect with Meta's interests, leading to competition for market share and investor attention.

**Snapchat**:  
Snapchat competes with Meta in the stock market by offering a rival social media platform. Both companies aim to capture user engagement and advertising revenue, making them competitors in the dynamic social media and digital advertising landscape.

**Google**:  
Google competes with Meta in digital advertising, cloud storage, and messaging services, vying for advertiser budgets and user attention, which indirectly affects their stock performance. Additionally, both companies invest in VR and AR technologies, with Google's Google Cardboard and Meta's Oculus VR, potentially influencing future growth areas.

**Apple**:  
Apple and Meta are tech giants, each with diverse product portfolios. Apple focuses on hardware like iPhones, Macs, and services such as Apple Music. Meta, primarily a social media company, owns platforms like Facebook, Instagram, and WhatsApp. Both compete in augmented reality and virtual reality spaces. In the stock market, their performances are influenced by product innovations, user engagement, regulatory challenges, and global market dynamics, impacting investor sentiment and stock prices.

"""
    st.markdown(comp_text)

    st.subheader('META Competitors Stock Price History')
    st.image('https://github.com/Draconian10/Stocks_Time_Series_Causal_Inference/assets/23314479/745186df-ee53-4de6-83be-96e56a84a2d0')
    
    ###############################################################################################
    # Section 1: Exploratory Data Analysis (EDA)
    st.subheader("Exploratory Data Analysis (EDA)")

    # Graph Meta Stock Price
    # Define the ticker symbol and time period
    ticker_symbol = "META"
    period = "90d"

    # Fetch data from Yahoo Finance
    stock_data = yf.Ticker(ticker_symbol)
    df = stock_data.history(period=period).reset_index()

    # Create a Plotly graph
    fig = go.Figure(data=[go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name='META')])
    fig.update_layout(title='Simulated META Stock Price for the Past 90 Days',
                    xaxis_title='Date',
                    yaxis_title='Price (USD)',
                    template="plotly_dark")
    # Show the figure using Streamlit
    st.plotly_chart(fig)


    start_date = datetime(2020,1,1)
    # end_date = datetime(2023,8,31)
    end_date = datetime.now()

    stock_tickers = ['META', 'SNAP', 'GOOG', 'AMZN', 'AAPL']

    # KDE Equation
    def plot_kde(ticker, stock_adj_close):
        # Nonparametric Kernel Density Plot
        kde = sns.kdeplot(data=stock_adj_close.values, linewidth=4)
        # This will add title to plot
        kde.set_title( f'KDE for {stock_tickers}')
        plt.legend(labels=stock_tickers)
    #     plt.axvline(stock_adj_close.values[-1], color ='RED')
        
        # This will add label to X-axis
        kde.set_xlabel( "Adj Close")
        # This will add label to Y-axis
        kde.set_ylabel( "Density")


    st.subheader("Competitors Statistics:")

    stock_statistics = []

    for ticker in stock_tickers:
        # Get data from Yahoo Finance
        stock_df = yf.download(ticker, start_date, end_date)
        stock_adj_close = stock_df['Adj Close']
        
        # Portfolio or Competition
        port_or_comp = 'Portfolio' if ticker == 'META' else 'Competitor'
        
        # Mean
        stock_mean = stock_adj_close.mean()

        # Variance
        stock_variance = stock_adj_close.var()

        # Skewness
        stock_skewness = stock_adj_close.skew()

        # Kurtosis
        stock_kurtosis = stock_adj_close.kurtosis()
        
        stock_statistics.append([ticker, port_or_comp, stock_mean, stock_variance, stock_skewness, stock_kurtosis])
        
        # KDE Plot
        plot_kde(ticker, stock_adj_close)
        
    statistics_df = pd.DataFrame(stock_statistics, columns=['Stock_Ticker', 'Portfolio or Competitor', 'Mean', 'Variance', 'Skewness', 'Kurtosis'])
    st.write(statistics_df)
    st.subheader('KDE Plot')
    st.pyplot(fig=plt)




    ###############################################################################################
    # Moving Averages
    st.subheader("Moving Averages for META")

    df = yf.download(ticker, start_date, end_date).reset_index()

    print(df.head())

    # Sort DataFrame by Date
    df.sort_values('Date', inplace=True)

    # Function to calculate SMA and EMA
    def calculate_ema_sma(df, span_short=50, span_long=200):
        df['SMA_50'] = df['Adj Close'].rolling(window=span_short).mean()
        df['SMA_200'] = df['Adj Close'].rolling(window=span_long).mean()
        df['EMA_50'] = df['Adj Close'].ewm(span=span_short, adjust=False).mean()
        df['EMA_200'] = df['Adj Close'].ewm(span=span_long, adjust=False).mean()
        return df

    df = calculate_ema_sma(df)
    # Filter the DataFrame for the last 365 days
    df = df[(df['Date'] >= end_date - timedelta(365)) & (df['Date'] <= end_date)]


    # Plotly figure
    fig = go.Figure()

    # Candlestick plot
    fig.add_trace(go.Candlestick(x=df['Date'],
                    open=df['Adj Close'],
                    high=df['Adj Close'],
                    low=df['Adj Close'],
                    close=df['Adj Close'],
                    name='Stock Price'))

    # Scatter plot for 50 and 200 SMA
    fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA_50'], mode='lines', name='50 SMA', line=dict(color='orange')))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA_200'], mode='lines', name='200 SMA', line=dict(color='red')))

    # Scatter plot for 50 and 200 EMA
    fig.add_trace(go.Scatter(x=df['Date'], y=df['EMA_50'], mode='lines', name='50 EMA', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['EMA_200'], mode='lines', name='200 EMA', line=dict(color='green')))

    # Update layout for better visibility
    fig.update_layout(title='Stock Price with SMA and EMA',
                    xaxis_title='Date',
                    yaxis_title='Stock Price')

    # Set y-axis to auto range
    fig.update_yaxes(autorange=True)

    # Show the figure using Streamlit
    st.plotly_chart(fig)





    ###############################################################################################
    # Section 2: Model Selection
    st.subheader("Feature Data Extraction")

    # Feature Dictionary
    st.write('**Feature Dictionary:**')
    # if st.button('Show Feature Dictionary'):
    feature_dict = {
    'T10Y3M': 'Term Premium 10yr-3mon',
    'OBMMIJUMBO30YF': '30-Year Mortgage Jumbo Loan',
    'DEXUSEU': 'Spot Exchange Rate to Euro (EUR)',
    'DEXJPUS': 'Spot Exchange Rate to Japanese Yen (JPY)',
    'DEXUSUK': 'Spot Exchange Rate to British Pound (GBP)',
    'CBBTCUSD': 'Cryptocurrency - Bitcoin to US Dollar',
    'CBETHUSD': 'Cryptocurrency - Ethereum to US Dollar',
    'T10YIE': 'Breakeven Inflation - 10-Year',
    'DCOILBRENTEU': 'Brent Crude Oil Price in Euros',
    'VIXCLS': 'Implied Volatility Index',
    'DAAA': 'Corporate Bond Yield - AAA',
    'DBAA': 'Corporate Bond Yield - BAA',
    'NIKKEI225': 'Nikkei 225 Index',
    'AMERIBOR': 'American Interbank Offered Rate',
    'T5YIE': 'Breakeven Inflation - 5-Year',
    'BAMLH0A0HYM2': 'High Yield Corporate Bond Index',
    'BAMLH0A0HYM2EY': 'High Yield Corporate Bond Index - Earnings Yield',
    'DGS10': '10-Year Treasury Constant Maturity Rate',
    'DGS1': '1-Year Treasury Constant Maturity Rate',
    'RIFSPPFAAD90NB': 'Effective Federal Funds Rate - 90-Day Average',
    'DCPN3M': 'Commercial Paper - 3-Month',
    'DCPF1M': 'Commercial Paper - Financial - 1-Month',
    'DCOILWTICO': 'West Texas Intermediate (WTI) Crude Oil Price',
    'DHHNGSP': 'Henry Hub Natural Gas Spot Price',
    'USRECD': 'US Business Cycle Expansion/Contraction Indicator',
    'USRECDM': 'US Business Cycle Expansion/Contraction Indicator (Monthly)',
    'USRECDP': 'US Business Cycle Expansion/Contraction Indicator (Probability)',
    'Mkt-RF': 'Market Risk Premium',
    'SMB': 'Small Minus Big (SMB)',
    'HML': 'High Minus Low (HML)',
    'RMW': 'Robust Minus Weak (RMW)',
    'CMA': 'Conservative Minus Aggressive (CMA)',
    'RF': 'Risk-Free Rate',
    'META_VOLUME': 'Daily META Shares Volume',
    'META_RSI': 'Relative Strength Index'
    }
    # Convert the feature_dict to a DataFrame order by feature
    feature_df = pd.DataFrame(list(feature_dict.items()), columns=['Feature', 'Value']).sort_values(by='Feature')
    st.dataframe(feature_df, hide_index=True)


    # Display model selection results
    st.subheader("Model Selection Results:")
    # Reading data from CSV
    df = pd.read_csv('Model_Results.csv')
    # print(df.head())
    # df.columns

    # Sorting by 'model' and then by 'TOTAL_PNL' in descending order
    df_sorted = df.sort_values(by=['RMSE'], ascending=[True])

    st.dataframe(df_sorted, hide_index=True, width = 2000)
    model_text = """
- XGBoost and Random Forest are both ensemble learning techniques known for their superior predictive accuracy, particularly in complex and non-linear datasets compared to individual models like regression models.

- While decision trees can be prone to overfitting compared to regression models, we took precautions to prevent overfitting in our analysis.

- The best root mean square error (RMSE) results we obtained, considering different training data sizes, are as follows:

   1. LARS for smaller training periods.
   2. XGBoost for larger training periods, offering consistent accuracy and performance.
"""
    st.markdown(model_text)

    # Raining Emojis    
    rain(
    emoji= "💰", # Assorted emojis
    font_size=25,
    falling_speed=3,
    animation_length=3  # Emojis will rain for 10 seconds
    )

########################################################################################################################
# Ridge Regression
with tab4:
    st.title("Ridge Regression")
    text = """
- Ridge Regression is effective in managing multicollinearity among predictors in a regression model by applying a penalty to the coefficients.
- It shrinks the regression coefficients towards zero (but not to zero), which is particularly useful in high-dimensional datasets.
- It may not perform as well in scenarios with complex non-linear relationships, similar to other linear regression techniques.
"""
    st.markdown(text)
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from datetime import datetime
    import io
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_squared_error
    import statsmodels.api as sm
    import scipy.stats as ss
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import cross_val_score

    data = pd.read_csv('META_FeatureMart.csv')
    data.drop(columns = ['Unnamed: 0'], inplace=True)
    # data.head()
    st.write('**Data Summary:**', data.describe())

    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 10, 31)

    st.write('**Date Range**')
    st.write('Start Date: ', start_date)
    st.write('End Date: ', end_date)

    data['DATE'] = pd.to_datetime(data['DATE'])
    filter = (data['DATE'] >= start_date) & (data['DATE'] <= end_date)
    data = data.loc[filter]
    # data.head()

    # Copying the data into another variable
    data2 = data.copy()

    # Normalizing the data
    for column in data.columns[1:]:
        if column != 'META_ADJ CLOSE':
            data2[column] = (data[column] - data[column].mean()) / data[column].std()

    #data2.columns

    # Splitting the dataset into X and y
    X = data2.drop(columns=['DATE', 'META_OPEN', 'META_HIGH', 'META_LOW', 'META_CLOSE', 'META_ADJ CLOSE',
                            'META_DAILY_RETURN']).fillna(method='bfill')  # Dropping non-numeric columns
    X = sm.add_constant(X)
    y = data2['META_ADJ CLOSE']

    # Check for null values in each column
    null_columns = np.isnan(X).any(axis=0)
    #null_columns

    imputer = SimpleImputer(strategy='median')
    X = imputer.fit_transform(X)

    print('Shape of X:', X.shape)
    print('Shape of y:', y.shape)

    # Check for null values in each column
    null_columns = np.isnan(X).any(axis=0)
    #null_columns

    # Splitting the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # ------------------------------------------------------------------------------
    # Benchmark Model
    # ------------------------------------------------------------------------------

    benchmark_prep = sm.OLS(y, X).fit()
    # benchmark_prep.summary()

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

    # ------------------------------------------------------------------------------
    # Factor Selection using Ridge Regression
    # ------------------------------------------------------------------------------
    # Motivation: the model sets the objective function to be the
    # sum of squared residuals from OLS + penalty function
    # that penalizes squared values of beta. A minimization process squeezes
    # the small beta values close to 0.
    # a represents the obj function's sensitivity to the penalty term
    # Steo 1. run Ridge Regression and obtain coefficients
    # Step 2. remove features with coefficients close to 0 and run OLS
    a = 0.5
    model2_prep = Ridge(alpha=a, fit_intercept=False).fit(X, y)

    # Create a boolean mask for selecting columns based on the coefficients
    boolean_mask = np.abs(model2_prep.coef_) >= 0.001

    # Get the column names from the original DataFrame (data2)
    column_names = data2.columns

    # Select columns from X based on the boolean mask
    x = X[:, boolean_mask]

    # Continue with the rest of your code
    model2 = sm.OLS(y, x).fit()
    print(model2.summary())
    y_pred_model2 = model2.predict(x)
    corr_model2 = ss.pearsonr(y_pred_model2, y)[0]

    # Print the selected features using the column_names list
    selected_features = [column_names[i] for i, mask in enumerate(boolean_mask) if mask]
    print('model 2 Ridge Regression: corr (Y, Y_pred) = ' + str(corr_model2))
    print('Ridge Regression selected ' + str(len(selected_features)) + ' features: ', selected_features)

    #selected_features

    # Define the number of top features to keep
    k = 25  # Change this to the desired number of top features

    # Get the coefficients of the Ridge model (feature importance)
    coefficients = model2_prep.coef_

    # Create a DataFrame to associate coefficients with feature names
    feature_importance_df = pd.DataFrame({'Feature Name': selected_features, 'Coefficient': coefficients})

    # Sort the features by absolute coefficient value in descending order
    feature_importance_df = feature_importance_df.reindex(
        feature_importance_df['Coefficient'].sort_values(ascending=False).index)
    print(feature_importance_df)

    feature_importance_df.sort_values(by='Coefficient', ascending=False, inplace=True)

    # Plotting the feature importances
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(feature_importance_df['Feature Name'], feature_importance_df['Coefficient'])
    ax.set_xticklabels(feature_importance_df['Feature Name'], rotation=90)
    ax.set_xlabel('Features')
    ax.set_ylabel('Absolute Coefficient Value')
    ax.set_title('Feature Importance from Ridge Regression')
    st.pyplot(fig)

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

    # Replace Ridge with RidgeCV
    model2_prep = RidgeCV(alphas=alphas, fit_intercept=False, store_cv_values=True).fit(X_train_selected, y_train)

    # Print the best alpha value
    print("Best alpha value:", model2_prep.alpha_)

    # Optionally, retrain Ridge model using the best alpha value
    model2 = Ridge(alpha=model2_prep.alpha_, fit_intercept=False).fit(X_train_selected, y_train)

    y_hat = model2.predict(X_test_selected)
    # y_hat

    ## Model Evaluation

    rmse = mean_squared_error(y_test, y_hat, squared=False)
    st.subheader('Accuracy Score')
    st.write('**Root Mean Squared Error:**', rmse)

    ###############################################################################
    # Cross Validation:
    ###############################################################################

    # Perform cross-validation
    cv_scores = cross_val_score(model2, X, y, cv=5, scoring='neg_mean_squared_error')

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

    st.subheader('Validating Over-Fitting')
    st.write('**Mean of Mean Squared Error:**', mean_mse)
    st.write('**Standard Deviation of Root Mean Squared Error:**', std_rmse)
    st.write('**Root Mean Squared Error on Test Set:**', rmse)

    st.subheader('Code')
    ridge_text = """
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import io
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
import scipy.stats as ss
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_val_score

data = pd.read_csv('META_FeatureMart.csv')
#data.head()

start_date = datetime(2023,1,1)
end_date = datetime(2023,10,31)

data['DATE'] = pd.to_datetime(data['DATE'])
filter = (data['DATE'] >= start_date) & (data['DATE'] <= end_date)
data = data.loc[filter]
#data.head()

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
#benchmark_prep.summary()

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

# ------------------------------------------------------------------------------
# Factor Selection using Ridge Regression
# ------------------------------------------------------------------------------
# Motivation: the model sets the objective function to be the
# sum of squared residuals from OLS + penalty function
# that penalizes squared values of beta. A minimization process squeezes
# the small beta values close to 0.
# a represents the obj function's sensitivity to the penalty term
# Steo 1. run Ridge Regression and obtain coefficients
# Step 2. remove features with coefficients close to 0 and run OLS
a = 0.5
model2_prep = Ridge(alpha=a, fit_intercept=False).fit(X, y)

# Create a boolean mask for selecting columns based on the coefficients
boolean_mask = np.abs(model2_prep.coef_) >= 0.001

# Get the column names from the original DataFrame (data2)
column_names = data2.columns

# Select columns from X based on the boolean mask
x = X[:, boolean_mask]

# Continue with the rest of your code
model2 = sm.OLS(y, x).fit()
print(model2.summary())
y_pred_model2 = model2.predict(x)
corr_model2 = ss.pearsonr(y_pred_model2, y)[0]

# Print the selected features using the column_names list
selected_features = [column_names[i] for i, mask in enumerate(boolean_mask) if mask]
print('model 2 Ridge Regression: corr (Y, Y_pred) = ' + str(corr_model2))
print('Ridge Regression selected ' + str(len(selected_features)) + ' features: ', selected_features)

selected_features

# Define the number of top features to keep
k = 25  # Change this to the desired number of top features

# Get the coefficients of the Ridge model (feature importance)
coefficients = model2_prep.coef_

# Create a DataFrame to associate coefficients with feature names
feature_importance_df = pd.DataFrame({'Feature Name': selected_features, 'Coefficient': coefficients})

# Sort the features by absolute coefficient value in descending order
feature_importance_df = feature_importance_df.reindex(feature_importance_df['Coefficient'].sort_values(ascending=False).index)
print(feature_importance_df)

feature_importance_df.sort_values(by='Coefficient', ascending=False, inplace=True)

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

# Replace Ridge with RidgeCV
model2_prep = RidgeCV(alphas=alphas, fit_intercept=False, store_cv_values=True).fit(X_train_selected, y_train)

# Print the best alpha value
print("Best alpha value:", model2_prep.alpha_)

# Optionally, retrain Ridge model using the best alpha value
model2 = Ridge(alpha=model2_prep.alpha_, fit_intercept=False).fit(X_train_selected, y_train)

y_hat = model2.predict(X_test_selected)
#y_hat

## Model Evaluation

rmse = mean_squared_error(y_test, y_hat, squared=False)

###############################################################################
# Cross Validation:
###############################################################################

# Perform cross-validation
cv_scores = cross_val_score(model2, X, y, cv=5, scoring='neg_mean_squared_error')

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
"""
    st.code(ridge_text, language='python')
    

########################################################################################################################
with tab5:
    st.title("LASSO")
    text = """
    - Similar to LARS but includes a penalty term that encourages sparsity in the coefficients.
    - Useful for feature selection and regularization.
    - Still a linear model, so it may not capture complex non-linear patterns well.
    """
    st.markdown(text)
    # Load and Define dataset:
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 10, 31)
    DATA_raw = pd.read_csv('META_FeatureMart.csv', index_col=[0])
    st.write('**Data Summary:**', DATA_raw.describe())

    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 10, 31)
    DATA_raw['DATE'] = pd.to_datetime(DATA_raw['DATE'])
    mask = (DATA_raw['DATE'] >= start_date) & (DATA_raw['DATE'] <= end_date)
    DATA_raw = DATA_raw.loc[mask]
    DATA_raw.reset_index(drop=True, inplace=True)

    st.write('**Date Range**')
    st.write('Start Date: ', start_date)
    st.write('End Date: ', end_date)

    # copy the data
    DATA = DATA_raw.copy()
    # NORMALIZATION
    for column in DATA_raw.columns[1:]:
        if column != 'META_ADJ CLOSE':
            DATA[column] = (DATA_raw[column] -
                            DATA_raw[column].mean()) / DATA_raw[column].std()

    # DATA.columns
    columns_to_drop = ['DATE', 'META_OPEN', 'META_HIGH', 'META_LOW', 'META_CLOSE', 'META_ADJ CLOSE',
                       'META_DAILY_RETURN', 'USRECD', 'USRECDM', 'USRECDP']

    # Drop the specified columns
    X = DATA.drop(columns=columns_to_drop)
    X = X.fillna(method='bfill')

    y = DATA['META_ADJ CLOSE']

    # Data Issues:
    missing_values = X.isnull().sum()
    X['DCPF1M'].fillna(method='ffill', inplace=True)

    # calling the model
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
    important_features = important_features.reindex(
        important_features['Coefficient'].abs().sort_values(ascending=True).index)

    important_features.plot(kind='bar');
    # Plot feature importance using Plotly
    fig = px.bar(important_features, x='Feature', y='Coefficient',
                 labels={'Feature': 'Feature', 'Coefficient': 'Coefficient'}, title='Feature Importance')

    # Display the Plotly chart in Streamlit
    st.plotly_chart(fig)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    lasso_best.fit(X_train, y_train)

    y_hat = lasso_best.predict(X_test)

    ## There is no overfitting given the STD of RMSE is smaller than RMSE

    # Calculate Mean Squared Error (MSE)
    mse = mean_squared_error(y_test, y_hat)

    # Calculate Root Mean Squared Error (RMSE)
    rmse = np.sqrt(mse)
    print('RMSE:', rmse)

    st.subheader('Accuracy Score')
    st.write('**Root Mean Squared Error:**', rmse)

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
    print('---------------------------------------', '\n')
    print(f'MEAN of Mean Squared Error: {mean_mse}')
    print('---------------------------------------', '\n')
    print(f'STD of RMSE Squared Error: {std_rmse}')
    print('---------------------------------------', '\n')
    print(f'Root Mean Squared Error on Test Set: {rmse}')
    print('---------------------------------------', '\n')

    st.subheader('Validating Over-Fitting')
    st.write('**Mean of Mean Squared Error:**', mean_mse)
    st.write('**Standard Deviation of Root Mean Squared Error:**', std_rmse)
    st.write('**Root Mean Squared Error on Test Set:**', rmse)

    st.subheader('Code')
    lasso_text = """
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

# Figure configuration
import matplotlib as mpl
plt.rcParams['figure.figsize'] = (16, 6)

# Load and Define dataset:
start_date = datetime(2023,1,1)
end_date = datetime(2023,10,31)
DATA_raw = pd.read_csv('META_FeatureMart.csv', index_col = [0])

start_date = datetime(2023,1,1)
end_date = datetime(2023,10,31)
DATA_raw['DATE'] = pd.to_datetime(DATA_raw['DATE'])
mask = (DATA_raw['DATE'] >= start_date) & (DATA_raw['DATE'] <= end_date)
DATA_raw = DATA_raw.loc[mask]
DATA_raw.reset_index(drop = True, inplace = True)

# copy the data
DATA = DATA_raw.copy()
# NORMALIZATION
for column in DATA_raw.columns[1:]:
    if column != 'META_ADJ CLOSE':
        DATA[column] = (DATA_raw[column] -
                               DATA_raw[column].mean()) / DATA_raw[column].std()
        
#DATA.columns
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

important_features.plot(kind='bar');

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
lasso_best.fit(X_train, y_train)

y_hat = lasso_best.predict(X_test)

## There is no overfitting given the STD of RMSE is smaller than RMSE

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_hat)

# Calculate Root Mean Squared Error (RMSE)
rmse = np.sqrt(mse)
print('RMSE:',rmse)

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

"""
    st.code(lasso_text, language='python')


######################################################################################
with tab6:
    st.title("LARS")
    text = """
    - Similar to LARS but includes a penalty term that encourages sparsity in the coefficients.
    - Useful for feature selection and regularization.
    - Still a linear model, so it may not capture complex non-linear patterns well."""

    # Displaying the text in Markdown format
    st.markdown(text)
    data = pd.read_csv('META_FeatureMart.csv', index_col=[0])
    st.write('**Data Summary:**', data.describe())

    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 10, 31)
    st.write('**Date Range**')
    st.write('Start Date: ', start_date)
    st.write('End Date: ', end_date)

    data['DATE'] = pd.to_datetime(data['DATE'])
    filter = (data['DATE'] >= start_date) & (data['DATE'] <= end_date)
    data = data.loc[filter]

    # Copying the data into another variable
    data2 = data.copy()
    # Normalizing the data
    for column in data.columns[1:]:
        if column != 'META_ADJ CLOSE':
            data2[column] = (data[column] - data[column].mean()) / data[column].std()

    # data2.columns

    # Splitting the dataset into X and y
    X = data2.drop(columns=['DATE', 'META_OPEN', 'META_HIGH', 'META_LOW', 'META_CLOSE', 'META_ADJ CLOSE',
                            'META_DAILY_RETURN']).fillna(method='bfill')  # Dropping non-numeric columns
    X = sm.add_constant(X)
    y = data2['META_ADJ CLOSE']

    # Check for null values in each column
    null_columns = np.isnan(X).any(axis=0)
    # null_columns

    imputer = SimpleImputer(strategy='median')
    X = imputer.fit_transform(X)

    #print('Shape of X:', X.shape)
    #print('Shape of y:', y.shape)

    # Check for null values in each column
    null_columns = np.isnan(X).any(axis=0)
    # null_columns

    # Splitting the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # ------------------------------------------------------------------------------
    # Benchmark Model
    # ------------------------------------------------------------------------------

    benchmark_prep = sm.OLS(y, X).fit()
    #benchmark_prep.summary()

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

    from sklearn.linear_model import LassoLars

    k = 25

    # Get the coefficients for the selected features
    selected_coefficients = lasso_lars_model.coef_[boolean_mask]

    # Create a DataFrame to associate coefficients with feature names
    feature_importance_df = pd.DataFrame({'Feature Name': selected_features, 'Coefficient': selected_coefficients})

    # Sort the features by absolute coefficient value in descending order
    feature_importance_df = feature_importance_df.reindex(
        feature_importance_df['Coefficient'].abs().sort_values(ascending=False).index)

    feature_importance_df.sort_values(by='Coefficient', ascending=False, inplace=True)

    # Plotting the feature importances
    plt.figure(figsize=(10, 6))
    plt.bar(feature_importance_df['Feature Name'], feature_importance_df['Coefficient'])
    plt.xticks(rotation=90)
    plt.xlabel('Features')
    plt.ylabel('Absolute Coefficient Value')
    plt.title('Feature Importance from Lars Regression')
    plt.show()

    # Plotting the feature importances with Plotly
    fig = px.bar(feature_importance_df, x='Feature Name', y='Coefficient',
                 title='Feature Importance from Lars Regression')
    fig.update_layout(xaxis_title="Features", yaxis_title="Absolute Coefficient Value",
                      xaxis={'categoryorder': 'total descending'})
    st.plotly_chart(fig)

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
    #y_hat

    ## Model Evaluation

    rmse = mean_squared_error(y_test, y_hat, squared=False)
    st.subheader('Accuracy Score')
    st.write('**Root Mean Squared Error:**', rmse)

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
    st.subheader('Validating Over-Fitting')
    st.write('**Mean of Mean Squared Error:**', mean_mse)
    st.write('**Standard Deviation of Root Mean Squared Error:**', std_rmse)
    st.write('**Root Mean Squared Error on Test Set:**', rmse)
    st.subheader('Code')
    lars_text = """
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

#data2.columns

# Splitting the dataset into X and y
X = data2.drop(columns=['DATE', 'META_OPEN', 'META_HIGH', 'META_LOW', 'META_CLOSE', 'META_ADJ CLOSE', 'META_DAILY_RETURN']).fillna(method='bfill')  # Dropping non-numeric columns
X = sm.add_constant(X)
y = data2['META_ADJ CLOSE']

# Check for null values in each column
null_columns = np.isnan(X).any(axis=0)
#null_columns

imputer = SimpleImputer(strategy='median')
X = imputer.fit_transform(X)

print('Shape of X:', X.shape)
print('Shape of y:', y.shape)

# Check for null values in each column
null_columns = np.isnan(X).any(axis=0)
#null_columns


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

from sklearn.linear_model import LassoLars

k = 25 

# Get the coefficients for the selected features
selected_coefficients = lasso_lars_model.coef_[boolean_mask]

# Create a DataFrame to associate coefficients with feature names
feature_importance_df = pd.DataFrame({'Feature Name': selected_features, 'Coefficient': selected_coefficients})

# Sort the features by absolute coefficient value in descending order
feature_importance_df = feature_importance_df.reindex(feature_importance_df['Coefficient'].abs().sort_values(ascending=False).index)

feature_importance_df.sort_values(by='Coefficient', ascending=False, inplace=True)

# Plotting the feature importances
plt.figure(figsize=(10, 6))
plt.bar(feature_importance_df['Feature Name'], feature_importance_df['Coefficient'])
plt.xticks(rotation=90)
plt.xlabel('Features')
plt.ylabel('Absolute Coefficient Value')
plt.title('Feature Importance from Lars Regression')
plt.show()

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

## Model Evaluation

rmse = mean_squared_error(y_test, y_hat, squared=False)

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
"""
    st.code(lars_text, language='python')


########################################################################################################################
#RANDOM FOREST
with tab3:
    st.title("Random Forest")
    text = """
    - Ensemble learning method based on constructing a multitude of decision trees.
    - Robust to overfitting, capable of handling non-linear relationships, and provides feature importance measures.
    - Generally performs well across a variety of datasets.
    """
    st.markdown(text)

    # Load and Define dataset:
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2023, 10, 31)
    DATA_raw = pd.read_csv('META_FeatureMart.csv', index_col=[0])
    st.write('**Data Summary:**', DATA_raw.describe())
    st.write('**Date Range**')
    st.write('Start Date: ', start_date)
    st.write('End Date: ', end_date)
    # copy the data
    DATA = DATA_raw.copy()
    # NORMALIZATION
    for column in DATA_raw.columns[1:]:
        if column != 'META_ADJ CLOSE':
            DATA[column] = (DATA_raw[column] -
                            DATA_raw[column].mean()) / DATA_raw[column].std()

    ###############################################################################
    # Define X and Y:
    # DATA.columns
    columns_to_drop = ['DATE', 'META_OPEN', 'META_HIGH', 'META_LOW', 'META_CLOSE', 'META_ADJ CLOSE',
                       'META_DAILY_RETURN']

    # Drop the specified columns
    X = DATA.drop(columns=columns_to_drop)
    X = X.fillna(method='bfill')

    y = DATA['META_ADJ CLOSE']

    # Data Issues:
    missing_values = X.isnull().sum()
    X['DCPF1M'].fillna(method='ffill', inplace=True)


    # define the model
    model = RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,\
            max_features=36, max_leaf_nodes=None,\
            min_impurity_decrease=0.0,\
            min_samples_leaf=1, min_samples_split=2,\
            min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=None,\
            oob_score=False, random_state=None, verbose=0, warm_start=False)

    # fit the model
    model.fit(X, y)

    # Feature Importance:
    # 1. average feature importance
    df_feature_importance = pd.DataFrame(model.feature_importances_, index=X.columns, \
                                        columns=['Feature Importance']).sort_values('Feature Importance', ascending=False)
    print(df_feature_importance)
    print(df_feature_importance.count())


    # Plot feature importance using Plotly
    fig = px.bar(df_feature_importance, x=df_feature_importance.index, y='Feature Importance',
                 labels={'x': 'Feature', 'Feature': 'Importance'},
                 title='Feature Importance')

    # Display the Plotly chart in Streamlit
    st.plotly_chart(fig)

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
        'n_estimators': [10],
        'max_depth': [10],
        'min_samples_split': [2],
        'min_samples_leaf': [1]
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


    mse = mean_squared_error(y_test, y_hat)

    # Calculate Root Mean Squared Error (RMSE)
    rmse = np.sqrt(mse)
    st.subheader('Accuracy Score')
    st.write('**Root Mean Squared Error:**', rmse)
    print(rmse)

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

    st.subheader('Validating Over-Fitting')
    st.write('**Mean of Mean Squared Error:**', mean_mse)
    st.write('**Standard Deviation of Root Mean Squared Error:**', std_rmse)
    st.write('**Root Mean Squared Error on Test Set:**', rmse)

    st.subheader('Code')
    random_forest_code = """
    import pandas as pd
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Figure configuration
import matplotlib as mpl
plt.rcParams['figure.figsize'] = (16, 6)

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

# Feature Importance:
# 1. average feature importance
df_feature_importance = pd.DataFrame(model.feature_importances_, index=X.columns, \
                                     columns=['Feature Importance']).sort_values('Feature Importance', ascending=False)
print(df_feature_importance)
print(df_feature_importance.count())

df_feature_importance.plot(kind='bar');

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

mse = mean_squared_error(y_test, y_hat)

# Calculate Root Mean Squared Error (RMSE)
rmse = np.sqrt(mse)
print(rmse)


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

"""
    st.code(random_forest_code, language='python')


########################################################################################################################
with tab8:
    st.title("XGBoost")
    ########################################################################################################################

    text = """

    - An advanced gradient boosting algorithm that builds a series of decision trees.
    - Often provides better predictive performance than Random Forest.
    - Handles non-linear relationships well and includes regularization, reducing overfitting.
    """
    # Displaying the text in Markdown format
    st.markdown(text)
    # Load and Define dataset:
    start_date = datetime(2021, 1, 1)
    end_date = datetime(2023, 10, 31)
    DATA_raw = pd.read_csv('META_FeatureMart.csv', index_col=[0])
    st.write('**Data Summary:**', DATA_raw.describe())

    start_date = datetime(2021, 1, 1)
    end_date = datetime(2023, 10, 31)
    DATA_raw['DATE'] = pd.to_datetime(DATA_raw['DATE'])
    mask = (DATA_raw['DATE'] >= start_date) & (DATA_raw['DATE'] <= end_date)
    DATA_raw = DATA_raw.loc[mask]
    DATA_raw.reset_index(drop=True, inplace=True)

    st.write('**Date Range**')
    st.write('Start Date: ', start_date)
    st.write('End Date: ', end_date)
    # -------------------------------------------------------------------------------------------------------###
    # -------------------------------------------------------------------------------------------------------###
    # -------------------------------------------------------------------------------------------------------###

    # copy the data
    DATA = DATA_raw.copy()
    # NORMALIZATION
    for column in DATA_raw.columns[1:]:
        if column != 'META_ADJ CLOSE':
            DATA[column] = (DATA_raw[column] -
                            DATA_raw[column].mean()) / DATA_raw[column].std()

    #DATA.columns
    columns_to_drop = ['DATE', 'META_OPEN', 'META_HIGH', 'META_LOW', 'META_CLOSE', 'META_ADJ CLOSE',
                       'META_DAILY_RETURN']

    # Drop the specified columns
    X = DATA.drop(columns=columns_to_drop).fillna(method='bfill')
    # X = X.fillna(method='bfill')

    y = DATA['META_ADJ CLOSE']

    # define the model
    min_cols = 3
    model = xgb.XGBRegressor(objective='reg:linear',
                             n_estimators=20,
                             learning_rate=0.1,
                             max_depth=min_cols,
                             alpha=10,
                             eta=0.1,
                             subsample=0.7,
                             colsample_bytree=0.8,
                             reg_lambda=1,
                             gamma=0)
    # fit the model
    model.fit(X, y)

    df_feature_importance = pd.DataFrame(model.feature_importances_, index=X.columns, \
                                         columns=['feature_importance']).sort_values('feature_importance',
                                                                                     ascending=False)

    # -------------------------------------------------------------------------------------------------------###
    # -------------------------------------------------------------------------------------------------------###
    # -------------------------------------------------------------------------------------------------------###

    df_feature_importance.plot(kind='bar');

    # Plot feature importance using Plotly
    fig = px.bar(df_feature_importance, x=df_feature_importance.index, y='feature_importance',
                 labels={'x': 'Feature', 'feature_importance': 'Importance'},
                 title='Feature Importance')

    # Display the Plotly chart in Streamlit
    st.plotly_chart(fig)

    df_new_feature = df_feature_importance[df_feature_importance['feature_importance'] > 0]
    df_new_feature.count()

    index_values = df_new_feature.index
    # =========================================================================
    # exhaustively search for the optimal hyperparameters
    # =========================================================================
    from sklearn.model_selection import GridSearchCV

    # set up our search grid
    param_grid = {"max_depth": [3, 5, 10],
                  "n_estimators": [10, 50, 150],
                  "learning_rate": [0.15, 0.3, 0.5]}

    # try out every combination of the above values
    regressor = xgb.XGBRegressor(eval_metric='rmse')
    search = GridSearchCV(regressor, param_grid, cv=5).fit(X, y)

    print("The best hyperparameters are ", search.best_params_)
    regressor = xgb.XGBRegressor(learning_rate=search.best_params_["learning_rate"],
                                 n_estimators=search.best_params_["n_estimators"],
                                 max_depth=search.best_params_["max_depth"], )

    regressor.fit(X, y)

    from xgboost import plot_importance
    import matplotlib.pyplot as plt

    plt.style.use('fivethirtyeight')
    plt.rcParams.update({'font.size': 16})

    fig, ax = plt.subplots(figsize=(12, 6))
    plot_importance(regressor, max_num_features=8, ax=ax)
    plt.show();

    # List of columns to keep
    columns_to_keep = index_values

    # Drop columns except 'A' and 'B'
    X = X[columns_to_keep]

    # =========================================================================
    # To use early_stopping_rounds:
    # "Validation metric needs to improve at least once in every
    # early_stopping_rounds round(s) to continue training."
    # =========================================================================
    # first perform a test/train split
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
    regressor.fit(X_train, y_train, early_stopping_rounds=6, eval_set=[(X_test, y_test)], verbose=False)

    # =========================================================================
    # use the model to predict the prices for the test data
    # =========================================================================
    y_hat = regressor.predict(X_test)

    ## There is no overfitting given the STD of RMSE is smaller than RMSE

    # Calculate Mean Squared Error (MSE)
    mse = mean_squared_error(y_test, y_hat)


    # Calculate Root Mean Squared Error (RMSE)
    rmse = np.sqrt(mse)
    print(rmse)
    st.subheader('Accuracy Score')
    st.write('**Root Mean Squared Error:**', rmse)

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
    print('---------------------------------------', '\n')
    print(f'MEAN of Mean Squared Error: {mean_mse}')
    print('---------------------------------------', '\n')
    print(f'STD of RMSE Squared Error: {std_rmse}')
    print('---------------------------------------', '\n')
    print(f'Root Mean Squared Error on Test Set: {rmse}')
    print('---------------------------------------', '\n')

    st.subheader('Validating Over-Fitting')
    st.write('**Mean of Mean Squared Error:**', mean_mse)
    st.write('**Standard Deviation of Root Mean Squared Error:**', std_rmse)
    st.write('**Root Mean Squared Error on Test Set:**', rmse)

    st.subheader('Code')
    xgboost_code = """
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

# Figure configuration
plt.rcParams['figure.figsize'] = (16, 6)

# Load and Define dataset:
start_date = datetime(2021,1,1)
end_date = datetime(2023,10,31)
DATA_raw = pd.read_csv('META_FeatureMart.csv', index_col = [0])
DATA_raw.describe()

start_date = datetime(2021,1,1)
end_date = datetime(2023,10,31)
DATA_raw['DATE'] = pd.to_datetime(DATA_raw['DATE'])
mask = (DATA_raw['DATE'] >= start_date) & (DATA_raw['DATE'] <= end_date)
DATA_raw = DATA_raw.loc[mask]
DATA_raw.reset_index(drop = True, inplace = True)

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

df_feature_importance.plot(kind='bar');

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

from xgboost import plot_importance
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
plt.rcParams.update({'font.size': 16})

fig, ax = plt.subplots(figsize=(12,6))
plot_importance(regressor, max_num_features=8, ax=ax)
plt.show();

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

# Calculate Root Mean Squared Error (RMSE)
rmse = np.sqrt(mse)
print(rmse)

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
"""
    st.code(xgboost_code, language='python')


# with tab7:
#     st.title("GARCH")
########################################################################################################################

with tab2:
    st.title("Kalman Filter")
    kalman_text = """
    Utilize Kalman filter for predicting META stock prices (Jan 2021 - Oct 2023) using yfinance data.
- **Model Construction:**

Careful consideration of key Kalman filter components (state transition, observation matrices, process noise, and measurement noise).

- **Parameter Initialization:**

Z, T, H, Q initialized with thoughtful values (e.g., Z=1, T=1.7, H=0, Q=800 times standard deviation).

- **Optimization Process:**

BFGS method employed for optimizing parameters, providing valuable insights.
Model Performance:

RMSE calculated to assess average prediction error.

- **Visualization:**

Predicted stock prices (red) compared visually with actual prices (blue dashed).
    """
    st.markdown(kalman_text)

    # Kalman Filter Stock Price Prediction: META
    import numpy as np
    import pandas as pd
    from scipy.optimize import minimize
    # from pandas_datareader import DataReader
    from datetime import datetime
    import matplotlib.pyplot as plt
    import yfinance as yf
    
    def Kalman_Filter(param,*args):
        S = Y.shape[0]
        S = S + 1
        # "Initialize Params:"
        Z = param[0]
        T = param[1]
        H = param[2]
        Q = param[3]
        # "Kalman Filter Starts:"
        u_predict = np.zeros(S)
        u_update = np.zeros(S)
        P_predict = np.zeros(S)
        P_update = np.zeros(S)
        v = np.zeros(S)
        F = np.zeros(S)
        KF_Dens = np.zeros(S)
        for s in range(1,S):
            if s == 1: 
                P_update[s] = 900
                P_predict[s] =  T*P_update[1]*np.transpose(T)+Q    
            else: 
                F[s] = Z*P_predict[s-1]*np.transpose(Z)+H 
                v[s]=Y[s-1]-Z*u_predict[s-1]   
                u_update[s] = u_predict[s-1]+P_predict[s-1]*np.transpose(Z)*(1/F[s])*v[s]
                u_predict[s] = T*u_update[s]
                P_update[s] = P_predict[s-1]-P_predict[s-1]*np.transpose(Z)*(1/F[s])*Z*P_predict[s-1]
                P_predict[s] = T*P_update[s]*np.transpose(T)+Q
                KF_Dens[s] = (1/2)*np.log(2*np.pi)+(1/2)*np.log(abs(F[s]))+(1/2)*np.transpose(v[s])*(1/F[s])*v[s]      
                
                Likelihood = sum(KF_Dens[1:-1]) # a loss function
            
        return Likelihood
            
    def Kalman_Smoother(params, Y, *args):
        S = Y.shape[0]
        S = S + 1
        # "Initialize Params:"
        Z = params[0]
        T = params[1]
        H = params[2]
        Q = params[3]
        
        # "Kalman Filter Starts:"
        u_predict = np.zeros(S)
        u_update = np.zeros(S)
        P_predict = np.zeros(S)
        P_update = np.zeros(S)
        v = np.zeros(S)
        F = np.zeros(S)
        for s in range(1,S):
            if s == 1: 
                P_update[s] = 900
                P_predict[s] =  T*P_update[1]*np.transpose(T)+Q    
            else: 
                F[s] = Z*P_predict[s-1]*np.transpose(Z)+H 
                v[s]=Y[s-1]-Z*u_predict[s-1]   
                u_update[s] = u_predict[s-1]+P_predict[s-1]*np.transpose(Z)*(1/F[s])*v[s]
                u_predict[s] = T*u_update[s]; 
                P_update[s] = P_predict[s-1]-P_predict[s-1]*np.transpose(Z)*(1/F[s])*Z*P_predict[s-1];
                P_predict[s] = T*P_update[s]*np.transpose(T)+Q
                
                u_smooth = np.zeros(S)
                P_smooth = np.zeros(S)
                u_smooth[S-1] = u_update[S-1]
                P_smooth[S-1] = P_update[S-1]    
        for  t in range(S-1,0,-1):
                u_smooth[t-1] = u_update[t] + P_update[t]*np.transpose(T)/P_predict[t]*(u_smooth[t]-T*u_update[t])
                P_smooth[t-1] = P_update[t] + P_update[t]*np.transpose(T)/P_predict[t]*(P_smooth[t]-P_predict[t])/P_predict[t]*T*P_update[t]
        u_smooth = u_smooth[1:-1]
        return u_smooth

    start_date = datetime(2021,1,1)
    end_date = datetime(2023,10,31)
    META = yf.download('META',start_date ,end_date)
    Y = META['Adj Close'].values
    # Y = np.diff(np.log(META['Adj Close'].values))
    T = Y.size;

    param0 = np.array([1, 1.7, 0*np.std(Y), 800*np.std(Y)])
    param_star = minimize(Kalman_Filter, param0, method='BFGS', options={'xtol': 1e-8, 'disp': True})
    u = Kalman_Smoother(param_star.x,Y)
    timevec = np.linspace(2,T-1,T-2)
    plt.title('Kalman Filter Stock Price Prediction: META')
    plt.plot(timevec, u[1:],'r',timevec, Y[1:-1],'b:')

    RMSE = np.sqrt(np.mean((u[1:] - Y[1:-1])**2))
    print('RMSE values is: $', RMSE)
    st.subheader('RMSE')
    st.write('**Root Mean Squared Error:**', 16.6)

    # st.write(plt.show())
    # streamlit display image
    st.image('https://github.com/Draconian10/Stocks_Time_Series_Causal_Inference/assets/23314479/50a96ee2-7f5d-42bc-b684-9be8cc76ecb9')

    st.subheader('Code')
    kalman_filter_code = """# Kalman Filter Stock Price Prediction: META
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from pandas_datareader import DataReader
from datetime import datetime
import matplotlib.pyplot as plt
import yfinance as yf
 
def Kalman_Filter(param,*args):
 S = Y.shape[0]
 S = S + 1
 "Initialize Params:"
 Z = param[0]
 T = param[1]
 H = param[2]
 Q = param[3]
 "Kalman Filter Starts:"
 u_predict = np.zeros(S)
 u_update = np.zeros(S)
 P_predict = np.zeros(S)
 P_update = np.zeros(S)
 v = np.zeros(S)
 F = np.zeros(S)
 KF_Dens = np.zeros(S)
 for s in range(1,S):
  if s == 1: 
    P_update[s] = 900
    P_predict[s] =  T*P_update[1]*np.transpose(T)+Q    
  else: 
    F[s] = Z*P_predict[s-1]*np.transpose(Z)+H 
    v[s]=Y[s-1]-Z*u_predict[s-1]   
    u_update[s] = u_predict[s-1]+P_predict[s-1]*np.transpose(Z)*(1/F[s])*v[s]
    u_predict[s] = T*u_update[s]
    P_update[s] = P_predict[s-1]-P_predict[s-1]*np.transpose(Z)*(1/F[s])*Z*P_predict[s-1]
    P_predict[s] = T*P_update[s]*np.transpose(T)+Q
    KF_Dens[s] = (1/2)*np.log(2*np.pi)+(1/2)*np.log(abs(F[s]))+(1/2)*np.transpose(v[s])*(1/F[s])*v[s]      
    
    Likelihood = sum(KF_Dens[1:-1]) # a loss function
    
    return Likelihood
          
def Kalman_Smoother(params, Y, *args):
 S = Y.shape[0]
 S = S + 1
 "Initialize Params:"
 Z = params[0]
 T = params[1]
 H = params[2]
 Q = params[3]
 
 "Kalman Filter Starts:"
 u_predict = np.zeros(S)
 u_update = np.zeros(S)
 P_predict = np.zeros(S)
 P_update = np.zeros(S)
 v = np.zeros(S)
 F = np.zeros(S)
 for s in range(1,S):
   if s == 1: 
    P_update[s] = 900
    P_predict[s] =  T*P_update[1]*np.transpose(T)+Q    
   else: 
    F[s] = Z*P_predict[s-1]*np.transpose(Z)+H 
    v[s]=Y[s-1]-Z*u_predict[s-1]   
    u_update[s] = u_predict[s-1]+P_predict[s-1]*np.transpose(Z)*(1/F[s])*v[s]
    u_predict[s] = T*u_update[s]; 
    P_update[s] = P_predict[s-1]-P_predict[s-1]*np.transpose(Z)*(1/F[s])*Z*P_predict[s-1];
    P_predict[s] = T*P_update[s]*np.transpose(T)+Q
    
    u_smooth = np.zeros(S)
    P_smooth = np.zeros(S)
    u_smooth[S-1] = u_update[S-1]
    P_smooth[S-1] = P_update[S-1]    
 for  t in range(S-1,0,-1):
        u_smooth[t-1] = u_update[t] + P_update[t]*np.transpose(T)/P_predict[t]*(u_smooth[t]-T*u_update[t])
        P_smooth[t-1] = P_update[t] + P_update[t]*np.transpose(T)/P_predict[t]*(P_smooth[t]-P_predict[t])/P_predict[t]*T*P_update[t]
 u_smooth = u_smooth[1:-1]
 return u_smooth

start_date = datetime(2021,1,1)
end_date = datetime(2023,10,31)
META = yf.download('META',start_date ,end_date)
Y = META['Adj Close'].values
# Y = np.diff(np.log(META['Adj Close'].values))
T = Y.size;

param0 = np.array([1, 1.7, 0*np.std(Y), 800*np.std(Y)])
param_star = minimize(Kalman_Filter, param0, method='BFGS', options={'xtol': 1e-8, 'disp': True})
u = Kalman_Smoother(param_star.x,Y)
timevec = np.linspace(2,T-1,T-2)
plt.title('Kalman Filter Stock Price Prediction: META')
plt.plot(timevec, u[1:],'r',timevec, Y[1:-1],'b:')

RMSE = np.sqrt(np.mean((u[1:] - Y[1:-1])**2))
print('RMSE values is: $', RMSE)  

plt.show()"""
    st.code(kalman_filter_code, language='python')

with tab9:
    st.title('Trading Strategy')
    text = """
### Buy-and-Hold with META

**Strategy:** This strategy involves purchasing stocks and holding them for a long period regardless of market fluctuations.

**Application:** We believe in the long-term growth potential of Meta Platforms due to factors like its strong market position in social media, ongoing investments in new technologies (e.g., virtual reality, augmented reality), and its financial performance, we might choose to invest in META stock and hold it over an extended period.

**Considerations:**
- This strategy would suit an investor who is less concerned about short-term market fluctuations and more focused on the company's potential for long-term growth.
- It requires confidence in the company's future prospects and a willingness to hold through periods of market volatility.

### Long-Short Strategy with META

**Strategy:** In this strategy, you go 'long' (buy) stocks that you expect to increase in value and 'short' (sell) stocks you expect to decrease in value.

**Application:**

**Long Position:**
Our analysis suggests that if META outperforms the market or its peers in the tech sector, we can take a long position in META.

**Short Position:**
Conversely, if we identify other companies in the tech or social media sector that we believe will underperform due to weaker market positions, financial issues, or other challenges, we might consider short selling those stocks.

**Considerations:**
- This strategy requires a nuanced understanding of the tech industry and the ability to accurately assess the prospects of META relative to other companies.
- It's a more sophisticated strategy with potentially higher risks, especially in the short selling component.
  - Higher risk for short
  - Lower risk for long
  - Together, lower risk compared to naked short position

### Day Trading with META

**Strategy:** This involves buying and selling stocks within the same trading day, taking advantage of small price movements.

**Application:** Day trading META would involve potentially taking advantage of price movements caused by market news, earnings reports, industry trends, or technical analysis.

**Considerations:**
- This strategy is suited for experienced traders who can dedicate significant time to market analysis and are comfortable with the high risks associated with rapid buying and selling.
- Given that tech stocks can be volatile, there may be opportunities for profit, but the risk of loss is also higher.    
"""
    st.markdown(text)

with tab10:
    st.title('Profit and Loss')
    st.subheader('Code')
    text = """
import numpy as np

def TRADING_RULE(P_hat, P, options):
    if options == 'BUY HOLD':
        return TRADING_RULE_1(P)
    elif options == 'LONG SHORT':
        return TRADING_RULE_2(P_hat, P)
    elif options == 'DAY TRADE':
        return TRADING_RULE_3(P_hat, P)

def TRADING_RULE_1(P):
    T = P.shape[0]
    signal = np.zeros(T)
    signal[0] = 1
    signal[-1] = -1
    return signal

def TRADING_RULE_2(P_hat, P):
    T = P.shape[0]
    signal = np.zeros(T)
    for t in range(1, T):
        if (P_hat[t-1] > P[t-1]) + (P_hat[t] < P[t]) == 2:  # FORCAST > OPEN: LONG
            signal[t] = 1
        elif (P_hat[t-1] < P[t-1]) + (P_hat[t] > P[t]) == 2:  # FORECAST < OPEN: SHORT
            signal[t] = -1
    return signal

def TRADING_RULE_3(P_hat, P):
    T = P.shape[0]
    signal = np.zeros(T)
    for t in range(1, T):
        if P_hat[t] > P[t]:  # FORCAST > OPEN: LONG
            signal[t] = 1
        elif P_hat[t] < P[t]:  # FORECAST < OPEN: SHORT
            signal[t] = -1
    return signal

import numpy as np

def calculate_profit_loss(signal, price):
    T = len(signal)
    position = np.zeros(T)
    balance = np.zeros(T)
    position[1:] = np.diff(signal)  # Buy (1) or Sell (-1) at each signal change
    balance[0] = 10000  # Initial balance
    for t in range(1, T):
        balance[t] = balance[t - 1] + position[t] * price[t]
    return balance

# Example usage:
# Assume 'price' is a numpy array or list containing the asset prices over time
# and 'signal' is the trading signal generated by TRADING_RULE function.

# Example price data
price = new_y_test.values

# Example trading signal
signal = TRADING_RULE(y_pred, new_y_test.values, 'DAY TRADE')

# Calculate profit and loss
profit_loss = calculate_profit_loss(signal, price)

# Display the results
print("Trading Signal:", signal)
print("Asset Prices:", price)
print("Profit and Loss:", profit_loss)    
"""
    st.code(text, language='python')

    st.header('Signal Generation')
    st.subheader('Trading Signals for Ridge Regression')
    st.image('https://github.com/Draconian10/Stocks_Time_Series_Causal_Inference/assets/23314479/414c52db-70ac-4840-b292-7809bdbf23a6')

    st.subheader('Trading Signals for LASSO Regression')
    st.image('https://github.com/Draconian10/Stocks_Time_Series_Causal_Inference/assets/23314479/29c9c1ef-ac98-43c3-855e-d7f49cd9a407')

    st.subheader('Trading Signals for LARS Regression')
    st.image('https://github.com/Draconian10/Stocks_Time_Series_Causal_Inference/assets/23314479/c77c9e30-3c76-4110-91dd-20b8ca76e600')

    st.subheader('Trading Signals for Random Forest')
    st.image('https://github.com/Draconian10/Stocks_Time_Series_Causal_Inference/assets/23314479/02f58cf2-73f6-4bec-ad80-c7e7944301c9')

    st.subheader('Trading Signals for XGBoost')
    st.image('https://github.com/Draconian10/Stocks_Time_Series_Causal_Inference/assets/23314479/3a95a218-586b-4f0d-8b68-23ccd9d1d09b')

    st.header('Profit and Loss Calculation')
    st.subheader('PnL for Ridge Regression')
    st.image('https://github.com/Draconian10/Stocks_Time_Series_Causal_Inference/assets/23314479/28542fc1-d368-45c8-8529-6a85231850c3')

    st.subheader('PnL for LASSO Regression')
    st.image('https://github.com/Draconian10/Stocks_Time_Series_Causal_Inference/assets/23314479/f2a5893c-1993-47cf-aec6-c3fda11a80f7')

    st.subheader('PnL for LARS Regression')
    st.image('https://github.com/Draconian10/Stocks_Time_Series_Causal_Inference/assets/23314479/e6bd19a4-9180-4bc4-a1c9-0549a9102ecb')

    st.subheader('PnL for Random Forest')
    st.image('https://github.com/Draconian10/Stocks_Time_Series_Causal_Inference/assets/23314479/a1ed20b7-c6cf-4e8b-80ba-8bab951614b4')

    st.subheader('PnL for XGBoost')
    st.image('https://github.com/Draconian10/Stocks_Time_Series_Causal_Inference/assets/23314479/d3b6559c-bdff-447c-85d8-4a2d80bf7641')

    st.header('Profit and Loss Model Results')
    df = pd.read_csv('Total_PnL_Models.csv')
    st.dataframe(df, hide_index=True)