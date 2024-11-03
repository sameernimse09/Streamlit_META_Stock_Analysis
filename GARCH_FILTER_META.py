# Garch Filter Stock Price Prediction: META
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from pandas_datareader import DataReader
from datetime import datetime
import matplotlib.pyplot as plt
import yfinance as yf

def GARCH(param, *args):
 "Initialize Params:"
 mu = param[0]
 omega = param[1]
 alpha = param[2]
 beta = param[3]
 T = Y.shape[0]
 GARCH_Dens = np.zeros(T) 
 sigma2 = np.zeros(T)   
 F = np.zeros(T)   
 v = np.zeros(T)   
 for t in range(1,T):
    sigma2[t] = omega + alpha*((Y[t-1]-mu)**2)+beta*(sigma2[t-1]); 
    F[t] = Y[t] - mu-np.sqrt(sigma2[t])*np.random.normal(0,1,1)
    v[t] = sigma2[t]
    GARCH_Dens[t] = (1/2)*np.log(2*np.pi)+(1/2)*np.log(v[t])+\
                    (1/2)*(F[t]/v[t])     
    
 Likelihood = np.sum(GARCH_Dens[1:-1])  
 return Likelihood


def GARCH_PROD(params, Y0, T):
 mu = params[0]
 omega = params[1]
 alpha = params[2]
 beta = params[3]
 Y = np.zeros(T)  
 sigma2 = np.zeros(T)
 Y[0] = Y0
 sigma2[0] = 0.003
 for t in range(1,T):
    sigma2[t] = omega + alpha*((Y[t-1]-mu)**2)+beta*(sigma2[t-1]); 
    Y[t] = mu+np.sqrt(sigma2[t])*np.random.normal(0,1,1)    
 return Y    

# 1. Simulated Data
# T  = 1000
# mu = 35;
# sig = 5;
# Y = np.random.normal(mu,sig,T);
# 2. Real Data
start_date = datetime(2021,1,1)
end_date = datetime(2023,10,31)
META = yf.download('META',start_date ,end_date)
Y = META['Adj Close'].values
# Y = np.diff(np.log(META['Adj Close'].values))
T = Y.size;


param0 = np.array([np.mean(Y), np.var(Y)/9, 0.3, 0.3])
param_star = minimize(GARCH, param0, method='BFGS', options={'xtol': 1e-8, 'disp': True})
# param_star = minimize(GARCH, param0, method='Powell', options={'disp': True, 'maxiter': 1000})

Y_GARCH = GARCH_PROD(param_star.x, Y[0], T)
timevec = np.linspace(1,T,T)
plt.plot(timevec, Y_GARCH,'r',timevec, Y,'b')
plt.title('Garch Filter Stock Price Prediction: META')

RMSE = np.sqrt(np.mean((Y_GARCH - Y)**2))
print('RMSE values is:', RMSE)

plt.show()