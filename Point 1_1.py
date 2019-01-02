#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Libraries

import random
import pandas as pd
import numpy as np
import datetime
import time
from sklearn.model_selection import train_test_split
from scipy.optimize import minimize
import operator
import itertools
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import warnings
warnings.filterwarnings("ignore")


# In[2]:


# Directories

d = pd.read_csv(open("DATA.csv","r"))


# In[3]:


# Transform columns of the data frame in array

X = d.iloc[:,0:2].values
y = d.iloc[:,2].values


# ### Functions

# In[4]:


# Define the main function

def f_x(v, w, b, N, x):
    z1 = np.dot(x,w) - b 
    a1 = np.tanh(z1)
    z2 = np.dot(a1,v)
    return z2


# In[5]:


# Define the error function

def error(pars, args, params = 1):
    '''params = 1 -> evaluate error with rho values
       params = 0 -> evaluate error without rho values
       Rho is used just for "minimazing the regularized training error" '''
    
    N = args[0]
    x = args[1]
    y = args[2]
    rho = args[3]
    
    P = len(x)
    
    v = pars[0:N]
    w = pars[N:N*3]
    b = pars[N*3:N*4]
    
    # reshape the w: form array to matrix
    w = np.array(w)
    w = w.reshape(2,N)
   
    # predict the y
    y_mlp = f_x(v, w, b, N, x)
    
    # compute the error
    if params == 0:
        return (1/(2*P))*(np.linalg.norm(y_mlp - y))**2
    else:
        return (1/(2*P))*(np.linalg.norm(y_mlp - y))**2 + (rho*(np.linalg.norm(v))**2 + rho*(np.linalg.norm(w))**2 
                                                           +rho*(np.linalg.norm(b))**2)


# In[6]:


# Function to find the best parameters with fixed rho

def best_values (x_train,y_train,x_test,y_test,N=21,rho=0.00001, ini=2):
    ''' x_train = x_train set
        y_train = y_train set
        x_test = x_test set
        y_test = y_test set
        N = numbers of hidden layers, def=21
        rho = initial rho values, def=0.0005 (midrange rho)
        ini = number of repetitions to reduce the beginnings with "bad points", def=2
    '''
    
    start_time = time.time()
    
    # initialize a dictionary and a counter
    results = {}
    nume=0
    for _ in range(ini):
        random.seed(_)
        # define the random initial parameters
        v_init = [random.uniform(0, 1) for i in range(N)]
        w_init = [random.uniform(0, 1) for i in range(N*2)]
        b_init = [random.uniform(0, 1) for i in range(N)]
        
        # "minimize function" needs a list, non a list of lists: let's give to it a list and 
        # then split it into the 3 different parameters inside the "error" function 
        params = (v_init + w_init + b_init)
        # define the argoument of the "error" function
        err_args = [N, x_train, y_train, rho]
        
        ini_training_erro=error(params,err_args,0)
        
        res = minimize(error, params, args=err_args, method='BFGS')
        grad= np.linalg.norm(res["jac"])
        nfev= res["nfev"]
        pars_final = res['x']
        v_final = pars_final[0:N]
        w_final = pars_final[N:3*N]
        b_final = pars_final[3*N:4*N]
        
        if nume == 0:
            results[N, rho] = {'v': v_final, 'w': w_final, 'b': b_final, 'Error': res['fun'], "Gradient": grad,
                               "Nfev": nfev }
        else:
            if results[N, rho]["Error"] > res['fun']:
                results[N, rho] = {'v': v_final, 'w': w_final, 'b': b_final, 'Error': res['fun'], "Gradient": grad,
                               "Nfev": nfev}

    nume+=1 
    end_time = time.time()
    tempo=(end_time - start_time)
    
    paramss = [results[N,rho]["v"],results[N,rho]["w"],results[N,rho]["b"]]
    paramss = list(itertools.chain(*paramss))
    err_argss = [N, x_test, y_test, rho]
    test_error = error(paramss, err_argss, 0)
    
    
    return (N,ini_training_erro,rho,results,tempo,test_error,ini)


# In[7]:


# Create train (75% of data) and test (25% of data) using "Matricola" as seed

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1656079)


# In[8]:


results=best_values(x_train,y_train,x_test,y_test)


# In[9]:


print("Number of Neurons N:... %s" %(results[0]))
print("Initial Training Error:... %s" %(results[1]))
print("Final Training Error:... %s" %(results[3][(results[0],results[2])]["Error"]))
print("Final Test Error:... %s" %(results[5]))
print("Optimization solver chosen:... BFGS")
print("Norm of the gradient at the optimal point:... %s" %(results[3][(results[0],results[2])]["Gradient"]))
print("Total Time for optimizing the network:... %s seconds" %(results[4]))
print("But we repeat code 3 times for avoiding 'Bad Starts', so time for optimizing the network:... %s seconds" %((results[4]/results[6])))
print("Number of function evaluations:... %s" %(results[3][(results[0],results[2])]["Nfev"]))
print("Value of sigma:... 1")
print("Value of rho:... %s" %(results[2]))
print("Other hyperparameters:... \n None")


# In[ ]:




