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
from sklearn.cluster import KMeans
from sklearn import preprocessing 


# In[2]:


# Directories

d = pd.read_csv(open("DATA.csv","r"))


# In[3]:


# Transform columns of data frame in array

X = d.iloc[:,0:2].values
y = d.iloc[:,2].values


# ## Functions

# In[4]:


# Create train (75% of data) and test (25% of data) using "Matricola" as seed

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1656079)


# In[5]:


def x_c(x,c,N):
    n = len(x)
    nd = len(x.T)
    
    c = np.array(c)
    c = c.reshape(N,nd,1)
    
    x = x.reshape(n,nd,1)
    
    return x-c.T


# In[6]:


def gaussian(xc,sigma):
        
    return np.exp(-(np.linalg.norm(xc, axis = 1)/sigma) ** 2)


# In[7]:


# Define the RBF network
## wants the argouments: 
### c, w (the parameters of the network)
### N: # of neurons in the hidden layer
### sigma: variance used in the gaussian function
### x: input data

def y_x(v, c, sigma, x, N):
    y=np.dot(gaussian(x_c(x,c,N),sigma),v)
        
    return y    


# In[8]:


# Define the error function to be optimized

def error_v(pars, args,params = 1):
    
    N = args[0]
    x = args[1]
    y = args[2]
    rho = args[3]
    sigma = args[4]
    c = args[5]
    P = len(x)
    
    v = pars
    
    # predict the y
    y_rbf = y_x(v, c, sigma, x, N)
    
    
    # compute the error
    if params == 0:
        return (1/(2*P))*(np.linalg.norm(y_rbf - y))**2
    if params ==1:
        return (1/(2*P))*(np.linalg.norm(y_rbf - y))**2 + (rho*(np.linalg.norm(v))**2)


# In[9]:


# Define the error function refer to c to be optimized

def error_c(pars, args,params = 1):
    
    N = args[0]
    x = args[1]
    y = args[2]
    rho = args[3]
    sigma = args[4]
    v = args[5]
    P = len(x)
    
    c = pars
    
    # predict the y
    y_rbf = y_x(v, c, sigma, x, N)
    
    
    # compute the error
    if params == 0:
        return (1/(2*P))*(np.linalg.norm(y_rbf - y))**2
    if params ==1:
        return (1/(2*P))*(np.linalg.norm(y_rbf - y))**2 + (rho*(np.linalg.norm(c))**2)


# In[10]:


#Gradient_c

def grad_wrt_c(pars, args):
    
    N = args[0]
    x = args[1]
    y = args[2]
    rho = args[3]
    sigma = args[4]
    v = args[5]
    
    c = pars[0:2*N]

    n=len(x)
    v=np.array(v)
    c=np.array(c)

    c = c.reshape(N, 2)
    x_reshaped = np.array(x).reshape(n, 1, 2)

    g = gaussian(x_c(x, c, N), sigma)[:, :, np.newaxis]
    arg_g = x_reshaped - c
    arg_g = g * arg_g * (2/sigma**2)
    
    vv = v.reshape(1, N, 1)
    rest = y_x(v, c, sigma, x, N) - y
    d_c = rest[:, np.newaxis, np.newaxis] * arg_g
    d_c = d_c * vv
    d_c = np.mean(d_c, axis = 0) + 2 * rho * c
    
    return d_c.reshape((N * 2, ))


# In[11]:


# Gradient_v

def grad_wrt_v(pars, args):
    
    N = args[0]
    x = args[1]
    y = args[2]
    rho = args[3]
    sigma = args[4]
    c = args [5]
    
    v = pars
    
    n=len(x)
    v=np.array(v)
    c=np.array(c)

    
    res = np.dot(gaussian(x_c(x,c,N),sigma),v)[:, np.newaxis] - y.reshape(n, 1)
    d_v = (np.dot(gaussian(x_c(x,c,N).T, sigma), res))/n
    d_v = d_v + 2 * rho * v.reshape(N,1)
    
    return d_v[:, 0]


# In[12]:


# Function to find the best parameters with fixed rho

def decomposition (x_train,y_train,x_test,y_test,N=50,rho=0.00001,error=0.001,stop=None, sigma=1):
    ''' x_train = x_train set
        y_train = y_train set
        x_test = x_test set
        y_test = y_test set
        N = numbers of hidden layers, def=50
        rho = initial rho values, def=0.00001 (min rho)
        error= target error, def = 0.001
        stop= early stopping, def = None
        sigma= sigma values, def=1
    '''
    
    
    random.seed(1656079)
    
    #set the initial parameters
    
    v_init = [random.uniform(0, 1) for i in range(N)]
    kmeans=KMeans(n_clusters=N, random_state=1656079)
    a=kmeans.fit(x_train)
    c_init=a.cluster_centers_
    c_init = list(itertools.chain(*c_init))
    
    
    # calculate the initial training error
    params = (v_init)
    err_args = [N, x_train, y_train, rho, sigma, c_init]
    
    ini_training_erro = error_v(v_init, err_args, 0)
    error_check = ini_training_erro
    
    #create the dictionary for results
    results = {}
    nfev = 0
    contatore = 0 #Number of outer iterations
    
    
    start_time = time.time()
    
    
    while error_check > error:
        
        #optimaze just v
        
        params = (v_init)
        err_args = [N, x_train, y_train, rho, sigma,c_init]
        res = minimize(error_v, params, args=err_args, method='BFGS', jac=grad_wrt_v)
        grad = np.linalg.norm(res["jac"])
        nfev += res["nfev"]
        
        v_init = res.x.tolist()
        contatore +=1
        
        params = (v_init) #upload for right values in test error
        err_args = [N, x_test, y_test, rho, sigma, c_init]
        
        error_check = res['fun']
        
        
        
        
        if error_check < error:
            break
            
        if stop != None:
            if contatore == stop:
                print("Early Stopping after %s iterations " %(contatore))
                break
        
        
        #optimaze c
        
        params = (c_init)
        err_args = [N, x_train, y_train, rho , sigma, v_init]
        res = minimize(error_c, params, args=err_args, method='BFGS',jac=grad_wrt_c)
        grad = np.linalg.norm(res["jac"])
        nfev += res["nfev"]
        
        c_init = res.x.tolist()
        contatore +=1
        
        params = (c_init)
        err_args = [N, x_test, y_test, rho , sigma, v_init]
        
        error_check = res['fun']
        
        if stop != None:
            if contatore == stop:
                print("Early Stopping after %s iterations " %(contatore))
                break
    
        
        
    end_time = time.time()
    tempo=(end_time - start_time)
    
    try:
        test_error = error_v(params, err_args, 0)
    except:
        test_error = error_c(params, err_args, 0)
        
    results[N, rho, sigma] = {'v': v_init, 'c': c_init, 'Train Error': res['fun'], 'Test Error': test_error,
                      "Gradient": grad}
        
    
    return (N,ini_training_erro,rho,results,tempo,nfev,test_error,contatore,sigma)


# In[13]:


results=decomposition(x_train,y_train,x_test,y_test,stop=115,N=50)


# In[14]:


print("Number of Neurons N:... %s" %(results[0]))
print("Initial Training Error:... %s" %(results[1]))
print("Final Training Error:... %s" %(results[3][(results[0],results[2],results[8])]["Train Error"]))
print("Final Test Error:... %s" %(results[6]))
print("Optimization solver chosen:... BFGS")
print("Norm of the gradient at the optimal point:... %s" %(results[3][(results[0],results[2],results[8])]["Gradient"]))
print("Total Time for optimizing the network:... %s seconds" %(results[4]))
print("Number of function evaluations:... %s" %(results[5]))
print("Value of sigma:... %s" %(results[8]))
print("Value of rho:... %s" %(results[2]))
print("Other hyperparameters:... \n Number of outer iterations: %s" %(results[7]))


# In[ ]:





# In[ ]:




