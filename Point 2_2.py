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


# Transform columns of the data frame in array

X = d.iloc[:,0:2].values
y = d.iloc[:,2].values


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


# In[9]:


# Define the error function to be optimized

def error(pars, args,params = 1):
    N = args[0]
    x = args[1]
    y = args[2]
    
    rho = args[3]
    
    sigma = args[4]
    c = args [5]
    
    P = len(x)
    
    v = pars[0:N]
    
    # reshape the c: form array to matrix
    c = np.array(c).reshape(N, 2)
    
    # predict the y
    y_rbf = y_x(v, c, sigma, x,N)
    # compute the error
    w=np.array(v)
    
    # compute the error
    if params == 0:
        return (1/(2*P))*(np.linalg.norm(y_rbf - y))**2
    if params ==1:
        return (1/(2*P))*(np.linalg.norm(y_rbf - y))**2 + (rho*(np.linalg.norm(v))**2)


# In[10]:


def best_values (x_train,y_train,x_test,y_test,N=50,rho=0.00001, ini=2, sigma=1.0):
    ''' x_train = x_train set
        y_train = y_train set
        x_test = x_test set
        y_test = y_test set
        N = numbers of hidden layers, def=50
        rho = initial rho values, def=0.00001 (min rho)
        ini = number of repetitions to reduce the beginnings with "bad points", def=2
        sigma = sigma values, def = 1.0
    '''
    
    start_time = time.time()
    # initialize a dictionary and a counter
    results = {}
    nume=0
    for _ in range(ini):
        random.seed(_)
        # define the random initial parameters
        v_init = [random.uniform(0, 1) for i in range(N)]
        
        kmeans=KMeans(n_clusters=N, random_state=_)
        a=kmeans.fit(x_train)
        c_init=a.cluster_centers_
        c_init = list(itertools.chain(*c_init))
        
        
        # "minimize function" needs a list, non a list of lists: let's give to it a list and 
        # then split it into the 3 different parameters inside the "error" function 
        
        params = (v_init)
            
        # define the argoument of the "error" function
        err_args = [N, x_train, y_train, rho, sigma, c_init]
        
        ini_training_erro=error(params,err_args,0)
        
        res = minimize(error, params, args=err_args, method='BFGS', jac=grad_wrt_v)
        grad= np.linalg.norm(res["jac"])
        nfev= res["nfev"]
        pars_final = res['x']
        v_final = pars_final[0:N]
        c_final = c_init
        
        if nume == 0:
            results[N, rho, sigma] = {'v': v_final, 'c': c_final, 'error': res['fun'],"Gradient": grad,
                               "Nfev": nfev }
        else:
            if results[N, rho, sigma]["error"] > res['fun']:
                results[N, rho, sigma] = {'v': v_final, 'c': c_final, 'error': res['fun'], "Gradient": grad,
                               "Nfev": nfev }
    nume+=1 
    end_time = time.time()
    tempo=(end_time - start_time)
    
    paramss = [results[N,rho,sigma]["v"]]
    paramss = list(itertools.chain(*paramss))
    err_argss = [N, x_test, y_test, rho, sigma,c_init]
    test_error = error(paramss, err_argss, 0)
        
    return (N,ini_training_erro,rho,results,tempo,test_error,ini,sigma)


# In[11]:


results=best_values(x_train,y_train,x_test,y_test, ini=5)


# In[12]:


print("Number of Neurons N:... %s" %(results[0]))
print("Initial Training Error:... %s" %(results[1]))
print("Final Training Error:... %s" %(results[3][(results[0],results[2],results[7])]["error"]))
print("Final Test Error:... %s" %(results[5]))
print("Optimization solver chosen:... BFGS")
print("Norm of the gradient at the optimal point:... %s" %(results[3][(results[0],results[2],results[7])]["Gradient"]))
print("Total Time for optimizing the network:... %s seconds" %(results[4]))
print("But we repeat code 5 times for avoiding 'Bad Starts', so time for optimizing the network:... %s seconds" %((results[4]/results[6])))
print("Number of function evaluations:... %s" %(results[3][(results[0],results[2],results[7])]["Nfev"]))
print("Value of sigma:... %s" %(results[7]))
print("Value of rho:... %s" %(results[2]))
print("Other hyperparameters:... \n None")


# In[ ]:




