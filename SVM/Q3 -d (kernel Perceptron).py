#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from scipy.optimize import minimize, LinearConstraint, Bounds
import time
from termcolor import colored
import math
import numpy.linalg as LA


# In[2]:


X_t = np.genfromtxt('bank-note/train.csv', delimiter=",")
X_test = np.genfromtxt('bank-note/test.csv', delimiter=",")
y_t = (2*X_t[:,-1] - 1).reshape(-1,)
y_test = (2*X_test[:,-1]-1).reshape(-1,)


# In[3]:


X_t = X_t[:,:-1]
X_test = X_test[:,:-1]


# In[4]:


def G_k(A, B, gamma):  #Resturn a matrix whose ijth entry is exp{-||A_i-B_j||**2/gamma}
    temp = np.sum(A * A, 1).reshape(A.shape[0], 1) + np.sum(B * B, 1).reshape(1, B.shape[0]) - 2 * A @ B.T
    return np.exp(-temp/gamma)


# In[5]:


num_train = X_t.shape[0]


# In[6]:


def f(G, C, j):
    return np.dot((C*y_t), G[j])


# In[7]:


def K(x,z,gamma):
    return(math.exp((- LA.norm(x-z)**2)/gamma))


# In[8]:


def predict(x,gamma):
    s = 0
    for i in range(num_train):
        s += C[i]*y_t[i]*K(X_t[i],x,gamma)
    if s>=0:
        return 1
    return -1


# In[9]:


def kernel_Perceptron(gamma):
    G = G_k(X_t, X_t, gamma)
    C = np.zeros(num_train)
    for i in range(10):
        L = np.random.permutation(num_train)
        for k in range(num_train):
            j = L[k]
            if y_t[j]*f(G, C,j) <= 0:
                C[j] += 1
    return C


# In[10]:


def pred(C, X, gamma):
    P = []
    for i in range(X.shape[0]):
        P.append(predict(X[i],gamma))
    return np.array(P)


# In[11]:


Gamma = [ 0.1, 0.5, 1, 5, 100]


# In[12]:


for gamma in Gamma:
    print('kernel Perceptron algorithm for $\gamma = {}$\\\\'.format(gamma))
    C = kernel_Perceptron(gamma)
    P_train = pred(C, X_t, gamma)
    print('Train error = {}'.format(1-(P_train == y_t).mean()))
    P_test = pred(C, X_test, gamma)
    print('Test error = {}'.format(1-(P_test == y_test).mean()))
    


# In[ ]:




