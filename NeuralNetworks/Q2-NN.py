#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import time
import random
from MyNN import *


# In[2]:


data = np.genfromtxt('bank-note/train.csv', delimiter=",")
test = np.genfromtxt('bank-note/test.csv', delimiter=",")


# In[3]:


X_train = data[:,:-1]
X_test = test[:,:-1]

y_train = data[:,-1].astype(int).reshape(-1,1)
y_test = test[:,-1].astype(int).reshape(-1,1)


# In[4]:


print("Initialize all the weights at random")


# In[5]:


for l in [5,10,25,100]:
    model = NN([l, l], std = 1)
    model.fit(X_train[:,:], y_train[:,:], gamma_0 = 0.01, d = .01, n_epoch= 5, reg = .0, batch_size = 1)
    train_error = 1 - np.sum(model.predict(X_train).reshape(-1,1) == y_train)/X_train.shape[0]
    test_error = 1 - np.sum(model.predict(X_test).reshape(-1,1) == y_test)/X_test.shape[0]
    print('for width = {}, train error = {} and test error = {}'.format(l, train_error, test_error))


# In[6]:


print("Initialize all the weights with 0")


# In[7]:


for l in [5, 10, 25, 100]:
    model = NN([l, l], std = 0)
    model.fit(X_train[:,:], y_train[:,:], gamma_0 = 0.01, d = 1, n_epoch= 5, reg = 0, batch_size = 10)
    train_error = 1 - np.sum(model.predict(X_train).reshape(-1,1) == y_train)/X_train.shape[0]
    test_error = 1 - np.sum(model.predict(X_test).reshape(-1,1) == y_test)/X_test.shape[0]
    print('for width = {}, train error = {} and test error = {}'.format(l, train_error, test_error))


# In[ ]:




