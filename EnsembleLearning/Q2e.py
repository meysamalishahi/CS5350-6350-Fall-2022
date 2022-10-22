#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from AllFunctions import *
from numpy import log2, log, sqrt
import matplotlib.pyplot as plt
import random
from random import sample
import copy


# In[2]:


# txtfile = open('bank/data-desc.txt', 'r')
# print(txtfile.read())


# In[3]:


C = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 
 'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'y']
types = ['numeric', 'categorical', 'categorical', 'categorical', 'binary', 'numeric', 
                      'binary', 'binary', 'categorical', 'numeric', 'categorical', 'numeric', 
                      'numeric', 'numeric', 'numeric', 'categorical', 'binary']
dic= dict(zip(C, types))


# In[4]:


train = pd.read_csv('bank/train.csv', names = C)
test = pd.read_csv('bank/test.csv', names = C)
#train.head()


# In[5]:


median_dict = {}
Train_new =pd.DataFrame()
Test_new =pd.DataFrame()
for name in C:
    if dic[name] == 'numeric':
        M = train[name].median()
        median_dict[name] = M
        Train_new[name+ '>' + str(M)] = np.where(train[name]  > M, 'yes', 'no')
        Test_new[name+ '>' + str(M)] = np.where(test[name]  > M, 'yes', 'no')
    else:
        Train_new[name] = train[name]
        Test_new[name] = test[name]


# In[6]:


Train = []
Label = []
for i in range(len(Train_new)):
    temp = list(Train_new.loc[i])
    Train.append(temp[:-1])
    Label.append(temp[-1])


# In[7]:


Test = []
Test_Label = []
for i in range(len(Test_new)):
    temp = list(Test_new.loc[i])
    Test.append(temp[:-1])
    Test_Label.append(temp[-1])


# In[8]:


for i in range(len(Label)):
    if Label[i] == 'yes':
        Label[i] = 1
    else:
        Label[i] = -1


# In[9]:


for i in range(len(Test_Label)):
    if Test_Label[i] == 'yes':
        Test_Label[i] = 1
    else:
        Test_Label[i] = -1


# In[10]:


atts = list(range(0,len(C)-1))


# In[12]:


# atts


# In[11]:


def sample_(X, Y, n_samples = None):
    n = len(X)
    if n_samples == None:
        n_samples = n
        
    s_t = []
    s_l = []
    for _ in range(0, n_samples):
        i = random.randint(0, n-1)
        s_t.append(X[i])
        s_l.append(Y[i])

    return s_t, s_l


# ### Bias-Variance

# In[12]:


n_bagges = 100
n_trees = 50


# In[13]:


Model_histroy = []
for i in range(n_bagges):

    F = []
    for _ in range(n_trees):
        
        X, Y = sample_(Train, Label)
        model = DT(X, Y, attss = [i for i in range(len(Train[0]))], depth = -1, 
                          randomness = 2)
        F.append(copy.copy(model))
    Model_histroy.append(F)


# In[14]:


def single_bias_var(x, y, Forest):
    n = len(Forest)
    h_star_x = np.zeros(n)
    for i in range(n):
        model = Forest[i]
        h_star_x[i] = model.predict(x)
    return (y - h_star_x.mean())**2, h_star_x.var()  


# In[15]:


Forest = [Model_histroy[i][0] for i in range(n_bagges)]


# In[23]:


#print(np.array([np.array([Forest[i].predict(Test[k]) for i in range(n_bagges)]).var() for k in range(1)]).mean())


# In[17]:


bias_1 = np.array([single_bias_var(Test[i], Test_Label[i], Forest)[0] for i in range(len(Test))]).mean()
var_1 = np.array([single_bias_var(Test[i], Test_Label[i], Forest)[1] for i in range(len(Test))]).mean()
print(bias_1, var_1)

general_squared_error = bias_1 + var_1
print(general_squared_error)


# In[24]:


print('bias, variance, and general_squared_error: ', bias_1, var_1, bias_1+ var_1)


# In[18]:


def Bagg_pred(x, bagg):
    y_pred = 0
    for model in bagg:
        y_pred += model.predict(x)
    if y_pred>= 0: return 1
    return -1


# In[19]:


def bias_var(x, y, List_baggs):
    n = len(List_baggs)
    h_star_x = np.zeros(n)
    for i in range(n):
        bagg = List_baggs[i]
        h_star_x[i] = Bagg_pred(x, bagg)
    return (y - h_star_x.mean())**2, h_star_x.var() 


# In[20]:


bias_bagg = np.array([bias_var(Test[i], Test_Label[i], Model_histroy)[0] for i in range(len(Test))]).mean()
var_bagg = np.array([bias_var(Test[i], Test_Label[i], Model_histroy)[1] for i in range(len(Test))]).mean()


# In[21]:


print(bias_bagg, var_bagg)


# In[22]:


print('bias, variance, and general_squared_error: ', bias_bagg, var_bagg, bias_bagg+ var_bagg)


# In[ ]:




