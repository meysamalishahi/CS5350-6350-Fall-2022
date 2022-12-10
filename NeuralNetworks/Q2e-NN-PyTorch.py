#!/usr/bin/env python
# coding: utf-8

# Implementation using PyTorch 
# --------

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict


# In[2]:


import pandas as pd
import matplotlib.pyplot as plt


# In[3]:


def load_data():
    
    train = []
    train_labels = []
    with open("bank-note/train.csv", "r") as f:
        for line in f:
            item = line.strip().split(",")
            train.append(item[:-1])
            train_labels.append(int(item[-1]))
            
    test = []
    test_labels = []
    with open("bank-note/test.csv", "r") as f:
        for line in f:
            item = line.strip().split(",")
            test.append(item[:-1])
            test_labels.append(int(item[-1]))
            
    return np.asarray(train, dtype= float), np.asarray(train_labels, dtype= int), np.asarray(test, dtype= float), np.asarray(test_labels, dtype= int)


# In[4]:


X_train, y_train, X_test, y_test =  load_data()


# In[5]:


X_tr = torch.from_numpy(X_train).type(torch.float32) 
y_tr = torch.from_numpy(y_train).type(torch.int64)
X_te = torch.from_numpy(X_test).type(torch.float32)
y_te = torch.from_numpy(y_test).type(torch.float64)


# In[6]:


class MyNN(nn.Module):
    def __init__(self, depth, input_size, hidden_size, n_classes, act, init = nn.init.xavier_normal_):
        super().__init__()
        self.layers = []
        
        self.layers.append(nn.Linear(in_features= input_size, out_features= hidden_size))
        init(self.layers[-1].weight)
        
        
        for _ in range(depth-2):
            self.layers.append(nn.Linear(in_features= hidden_size, out_features= hidden_size))
            init(self.layers[-1].weight)
            
        self.layers.append(nn.Linear(in_features=hidden_size, out_features = n_classes))
        init(self.layers[-1].weight)           
        
        self.act = act
        
        self.fc = nn.ModuleList(self.layers)
    
    def forward(self, x):
        for i in range(len(self.fc)-1):
            x = self.act(self.fc[i](x))
        x = self.fc[-1](x)
        return x


# In[7]:


print('"tanh" as activation function with "Xavier" initialization')


# In[8]:


for depth in [3,5,9]:
       print('depth = {}'.format(depth))
       for width in [5,10,25,50, 100]:
           model = MyNN(depth, 4, width, 2, torch.tanh)
           criterion = nn.CrossEntropyLoss()
           optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
           for i in range(50):
               y_hat = model(X_tr)
               loss = criterion(y_hat, y_tr)

               optimizer.zero_grad()
               loss.backward()
               optimizer.step()
               
           y_hat = model.forward(X_tr)
           e_1 = 1- (y_hat.max(axis = 1)[1] == y_tr).sum()/X_tr.shape[0]
           
           y_hat = model.forward(X_te)
           e_2 = 1- (y_hat.max(axis = 1)[1] == y_te).sum()/X_te.shape[0]
           print('for width = {}: train error = {} and test error = {}'.format(width, 
                                                                                              e_1, e_2))
       print('\n')
   


# In[9]:


print('"ReLU" as activation function with "he" initialization')


# In[11]:


for depth in [3,5,9]:
        print('depth = {}'.format(depth))
        for width in [5,10,25,50, 100]:
            model = MyNN(depth, 4, width, 2, nn.ReLU(), init = nn.init.kaiming_normal_)

            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)
            for i in range(100):
                y_hat = model(X_tr)
                loss = criterion(y_hat, y_tr)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
            y_hat = model.forward(X_tr)
            e_1 = 1- (y_hat.max(axis = 1)[1] == y_tr).sum()/X_tr.shape[0]
            
            y_hat = model.forward(X_te)
            e_2 = 1- (y_hat.max(axis = 1)[1] == y_te).sum()/X_te.shape[0]
            print('width = {}: train error = {} and test error = {}'.format(width, 
                                                                                               e_1, e_2))
        print('\n')


# In[ ]:




