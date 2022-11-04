#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from random import shuffle
import time


# In[2]:


# txtfile = open('bank-note/data-desc', 'r')
# print(txtfile.read())


# In[3]:


train = np.genfromtxt('bank-note/train.csv', delimiter=",")
test = np.genfromtxt('bank-note/test.csv', delimiter=",")
train_label = 2*train[:,-1] - 1
test_label = 2*test[:,-1]-1
train[:,-1] = np.ones(train.shape[0])
test[:,-1] = np.ones(test.shape[0])


# In[4]:


print(train.shape, test.shape)


# In[5]:


class Perceptron:
    def __init__(self, X, y, n_epoch = 10, Shuffle = True):
        self.w = np.zeros(X.shape[1])
        self.W = [self.w.copy()]
        self.C = [1]
        
        self.train(X, y, n_epoch=n_epoch, Shuffle=Shuffle)
        
        
    def train(self,X, y, n_epoch, Shuffle):
        N = X.shape[0]
        
        
        for i in range(n_epoch):
            if Shuffle:
                I = list(range(N))
                shuffle(I)
                X_ = X[I,:]
                y_ = y[I]
            else:
                X_ = X
                y_ = y
            for j in range(N):
                if y_[j] * np.dot(X_[j], self.w) <= 0:
                    self.w += y_[j] * X_[j]
                    self.W.append(self.w.copy())
                    self.C.append(1)
                else:
                    self.C[-1] += 1

    def pred(self,data, type_Perceptron = 'standard'):
        N = data.shape[0]
        
        if type_Perceptron == 'standard':
            return 2 * (data @ self.W[-1] >= 0) - 1
        
        if type_Perceptron == 'voted': 
            W = np.array(self.W).T
            C = np.array(self.C)
            return 2*(((2*(data@W >=0)-1)@C)>=0)-1
        
        if type_Perceptron == 'average':
            W = np.array(self.W).T
            C = np.array(self.C)
            return 2*((data@W@C)>=0)-1
        
        
        
    def error(self,data, label, type_Perceptron = 'standard'):
        y_hat = self.pred(data, type_Perceptron = type_Perceptron)
        return (y_hat == label).mean()


# ### Standard Perceptron. 

# In[6]:


print('Standard Perceptron:')


# In[7]:


model = Perceptron(train, train_label)


# In[8]:


print('Train error for standard perceptron: {}'.format(1-model.error(train, train_label)))


# In[9]:


print('Test error for standard perceptron: {}'.format(1-model.error(test, test_label)))


# In[10]:


print('learned weight vector: {}'.format(model.W[-1]))


# ### Voted Perceptron

# In[11]:


print('Voted Perceptron')


# In[12]:


print('Train error for voted perceptron: {}'.format(1-model.error(train, train_label, 
                                                                     type_Perceptron = 'voted')))


# In[13]:


print('Test error for voted perceptron: {}'.format(1-model.error(train, train_label, 
                                                                     type_Perceptron = 'voted')))


# In[14]:


W_voted = np.array(model.W)
C = np.array(model.C)


# In[15]:


print('learned weight vectors:\n {}'.format(W_voted[[1,2,3,4,5,-1,-2,-3,-4,-5]]))


# In[16]:


print('counts:\n {}'.format(C[[1,2,3,4,5,-1,-2,-3,-4,-5]]))


# ### Average Perceptron

# In[17]:


print('Average Perceptron')


# In[18]:


print('Train error for standard perceptron: {}'.format(1-model.error(train, train_label, 
                                                                     type_Perceptron = 'average')))


# In[19]:


print('Test error for standard perceptron: {}'.format(1-model.error(test, test_label, 
                                                                     type_Perceptron = 'average')))


# In[20]:


W_average = C @ W_voted/np.sum(C)


# In[21]:


print('learned weight vectors:\n {}'.format(W_average))


# In[ ]:




