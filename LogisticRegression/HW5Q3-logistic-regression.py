#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# # Data loading and cleaning

# In[2]:


data_t = np.genfromtxt("bank-note/train.csv", dtype=float, delimiter=',')
data_t = np.hstack((np.ones((data_t.shape[0],1)), data_t))
# data_t.shape


# In[3]:


data_train = data_t[:,:-1]
label_train = data_t[:,-1].astype(int)
# data_train.shape, label_train.shape


# In[4]:


data_test = np.genfromtxt("bank-note/test.csv", dtype=float, delimiter=',')
data_test = np.hstack((np.ones((data_test.shape[0],1)), data_test))
# data_test.shape


# In[5]:


test = data_test[:,:-1]
label = data_test[:,-1].astype(int)
# test.shape, label.shape


# # Defining functions 

# In[6]:


def sigmoid(z):
    if z < 0:
        return np.exp(z)/(1+np.exp(z))
    return 1/(1+np.exp(-z))
    
def sig(z):   
    mask = (z>=0).astype(int)
    return mask/(1+np.exp(-mask*z)) + (1-mask) *np.exp((1-mask)*z)/(1+np.exp((1-mask)*z))

def grad_neg_log_likelihood(w, x, l):
    s = np.dot(w,x)
    y = 2*l - 1 
    return -y*sigmoid(-y*s) * x

def accuracy(w, data, label):
    y_hat = (sig(data@w) >= 0.5).astype(int)
    return 1 - (y_hat == label).mean()


# In[7]:


lr = lambda gamma_0, d, t: gamma_0/(1+gamma_0 * t/d)


# In[8]:


def train(data, label, gamma_0, d, n_epoch, reg):
    n_samples, n_features = data.shape
    w = np.random.normal(loc = 0, scale = 1, size = (n_features,))
    L = list(range(n_samples))
    for i in range(n_epoch):
        L = np.random.permutation(n_samples)
        X = data[L,:]
        y = label[L]
        for j in range(n_samples):
            w = w - lr(gamma_0, d, i)*reg* w - n_samples*lr(gamma_0, d, i) * grad_neg_log_likelihood(w, X[j,:], y[j])
    return w 


# In[9]:


w_map = train(data_train, label_train, gamma_0 = .1, d = .1, n_epoch = 10, reg =1)


# In[10]:


accuracy(w_map, data_train, label_train), accuracy(w_map, test, label)


# In[11]:


print("Using the MAP estimation with prior variance from {0.01, 0.1, 0.5, 1, 3, 5, 10, 100}")


# In[12]:


L = ['0.01', '0.1', '0.5', '1', '3', '5', '10', '100']
LL = [0.01, 0.1, 0.5, 1, 3, 5, 10, 100]
for i in range(8):
    var = LL[i]
    w_map = train(data_train, label_train, gamma_0 = .001, d = .001, n_epoch = 100, reg =2/var)
    train_error, test_error = accuracy(w_map, data_train, label_train), accuracy(w_map, test, label)
    print('For variance = {}, train error = {} and test error = {}'.format(var, train_error, test_error))


# In[13]:


print("Using maximum likelihood (ML) estimation")


# In[14]:


w_ML = train(data_train, label_train, gamma_0 = .01, d = .01, n_epoch = 10, reg =0)
train_error, test_error = accuracy(w_map, data_train, label_train), accuracy(w_map, test, label)
print('train error = {} and test error = {}'.format(train_error, test_error))

