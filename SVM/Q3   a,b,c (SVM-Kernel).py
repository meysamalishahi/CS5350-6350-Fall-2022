#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from scipy.optimize import minimize, LinearConstraint, Bounds
import time
from termcolor import colored


# In[2]:


D = open ('bank-note//data-desc', 'r').read()
print(D)


# In[3]:


X_t = np.genfromtxt('bank-note/train.csv', delimiter=",")
X_test = np.genfromtxt('bank-note/test.csv', delimiter=",")
y_t = (2*X_t[:,-1] - 1).reshape(-1,1)
y_test = (2*X_test[:,-1]-1).reshape(-1,1)
# X_t[:,-1] = np.ones(train.shape[0])
# X_test[:,-1] = np.ones(test.shape[0])


# In[4]:


X_t = X_t[:,:-1]
X_test = X_test[:,:-1]


# In[5]:


Z = X_t[:, :]
y = y_t[:, :]
C_list = [100/873,500/873, 700/873]


# In[6]:


T = (Z*y) @ (Z*y).T
f = lambda x: .5 * x.T @ T @ x - np.sum(x)
eq_cons = {'type': 'eq',
            'fun' : lambda x: x @ y,
            'jac' : lambda x:  y.T}


# In[7]:


res = []
x0 = np.zeros_like(y)
for C in C_list:
    start = time.time()
    bounds = Bounds(np.zeros(y.shape[0]), C * np.ones(y.shape[0]))
    res.append(minimize(f, x0, method='SLSQP',
               constraints=[eq_cons], options={'ftol': 1e-5, 'disp': True},
                bounds=bounds))
    print(time.time() - start)


# In[8]:


# I = res[0].x>1e-5
# w_star = (res[0].x[I]*y_t[I].reshape(-1,))@ X_t[I]
# w_star


# In[9]:


# J = (res[0].x>1e-5)*(res[0].x<C_list[0] - 1e-5)
# (y_t[J] - X_t[J]@w_star).mean()


# ### First, run your dual SVM learning algorithm with   $C$ in $\{\frac{100}{873}, \frac{500}{873}, \frac{700}{873}\}$. Recover the feature weights $\mathbf{w}$ and the bias $b$. Compare with the parameters learned with stochastic sub-gradient descent in the primal domain (in Problem 2) and the same settings of $C$, what can you observe? What do you conclude and why? Note that if your code calculates the objective function with a double loop, the optimization can be quite slow. To accelerate, consider writing down the objective in terms of the matrix and vector operations, and treat the Lagrange multipliers that we want to optimize as a vector! Recall, we have discussed about it in our class. 

# In[10]:


C = ['100/873','500/873', '700/873']
for j in range(3):
    print('C = {}'.format(C[j]))
    I = res[0].x > 1e-5
    w_star = (res[j].x[I]*y_t[I].reshape(-1,))@ X_t[I]
    J = (res[j].x>1e-8)*(res[j].x<C_list[j] - 1e-2)
    print('w_star = {}'.format(w_star))
#     print('b_star = {}'.format((y_t[J] - X_t[J]@w_star).mean()))
    print('\n')


# In[11]:


def predict(X, alps, Vecs, lbls, C1):
    
    beta = list(np.where(C1 - alps  > 1e-5)[0])
#     print(len(beta))
    b = (sum(lbls[beta]) - np.sum(Vecs[beta,:] @ (alps * lbls * Vecs).T))/len(beta)
    print('b_star = {}'.format(b))

    return np.where(np.sum(X @ (alps * lbls * Vecs).T, 1) + b  > 0 ,1, -1)


# In[12]:


for i in range(3):
    print ('C = {}'.format(C[i]))
    Sup_inds = list(np.where(res[i].x > 1e-5)[0])
    print('numebr of support vectors = {}'.format(len(Sup_inds)))
    Sup_alps = res[i].x[Sup_inds].reshape(-1,1)
    Sup_Vecs = Z[Sup_inds,:]
    Sup_lbls = y[Sup_inds]
    P_t = predict(X_t, Sup_alps, Sup_Vecs, Sup_lbls, C_list[i]).reshape(-1,1)
    print('Train Error for C ={} is {}'.format(C[i],  np.sum(P_t * y_t < 0)/y_t.shape[0]))
    P = predict(X_test, Sup_alps, Sup_Vecs, Sup_lbls, C_list[i]).reshape(-1,1)
    print('Test Error for C ={} is {}'.format(C[i],  np.sum(P * y_test < 0)/y_test.shape[0]))
    print('\n')


# # Applying Kernel Trick "Gaussian kernel"

# In[13]:


Z2 = X_t[:, :]
y2 = y_t[:, :]
C_list = [100/873,500/873, 700/873]
gamma_list = [0.1, 0.5, 1, 5, 100]


# In[14]:


def G_k(A, B, gamma):  #Resturn a matrix whose ijth entry is exp{-||A_i-B_j||**2/gamma}
    temp = np.sum(A * A, 1).reshape(A.shape[0], 1) + np.sum(B * B, 1).reshape(1, B.shape[0]) - 2 * A @ B.T
    return np.exp(-temp/gamma)


# In[15]:


res_K = {}
x0 = np.zeros_like(y2)

for i in range(len(gamma_list)):

    
    
    K1 = y2 * G_k(Z2, Z2, gamma_list[i]) * y2.T
    print(K1.shape)
    f = lambda x: .5 * x.T @ K1 @ x - np.sum(x)
    eq_cons = {'type': 'eq',
                'fun' : lambda x: x @ y2,
                'jac' : lambda x:  y2.T}
    
    for j in range(len(C_list)):
        
        start = time.time()
        bounds = Bounds(np.zeros(y2.shape[0]), C_list[j] * np.ones(y2.shape[0]))
        res_K[(j, i)] = minimize(f, x0, method='SLSQP',
                   constraints=[eq_cons], options={'ftol': 1e-9, 'disp': True},
                    bounds=bounds)
        
        print('spent time for (C, gamma) = {} is'.format((C_list[j], gamma_list[i])), time.time() - start)
        print('---------------------------------------------------------')


# In[16]:


C_l_str = ['100/873', '500/873', '700/873']
gamma_l_str = ['0.1', '0.5', '1', '5', '100']


# In[17]:


Beta = {}


# In[18]:


def predict_K(X, alps, Vecs, lbls, gamma, C, i, j):
    
    beta = list(np.where(alps < C - 1e-9)[0])   
    Beta[(j,i)] = beta
#     print('numebr of support vectors = {}'.format(len(beta)))
        
    if len(beta)>0:
        
        k = np.argmin(alps[beta])
        
        X_j = Vecs[beta[k]].reshape(1,-1)
        H = Vecs

        N = np.sum(X_j * X_j, 1).reshape(-1,1) + np.sum(H * H, 1).reshape(-1,1) - 2 * H @ X_j.T
        M = lbls * alps * np.exp(- N/gamma)
        b = lbls[beta[k]] - np.sum(M)
        
#         print('b=', b)
    else:
        b =0
#         print('b=', 0)
        
    temp2 = G_k(X , Vecs, gamma) * (alps * lbls).T

    return np.where(np.sum(temp2, 1) + b > 0 , 1, -1) , np.where(np.sum(temp2, 1) > 0 , 1, -1)


# In[19]:


for j in range(len(C_list)):
    for i in range(len(gamma_list)):
        
        print ("$-------------------------------$\\\ ")
        print('$C = {}, \gamma = {}$\\\\'.format(C_l_str[j], gamma_l_str[i]))

    
#         Sup_inds = list(np.where(res_K[(j,i)].x > 1e-9)[0])
        Sup_inds = (res_K[(j,i)].x > 1e-9)
        print('\#support vectors = {}\\\\'.format(Sup_inds.sum()))
#         print('#support vectors = {}'.format(len(Sup_inds)))
        
        Sup_alps = res_K[(j,i)].x[Sup_inds].reshape(-1,1)
        Sup_Vecs = Z2[Sup_inds, :]
        Sup_lbls = y2[Sup_inds, :]
        P = predict_K(X_test, Sup_alps, Sup_Vecs, Sup_lbls, gamma_list[i], C_list[j], i, j)
        P_0 = P[0].reshape(-1,1)
#         P_1 = P[1].reshape(-1,1)
#         print('(i,j) =', (j,i))
        print('Test Error for $(C, \gamma) ={}$ is {}\\\\'.format((C_l_str[j], gamma_l_str[i]), 
                                                            np.sum(P_0 * y_test < 0)/y_test.shape[0]))
        
        
        P = predict_K(X_t, Sup_alps, Sup_Vecs, Sup_lbls, gamma_list[i], C_list[j], i, j)
        P_0 = P[0].reshape(-1,1)
#         P_1 = P[1].reshape(-1,1)
        print('Train Error for $(C, \gamma) ={}$ is {}\\\\'.format((C_l_str[j], gamma_l_str[i]), 
                                                            np.sum(P_0 * y_t < 0)/y_t.shape[0]))
        
#         print(colored('With no b = ', 'red'),  
#               colored(100 * np.sum(P_1 * y_test < 0)/y_test.shape[0], 'red'))
        if i < len(gamma_list)-1:
            temp = (res_K[(j,i)].x > 1e-9) * (res_K[(j,i+1)].x > 1e-9)
#             print(temp)
            print('\#overlapped support vectors between values of $\gamma_{}, \gamma_{} = {}$\\\\'.format(
                i , i+1, temp.sum()))
        #print(colored('Clusters for are:', 'blue'))
#         print ("----------------------------------------------------------------------------")


# In[ ]:




