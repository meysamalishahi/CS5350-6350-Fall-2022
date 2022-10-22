
# coding: utf-8

# In[22]:


def load_bank():
    
    train = []
    train_labels = []
    with open("bank-1/train.csv", "r") as f:
        for line in f:
            item = line.strip().split(",")
            train.append(item[:-1])
            train_labels.append(item[-1])

    test = []
    test_labels = []
    with open("bank-1/test.csv", "r") as f:
        for line in f:
            item = line.strip().split(",")
            test.append(item[:-1])
            test_labels.append(item[-1])

    for attr in [0, 5, 9, 11, 12, 13, 14]:
        convert_numeric(train, test, attr)


    return train, train_labels, test, test_labels

import random
def my_sample(train, train_labels, n_samples = None):
    if n_samples == None:
        n_samples = len(train_labels)
    s_train = []
    s_lable = []
    for i in range(0, n_samples):
        x = random.randint(0, n_samples-1)
        s_train.append(train[x])
        s_lable.append(train_labels[x])

    return s_train, s_lable


# In[23]:


import statistics as st

def convert_numeric(train, test, attr):
    temp = [float(x[attr]) for x in train]
    temp.sort()
    
    if len(temp)// 2 == 0:
        median = (temp[int(len(temp)/2)] + temp[int(len(temp)/2)+1])/2
    else:
        median = temp[int(len(temp)/2)]

    #median = st.median(temp)                   
                     
    #print(median)
                       
    for x in train:
        x[attr] = True if float(x[attr]) > median else False

    for x in test:
        x[attr] = True if float(x[attr]) > median else False


# In[24]:


# return the value with majority and also the number of values in lables

def Majority(labels, weights = None):
    
    
    if weights == None:
        L = len(labels)
        weights = [1]*L
    
    W = {}
    for x in range(len(labels)):
        
        if labels[x] not in W:
            W[labels[x]] = 0
            
        W[labels[x]] += weights[x]
    
    Max = -1
    majority = None    
    for y in W:
        if W[y] > Max:
            Max = W[y]
            majority = y
    #print(W)        
    return(majority, len(W))


# In[33]:



from math import log2
def entropy(labels, weights = None):
    
    n = len(labels)
    if weights == None:
        weights = [1]*n
            
    W = {}
    Sum = 0
    for i in range(n):
        if labels[i] not in W:
            W[labels[i]] = 0
        
        W[labels[i]] += weights[i]
            
        Sum += weights[i]
        
    S = 0
    for x in W:
        S += (W[x]/Sum) * log2(Sum / W[x])

    return S


# In[35]:
  

def Entropy_given_attribute(train, labels, attribute, weights = None):
    
    n = len(labels)
    if weights == None:
        weights = [1]*n
    
    
    split_l = {}
    split_w = {}
    sum_weights = sum(weights)
    
    for x in range(n):
        
        txa = train[x][attribute]
        if txa not in split_w:
            
            split_w[txa] =[]
            split_l[txa] = []
            
        split_w[txa].append(weights[x])
        split_l[txa].append(labels[x])  
        
    En = 0        
    for x in split_w:
        
        En += sum(split_w[x]) * entropy(split_l[x], split_w[x]) / sum_weights
        
    return(En, list(split_w.keys()))


# In[36]:


def old_best_att(train, labels, attributes, weights = None):
    
    
    lable_Ent = entropy(labels, weights)
    Max = -1
    Best = None
    Best_values = None
    
    for attribute in attributes: 
        temp, temp_values = Entropy_given_attribute(train, labels, attribute, weights) 
        if lable_Ent - temp >  Max:
            Max = lable_Ent - temp
            Best = attribute
            Best_values = temp_values
                    
    return(Best, Best_values)


# In[37]:


def split(train, label, attribute, weights = None):
    
    n = len(label)
    if weights == None:
        weights = [1]*n
    
    split_w = {}
    split_t = {}
    split_l = {}
    
    for x in range(len(label)):
        
        #print('x = ', x)
        #print('attribute = ', attribute)
        txa = train[x][attribute]
        if txa not in split_t:
            
            split_w[txa] = []
            split_t[txa] = []
            split_l[txa] = []
            
        split_w[txa].append(weights[x])
        split_t[txa].append(train[x])
        split_l[txa].append(label[x])
        
    return (split_t, split_l, split_w)
        

    
    
    
def error(dt, x, y):
    count = 0
    for i in range(len(x)):
        xi = x[i]
        yi = dt.predict(xi)
        if yi != y[i]:
            count += 1

    return count / len(x)    
    
#######################################################
#######################################################


class DecisionTree(object):
    def __init__(self, train, labels, attributes, depth = -1, weights = None):
        
        #if weights == None:
            #L = len(labels)
            #weights =[1]* L
        
        self.leaf = False 
        self.label, n_values = Majority(labels, weights) 
        
        if len(attributes) == 0 or n_values == 1 or depth == 0:
            
            self.leaf = True  
            return
        
        self.att_split, values = self.best_att(train, labels, attributes, weights)
        #print(self.att_split)
        
        train_s, lables_s, weight_s = split(train, labels, self.att_split, weights) #returns splited train, labels, weights as dicts
        
        self.Tree = {}
        
        attributes.remove(self.att_split)
        
        for v in train_s: # train_s is a dict whose keys are the values in column self.att_split
               
            self.Tree[v] = DecisionTree(train_s[v], lables_s[v], attributes, depth - 1, weight_s[v])

        attributes.append(self.att_split)
            
    
    def predict(self, instance):
        
        if self.leaf:
            return self.label
        
        if instance[self.att_split] in self.Tree:
            return self.Tree[instance[self.att_split]].predict(instance)   
        
        return self.label   
    
    
    def best_att(self, train, labels, attributes, weights):
        
        lable_Ent = entropy(labels, weights)
        Max = -1
        Best = None
        Best_values = None
    
        for attribute in attributes: 
            temp, temp_values = Entropy_given_attribute(train, labels, attribute, weights) 
            if lable_Ent - temp >  Max:
                Max = lable_Ent - temp
                Best = attribute
                Best_values = temp_values
                    
        return(Best, Best_values)