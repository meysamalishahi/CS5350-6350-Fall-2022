# Functions


from numpy import log2, log, sqrt

def Gini(Ls, WW = None):
    
    
    if WW == None:
        L = len(Ls)
        WW = [1]*L
    
    W = {}
    Sum = 0
    for x in range(len(Ls)):
        
        if Ls[x] not in W:
            W[Ls[x]] = 0
            
        W[Ls[x]] += WW[x]
        Sum += WW[x]
    
    S = 0
    for x in W:
        S += (W[x]/Sum)**2
        
    return 1 - S

#################################################################


def majority(Ls, WW = None):
    
    
  
    if WW == None:
        L = len(Ls)
        WW = [1]*L
    
    W = {}
    for x in range(len(Ls)):
        
        if Ls[x] not in W:
            W[Ls[x]] = 0
            
        W[Ls[x]] += WW[x]
    
    Max = -1
    majority = None    
    for y in W:
        if W[y] > Max:
            Max = W[y]
            majority = y        
    return 1 - Max/sum(WW)

#################################################################

def entropy(Ls, WW = None):
    
    n = len(Ls)
    if WW == None:
        WW = [1]*n
            
    W = {}
    Sum = 0
    for x in range(n):
        if Ls[x] not in W:
            W[Ls[x]] = 0
        
        W[Ls[x]] += WW[x]
            
        Sum += WW[x]
        
    S = 0
    for x in W:
        S += (W[x]/Sum) * log2(Sum / W[x])

    return S


#################################################################

def Majority(Ls, WW = None, ignor = None):
    
    
    if WW == None:
        L = len(Ls)
        WW = [1]*L
    
    W = {}
    for x in range(len(Ls)):
        
        if Ls[x] not in W:
            W[Ls[x]] = 0
        
        if ignor is None:
            W[Ls[x]] += WW[x]
        else:
            if Ls[x]!= ignor:
                W[Ls[x]] += WW[x]
                
    
    Max = -1
    majority = None    
    for y in W:
        if W[y] > Max:
            Max = W[y]
            majority = y        
    return(majority, len(W))



#################################################################


def Entropy_given_attribute(T, L, a, W = None, Entropy_function = entropy):
    
    n = len(L)
    if W == None:
        W = [1]*n
    
    
    split_l = {}
    split_w = {}
    sum_W = sum(W)
    
    for x in range(n):
        u = T[x][a]
        if u not in split_w:
            
            split_w[u] =[]
            split_l[u] = []
            
        split_w[u].append(W[x])
        split_l[u].append(L[x])  
        
    En = 0        
    for x in split_w:
        
        En += sum(split_w[x]) * Entropy_function(split_l[x], split_w[x]) / sum_W
        
    return(En, list(split_w.keys()))    


#################################################################


from random import sample
class DT(object):
    def __init__(self, train, L, attss, depth = -1, WW = None, 
                 Entropy_function = entropy, randomness = None):
        self.randomness = randomness
        self.Entropy_function = Entropy_function
        self.leaf = False 
        self.label, n_values = Majority(L, WW) 
        
        if len(attss) == 0 or n_values == 1 or depth == 0:
            
            self.leaf = True  
            return
        
        self.at_splt, values = self.best_att(train, L, attss, WW)
        
        train_s, lables_s, weight_s = split(train, L, self.at_splt, WW) 
        self.Tree = {}
        
        attss.remove(self.at_splt)
        
        for v in train_s:
               
            self.Tree[v] = DT(train_s[v], lables_s[v], attss, depth - 1, 
                                        weight_s[v], Entropy_function)

        attss.append(self.at_splt)
            
    
    def predict(self, instance):
        
        if self.leaf:
            return self.label
        
        if instance[self.at_splt] in self.Tree:
            return self.Tree[instance[self.at_splt]].predict(instance)   
        
        return self.label   
    
    
    def best_att(self, train, L, attss, WW):
        
        if self.randomness:
            if len(attss)>= 2:
                k = self.randomness
            else: k = 1
            samples = sample(attss, k)
            
        else: samples = attss.copy()
        
        lable_Ent = entropy(L, WW)
        Max = -1
        Best = None
        Best_values = None
    
        for attribute in samples: 
            temp, temp_values = Entropy_given_attribute(train, L, 
                                                        attribute, WW, self.Entropy_function) 
            if lable_Ent - temp >  Max:
                Max = lable_Ent - temp
                Best = attribute
                Best_values = temp_values
                    
        return(Best, Best_values)

    
    
#################################################################    
    
def split(train, label, attribute, WW = None):
    
    n = len(label)
    if WW == None:
        WW = [1]*n
    
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
            
        split_w[txa].append(WW[x])
        split_t[txa].append(train[x])
        split_l[txa].append(label[x])
        
    return (split_t, split_l, split_w)


###################################################################
###################################################################


class Adaboost:
    
    def __init__(self, train, labels, n_iters, depth):
        
        
        attss = [i for i in range(len(train[0]))]
        self.Trees = []
        self.Coeffs = []   
        self.W_ = [] 
        self.Eps = []
        
        self.train(train, labels, n_iters, depth)
        
#         model = None
#         W_  = None
#         for _ in range(n_iters):
        
#             self.update_weights(train, labels, model)
#             model = DT(train, labels, attss = attss , depth = depth, WW = self.W_)
#             self.Trees.append(model)
   

        
        
        
    def train(self, train, labels, n_iters, depth):
        print('starting adaboost for 500 iterartions')
        model = None
        W_  = None
        for _ in range(n_iters):
        
            self.update_weights(train, labels, model)
            model = DT(train, labels, attss = [i for i in range(len(train[0]))], 
                       depth = depth, WW = self.W_, randomness = None)
            self.Trees.append(model)
    
    
    def update_weights(self, train, labels, model):
    
        n = len(labels)

        if model is None:
            self.W_ = [1/n]*n
            self.Coeffs.append(1)
            return 
    
        eps, preds = Error(train, labels, model, self.W_)
        
        self.Eps.append(eps)

        
        alpha = log((1-eps)/eps)/2
        
        
        self.Coeffs.append(alpha)
    
        for i in range(n):
            if preds[i]:
                self.W_[i] *= (sqrt(eps/(1-eps)))
            else:
                self.W_[i] *= (sqrt((1-eps)/eps))
    
        S = sum(self.W_)
        for i in range(n): 
            self.W_[i] /= S
        
    
     
    def adapredict(self, x, n = None):
        
        if n is None:
            n = self.n_iters 
            
        Sum = 0
        for i in range(n):
            Sum += self.Coeffs[i] * self.Trees[i].predict(x) 
        
        if Sum >= 0:
            return (1)
        return (-1) 
            
            

    def Error(self, x, y, n):
        
        Errors = [0] * n
        
        
        for i in range(len(y)):
            Sum = 0
            
            for j in range(n):
                u = self.Trees[j].predict(x[i])
                Sum += self.Coeffs[j] *  u
                
                if (Sum >= 0 and y[i] == -1) or (Sum < 0 and y[i] == 1):
                    Errors[j] += 1/len(y)
                    
        return (Errors)

def Error(x, y, model, weights = None):
    
    n = len(x)
    preds = [True] * n
    p = 0
    
    if weights is None:
        weights = [1/n]*n
        
        
    S = sum(weights)

    for i in range(n): 
        y_hat = model.predict(x[i])
        if y_hat != y[i]:
            p += weights[i]/S
            preds[i] = False

    return (p, preds) 
