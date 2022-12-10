import numpy as np


def gamma(t, gamma, d):
    return gamma/(1+gamma*t/d)

class NN:
    
    def __init__(self, list_layers, std = 1e-5):
        
        
        self.list_hlayers = list_layers
        self.n_hlayers = len(list_layers)
        
        self.W = {}
        self.dW = {}
        
        self.b = {}
        self.db = {}
        
        self.S = {}
        self.dS = {}
        
        self.Z = {}
        self.dZ = {}
        
        self.std = std
        
    def fit(self, X_train, y_train, n_epoch, reg, gamma_0, d, batch_size = 1):
            
        self.X = X_train
        self.y = y_train
        
        self.N = X_train.shape[0]
        self.m = X_train.shape[1]
        
        
        self.n_calss = len(np.unique(y_train))  
        self.l = lambda t: gamma(t, gamma_0, d)
        
        
        self.list_layers = [self.m] + self.list_hlayers + [self.n_calss]
        
        self.n_layers = len(self.list_layers) 
        
        self.reg = reg
        
        
        
        for i in range(self.n_layers - 1):
            
            self.W[i] = self.std * np.random.randn(self.list_layers[i], self.list_layers[i+1])
            self.b[i] = np.zeros((1, self.list_layers[i+1]))

        num_train = X_train.shape[0] 
        for i in range(n_epoch):
            L = np.random.permutation(num_train)
            X = X_train[L, :] 
            y = y_train[L, :]
            
            for k in range(0, num_train, batch_size):
                

                X_bach = X[k:k+batch_size, :]
                y_bach = y[k:k+batch_size, :]

                self.forward_pass(X_bach, y_bach)

                self.update_grads(y_bach)

                for j in range(self.n_layers - 1):

                    self.W[j] = self.W[j] - (1/batch_size) * self.l(i) * self.dW[j] - 2 * self.l(i) * self.reg * self.W[j]
                    self.b[j] = self.b[j] - (1/batch_size) * self.l(i) * self.db[j]
                
#------------------------------------------------------------------------------------------------            
#------------------------------------------------------------------------------------------------            
#------------------------------------------------------------------------------------------------                  
                 
            
    def forward_pass(self, X, y):
        
        n_batch = X.shape[0]
                
        self.Z[0] = X
        self.S[0] = X
               
        for i in range(self.n_layers - 1):
            
            self.S[i+1] = self.Z[i] @ self.W[i] + self.b[i]

            
            if i != self.n_layers - 2:
                self.Z[i+1] = np.maximum(0, self.S[i+1]) # ReLU
            else:
                self.scores = self.S[i+1]
                  
        temp = self.scores - np.max(self.scores, axis=1, keepdims = True) # avoiding overflow
        exp_scores = np.exp(temp)
        
        
        softmax_matrix = exp_scores/np.sum(exp_scores, axis=1, keepdims = True)
        
        
        temp1 = -np.log(1e-30 + softmax_matrix[np.arange(n_batch), y.reshape(-1,)])
        
        reg_loss = 0
        for i in range(self.n_layers-1):
            reg_loss += np.sum(self.W[i] * self.W[i])
        
        self.softmax_loss = np.sum(temp1)/n_batch + self.reg * reg_loss
        
    
        softmax_matrix[np.arange(n_batch), y.reshape(-1,)] -= 1 
        
        self.dS[self.n_layers-1] = softmax_matrix #gradiant of loss w.r.t scores
        
        return (self.softmax_loss)
                                                
#------------------------------------------------------------------------------------------------           
#------------------------------------------------------------------------------------------------            
#------------------------------------------------------------------------------------------------

    def update_grads(self, y): # back-propagation
                
        n = y.shape[0] #size of batch
        
        for i in range(self.n_layers-2, 0, -1):
            
            self.dW[i] = self.Z[i].T @ self.dS[i+1]
            self.db[i] = np.sum(self.dS[i+1], axis=0, keepdims = True)
            self.dZ[i] = self.dS[i+1] @ self.W[i].T 
            self.dS[i] = self.dZ[i] * (self.Z[i] > 0)
            
            
        self.dW[0] = self.Z[0].T @ self.dS[1]
        self.db[0] = np.sum(self.dS[1], axis=0, keepdims = True)

#------------------------------------------------------------------------------------------------           
#------------------------------------------------------------------------------------------------            
#------------------------------------------------------------------------------------------------        
        
        
    def predict(self, X):
        
        n = X.shape[0]
        test_scores = X        

        
        for i in range(self.n_layers-1):
            
            test_scores = test_scores @ self.W[i] + self.b[i]
            
            if i != self.n_layers-2:
                test_scores = np.maximum(0, test_scores) # ReLU
                
        return np.argmax(test_scores, axis = 1)
