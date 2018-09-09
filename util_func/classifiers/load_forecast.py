# coding: utf-8
import numpy as np
import tensorflow as tf
from util_func.layers import *


class linear_regression(object):    
    
    def __init__(self, input_dim, reg = 0.0):
    # Initialize parameters for linear regression
    
        self.params = {}
        self.reg = reg
        
        self.params['W1'] = np.random.randn(input_dim, 1) / np.sqrt(input_dim) # use Xavier initialization
        self.params['b1'] = np.zeros((1,))
        
        
    def loss(self, X, y = None):
        
        m = X.shape[0]
        n = X.shape[1]
        
        W1, b1 = self.params['W1'], self.params['b1']
        y_hat, cache = affine_forward(X, W1, b1)
        if y is None:
            return y_hat
        
        loss = np.sum(np.square(y_hat - y))/(2*m) + 0.5*self.reg*np.sum(np.square(W1))  # use least squared error
        grads = {}
        dy_hat = (y_hat-y)/m
        
        dW, db, dX = affine_backward(dy_hat, cache)
        grads['W1'] = dW + self.reg*(W1)
        grads['b1'] = db
        
        return loss, grads

    
class two_layer_net(object):    
    
    def __init__(self, input_dim, hidden_dim, activation, reg = 0.0):
        # initialize parameters for two_net_layer            
        self.params = {}
        self.reg = reg
        self.activation = activation
             
        self.params['W1'] = np.random.randn(input_dim, hidden_dim) / np.sqrt(input_dim)
        self.params['b1'] = np.zeros((hidden_dim, ))
        self.params['W2'] = np.random.randn(hidden_dim, 1) / np.sqrt(hidden_dim)
        self.params['b2'] = np.zeros((1,))
            
    def loss(self, X, y = None):
            
        m = X.shape[0]
        n = X.shape[1]
        grads = {}
            
        W1, b1, W2, b2 = self.params['W1'], self.params['b1'], self.params['W2'], self.params['b2']
            
        #implement loss            
        Z1, cache1 = affine_forward(X, W1, b1)
        if self.activation == 'sigmoid':
            A1 = sigmoid(Z1)
                
        elif self.activation == 'relu':
            A1 = np.maximum(Z1, 0)
            
        Z2, cache2 = affine_forward(A1, W2, b2)
        score = Z2
        if y is None:
            return score
            
        loss = np.sum(np.square(score-y))/(2*m)       # use least square error
        loss += 0.5*self.reg*np.sum((np.square(W1))) + 0.5*self.reg*np.sum((np.square(W2)))
        
        #implement gradient            
        dscore = (score - y) / m
        
        dW2, db2, dX2 = affine_backward(dscore, cache2)
        grads['W2'] = dW2 + self.reg*(W2)
        grads['b2'] = db2
        dA1 = dX2
            
        if self.activation == 'sigmoid':
            dZ1 = dA1*sigmoid_grad(A1)
                
        elif self.activation == 'relu':
            dZ1 = dA1*relu_backward(A1)
                
        
        dW1, db1, dX1 = affine_backward(dZ1, cache1)
        grads['W1'] = dW1 + self.reg*(W1)
        grads['b1'] = db1
            
        return loss, grads
    
