# coding: utf-8
import numpy as np
import tensorflow as tf

def sigmoid(x):
    y = 1/(1+np.exp(-x))
    return y

def sigmoid_grad(x):
    g = x*(1-x)
    return g

def affine_forward(X, W, b):
    Z = np.dot(X, W) + b
    cache = X, W, b, Z
    return Z, cache

def affine_backward(dZ, cache):
    X, W, b, Z = cache
    
    dW = np.dot(np.transpose(X), dZ)
    db = np.sum(dZ, axis = 0)
    dX = np.dot(dZ, np.transpose(W))
    return dW, db, dX

def relu_forward(x):
    y = np.maximum(x, 0)
    return y

def relu_backward(x):
    g = x>0
    return g
    
    
    
    

