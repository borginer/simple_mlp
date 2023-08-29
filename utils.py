import numpy as np

def ReLU(vec):
    return np.maximum(vec, 0)

def stable_softmax(vec):
    exps = np.exp(vec - np.max(vec))
    return exps / np.sum(exps)

def ReLU_grad(grad, vec):
    vec = np.where(vec > 0 , 1, 0)
    return grad * vec

""" combination of softmax and cross entropy """
def softmax_cross_ent_grad(vec, y):
    return vec - y
            
def layer_weights_grad(grad, vec):
    return np.dot(grad, vec.T) 


    