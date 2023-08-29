import numpy as np
from utils import *

class Layer: 
    
    def __init__(self, in_size , out_size, act_func) -> None:
        self.input_size = in_size
        self.out_size = out_size
        
        self.bias = np.zeros((out_size, 1))
        self.bias_grad = np.zeros((out_size, 1))
        
        self.weights = np.random.normal(0, np.sqrt(2/in_size), size = (out_size, in_size))
        self.weights_grad = np.zeros((out_size, in_size))
        self.mask = np.ones(self.weights.shape)
        
        self.act_func = act_func

    def forward(self, x):
        output = np.dot(self.weights, x)
        output = output + self.bias
        return output, self.act_func(output)
    
    def update_params(self, learning_rate, batch_size):
        self.weights -= (learning_rate/batch_size) * self.weights_grad
        self.bias -= (learning_rate/batch_size) * self.bias_grad
        self.weights_grad = np.zeros((self.out_size, self.input_size))
        self.bias_grad = np.zeros((self.out_size, 1))

class Multilayer:
    
        def __init__(self, layer_sizes) -> None:
            layers = [None]*(len(layer_sizes) - 1)
            for i in range(len(layers) - 1):
                layers[i] = Layer(layer_sizes[i], layer_sizes[i + 1], ReLU)
            layers[-1] = Layer(layer_sizes[-2], layer_sizes[-1], stable_softmax)
            self.layers = layers
            
            self.activation_log = [None]*(len(self.layers) + 1)
            self.input_log = [None]*(len(self.layers))
            self.learning_rate = 0.1
        
        def set_learning_rate(self, rate):
            self.learning_rate = rate
        
        def layers_stats(self):
            for i, layer in enumerate(self.layers):
                print("Layer {}: Avg: {} | Max: {} | Min: {}".format(i, 
                                              round(np.mean(self.layers[i].weights), 5), 
                                              round(np.max(self.layers[i].weights), 5),
                                              round(np.min(self.layers[i].weights), 5)))
        
        def update_mlp(self, batch_size):
            for layer in self.layers:
                layer.update_params(self.learning_rate, batch_size)
                layer.weights *= layer.mask
                
        def feed_forward(self, x):
            a = x
            for layer in self.layers:
                _, a = layer.forward(a)
            return a
        
        """ feed forward with saving the inputs and activations"""
        def sfeed_forward(self, x):
            a = x
            self.activation_log[0] = x
            for i, layer in enumerate(self.layers):
                self.input_log[i], a = layer.forward(a)
                self.activation_log[i + 1] = a
            return a
        
        def back_prop(self, y):
            grad = softmax_cross_ent_grad(self.activation_log[-1], y)
            self.layers[-1].bias_grad += grad
            self.layers[-1].weights_grad += layer_weights_grad(grad, self.activation_log[-2])
            
            for i in range(2, len(self.layers) + 1):
                grad = ReLU_grad(np.dot(self.layers[-i + 1].weights.T, grad), self.input_log[-i])
                self.layers[-i].bias_grad += grad
                self.layers[-i].weights_grad += layer_weights_grad(grad, self.activation_log[-i - 1])
        
        """ zeros out random portion p of weights in each layer """
        def dilution(self, p):
            for layer in self.layers:
                sample = np.random.binomial(1,1-p,size=(layer.weights.shape))
                layer.weights *= sample
                
        """ cuts out randomly and permanently portion p of the weights in each layer """            
        def dropout(self, p, max_drop):
            for layer in self.layers:
                if np.count_nonzero(layer.weights) > (1 - max_drop)*np.prod(layer.weights.shape):
                    sample = np.random.binomial(1,1-p,size=(layer.weights.shape))
                    layer.mask *= sample
                    
        """ runs stochastic gradient descent on the MLP 
            zeroes out percent p of the weights after each epoch
            linearly decays the learning rate from LR_start to LR_end over the epochs
            """
        def run_SGD(self, epochs, train_points, train_targets, test_points, test_targets, 
                    batch_size = 1, p = 0, decay_rate = 0.95, LR_start = 0.1, LR_end = 0.01):
            
            for i in range(epochs):
                self.learning_rate = LR_start + (LR_end - LR_start) * i / epochs
                for j in range(int(len(train_points)/batch_size)):
                    self.train_batch(train_points, train_targets, j * batch_size, batch_size)
                    
                print("epoch {}:".format(i + 1))
                winrate = self.test(train_points, train_targets)
                print("data winrate: {}%".format(round(winrate, 2)))
                
                winrate = self.test(test_points, test_targets)
                print("test winrate: {}% \n".format(round(winrate, 2)))
                
                self.dilution(p)
                p *= decay_rate
                   
        def train_batch(self, train_points, train_targets, batch_start, batch_size):
            for i in range(batch_size):
                self.sfeed_forward(train_points[batch_start + i])
                self.back_prop(train_targets[batch_start + i])
            self.update_mlp(batch_size)
                       
        def test(self, test_points, test_targets):
            wins = 0
            for j in range(len(test_points)):
                guess = np.argmax(self.feed_forward(test_points[j]))
                if guess == np.argmax(test_targets[j]):
                    wins += 1  
            return 100 * wins / len(test_points)
            
            
        

            
