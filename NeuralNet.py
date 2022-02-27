#!/usr/bin/env python
# coding: utf-8

# In[19]:


conda update numpy

import numpy as np

np.randon.seed(0)

X = [[1, 2, 3, 2.5],
     [2.0, 5.0, -1.0, 2.0],
     [-1.5, 2.7, 3.3, -0.8]]


class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def foreward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
    
layer1 = Layer_Dense(4,5)
layer2 = Layer_Dense(5,2)

layer1.foreward(X)
print(layer1.output)
    


# In[ ]:





# In[1]:


import numpy as np
import nnfs

nnfs.init()

X = [[1, 2, 3, 2.5],
     [2.0, 5.0, -1.0, 2.0],
     [-1.5, 2.7, 3.3, -0.8]]


class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
    def foreward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

    class Activation_ReU:
        def foreward(self, inputs):
            self.output = np.maximum(0, inputs)

    layer1 = Layer_Dense(4,5)
    layer2 = Layer_Dense(5,2)
    
    layer1.foreward(X)
    print(layer1.output)
    layer2.foreward(layer1.output)
    print(layer2.output)


# In[ ]:




