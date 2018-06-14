# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 14:26:25 2018

@author: tyty
"""

import random
import numpy as np 

def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))


class Network():
    def __init__(self, sizes):
        #input like sizes = [4, 5, 4]
        self.layerNums = len(sizes)
        self.sizes = sizes
        print (sizes[:-1], sizes[1:])
        #sigma * np.random.randn(size1, size2) + mu
        #weights size = [[4]->[5], [5]->[4]]
        self.weights = [np.random.randn(y, x) 
        for x, y in zip(sizes[:-1], sizes[1:])]
        print (self.weights)
        #biases size= [[5], [4]]
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]

    def feedforwoard(self, a):
        for w, b in zip(self.weights, self.biases):
            a = sigmoid(np.dot(w, a) + b)

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        if test_data:
            n_test = len(test_data)
        n = len(training_data)
        for j in xrange(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size]
            for k in xrange(0, n, mini_batch_size)]
            
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print ( "Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), n_test))
            else:
                print ("Epoch {0} complete".format(j))
        
        
        
        
clf = Network([4, 5, 4])
print (clf)


