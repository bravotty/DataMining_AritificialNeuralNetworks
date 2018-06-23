# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 22:58:17 2018

@author: tyty
"""

import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def dsigmoid(y):
    return y * (1 - y)


def tanh(x):
    return np.tanh(x)


def dtanh(y):
    return 1.0 - y ** 2


def relu(y):
    tmp = y.copy()
    tmp[tmp < 0] = 0
    return tmp


def drelu(x):
    tmp = x.copy()
    tmp[tmp >= 0] = 1
    tmp[tmp < 0] = 0
    return tmp



class MLPClassifier(object):
    """多层感知机，BP 算法训练"""

    def __init__(self,
                 layers,
                 activation='tanh',
                 epochs=20, batch_size=1, learning_rate=0.01):
        """
        :param layers: 网络层结构
        :param activation: 激活函数
        :param epochs: 迭代轮次
        :param learning_rate: 学习率 
        """
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.layers = []
        self.weights = []
        self.batch_size = batch_size
        self.thresholds = []

        for i in range(0, len(layers) - 1):
            weight = np.random.random((layers[i], layers[i + 1]))
            layer = np.ones(layers[i])
            self.layers.append(layer)
            self.weights.append(weight)
        self.layers.append(np.ones(layers[-1]))
        print ('layer!!')
        print (self.layers)
        print ('weights!!')
        print (self.weights)
        for i in range(1, len(layers)):
            threshold = np.random.random(layers[i])
            self.thresholds.append(threshold)
        print ('biases!!')
        print (self.thresholds)
        if activation == 'tanh':
            self.activation = tanh
            self.dactivation = dtanh
        elif activation == 'sigomid':
            self.activation = sigmoid
            self.dactivation = dsigmoid
        elif activation == 'relu':
            self.activation = relu
            self.dactivation = drelu

    def fit(self, X, y):
        """
        :param X_: shape = [n_samples, n_features] 
        :param y: shape = [n_samples] 
        :return: self
        """
        #//去整除
        
        for _ in range(self.epochs * (X.shape[0] // self.batch_size)):
            #print (self.epochs * (X.shape[0] // self.batch_size))
            #300*200//1
            i = np.random.choice(X.shape[0], self.batch_size)
            #print (i)
            #在输入的X中随机取一个序号
            # i = np.random.randint(X.shape[0])
            self.update(X[i])
            self.back_propagate(y[i])

    def predict(self, X):
        """
        :param X: shape = [n_samples, n_features] 
        :return: shape = [n_samples]
        """
        self.update(X)
        return self.layers[-1].copy()

    def update(self, inputs):
        self.layers[0] = inputs
        for i in range(len(self.weights)):
            print ('line')
            print (self.layers[i])
            print ('line2')
            print (self.weights[i])
            #print (self.weights[i]*self.layers[i])
            print ('line3')
            print (self.layers[i] @ self.weights[i])
            
            next_layer_in = self.layers[i] @ self.weights[i] - self.thresholds[i]
            
            self.layers[i + 1] = self.activation(next_layer_in)

    def back_propagate(self, y):
        errors = y - self.layers[-1]

        gradients = [(self.dactivation(self.layers[-1]) * errors).sum(axis=0)]

        self.thresholds[-1] -= self.learning_rate * gradients[-1]
        for i in range(len(self.weights) - 1, 0, -1):
            tmp = np.sum(gradients[-1] @ self.weights[i].T * self.dactivation(self.layers[i]), axis=0)
            gradients.append(tmp)
            self.thresholds[i - 1] -= self.learning_rate * gradients[-1] / self.batch_size
        gradients.reverse()
        for i in range(len(self.weights)):
            tmp = np.mean(self.layers[i], axis=0)
            self.weights[i] += self.learning_rate * tmp.reshape((-1, 1)) * gradients[i]
            
            
n = MLPClassifier((2, 4, 1), activation='relu', epochs=300, learning_rate=0.01)
import sklearn.datasets
X, y = sklearn.datasets.make_moons(200, noise=0.20)


y = y.reshape((-1, 1))
n.fit(X, y)


