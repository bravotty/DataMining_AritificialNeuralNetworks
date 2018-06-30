# -*- coding: utf-8 -*-
# Author : tyty
# Date   : 2018-6-21
from  __future__ import division
import numpy as np
import pandas as pd
import random
import tools as tl


class AritificialNeuralNetworks(object):
    def __init__(self, layers, learningRate, trainX, trainY, testX, testY, epoch):
        # input params
        self.layers   = layers
        self.lr       = learningRate
        self.epoch    = epoch

        # cal the mean and std
        self.mean     = [np.mean(i) for i in trainX.T]
        self.stdVar   = [np.std(i)  for i in trainX.T]

        self.trainXPrediction = trainX
        self.trainYPrediction = trainY
        self.testXPrediction  = testX
        self.testYPrediction  = testY
        self.trainX   = self.dataNormalization(trainX)
        # print len(trainX)
        self.trainY   = self.onHotDataProcessing(trainY)
        # self.trainY = trainY
        self.weights  = [np.random.uniform(-1, 1, [y, x]) for x, y in zip(layers[:-1], layers[1:])]
        self.biases   = [np.zeros([y, 1]) for y in layers[1:]]
        # self.biases   = [np.random.uniform(-1, 1, [y, 1]) for y in layers[1:]]
        # print self.weights[0]
        self.cntLayer = len(self.layers) - 1
        self.error    = None

    # a = np.array(self.weights)
    # b = np.array(self.biases)
    # print a.shape

    def fitTransform(self):
        for i in range(self.epoch):
            for trainX, trainY in zip(self.trainX, self.trainY):
                # 2 step to train the network
                # 1.forwardUpdate the network params
                netLayerInput, netLayerOuput = self.forwardUpdate(trainX)
                # 2. backForwardUpdata the network params
                self.backForwardUpdate(netLayerInput, netLayerOuput, trainY)

            print ("Epoch {0}: {1} / {2}".format(i, self.prediction(testX=self.testXPrediction,\
                                                    testY=self.testYPrediction)[1], 11))

    def forwardUpdate(self, trainX):
        d = trainX
        layerOutput = []
        layerInput  = []
        for layer in range(len(self.layers) - 1):
            layerInput.append(d)
            d = np.dot(d, self.weights[layer].T) + self.biases[layer].T
            d = [self.sigmoid(i) for i in d]
            layerOutput.append(d)
        return layerInput, layerOutput

    def backForwardUpdate(self, netLayerInput, netLayerOutput, trainY):
        trainY = np.array(trainY)
        #reverse the order to cal the gradient
        for layerIndex, netInput, netOutput in zip(range(self.cntLayer)[ : : -1],\
                                     netLayerInput[ : : -1], netLayerOutput[ : : -1]):
            netIn  = np.array(netInput)
            netOut = np.array(netOutput)

            if layerIndex == (self.cntLayer - 1):
                #cal the error of y
                self.error = netOut * (1 - netOut) * self.costFunction(realY=trainY,\
                                                                       inputY=netOut)
            else:
                #update the error of hidden layer
                self.error = np.dot(self.error, self.weights[layerIndex + 1])
                self.error = netOut * (1 - netOut) * self.error

            for n in range(len(self.weights[layerIndex])):
                self.weights[layerIndex][n] = self.weights[layerIndex][n] + self.lr * self.error[0][n] * netIn
                self.biases[layerIndex] = (self.biases[layerIndex].T + self.lr * self.error).T


    
    def sigmoid(self, inputX):
        return [1 / (1 + np.math.exp(-i)) for i in inputX]

    def costFunction(self, realY, inputY):
        return (realY - inputY)

    def dataNormalization(self, trainX):
        # reverse the trainX [40,4]->[4->40]
        # print data.shape
        data = trainX.T
        for i in range(len(data)):
            data[i] = (data[i] - self.mean[i]) / self.stdVar[i]
        return data.T

    def onHotDataProcessing(self, trainY):
        res = []
        # lenght of onehot data is self.layers[-1]
        # print self.layers[-1]
        for i in trainY:
            # initial the temp list -> 0
            temp = [0] * self.layers[-1]
            # cal the idx of '1'
            idx = int(i - 1)
            # mark the '1'
            temp[idx] = 1
            res.append(temp)
        # print res
        return res

    def prediction(self, testX, testY):
        res = 0
        result = []
        testX = np.array(testX).T
        # print (self.mean)
        mean   = [np.mean(i) for i in testX]
        stdVar = [np.std(i) for i in testX]
        for i in range(len(testX)):
            # use trainSet mean std or testSet mean std
            # testX[i] = (testX[i]- self.mean[i]) / self.stdVar[i]
            testX[i] = (testX[i] - mean[i]) / stdVar[i]
        testX = testX.T
        for tX in testX:
            tmp = tX
            for layer in range(self.cntLayer):
                tmp = np.dot(tmp, self.weights[layer].T) + self.biases[layer].T
                tmp = [self.sigmoid(i) for i in tmp]
            result.append(np.argmax(tmp) + 1)
        for realY, predY in zip(testY, result):
            if realY == predY:
                res += 1
        accuracy = res / len(testY)

        return  accuracy, res

def AritificialNeuralNetworksModelMain():
    train, trainy, test, testy = tl.createDataSet()
    # layers, learningRate, trainX, trainY, testX, testY, epoch
    ANNModel = AritificialNeuralNetworks(layers=[4, 100, 100, 4], learningRate=0.1, trainX=train,\
                                         trainY=trainy, testX=test, testY=testy, epoch = 300)
    # fit the model with training data
    ANNModel.fitTransform()
    # cal the accuracy
    accuracy = ANNModel.prediction(testX=test, testY=testy)[0]
    print ("The accuracy of the test dataSet :  " + str(accuracy))


if __name__ == '__main__':
    AritificialNeuralNetworksModelMain()


