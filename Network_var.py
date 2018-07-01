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
        self.stdVar   = [np.var(i)  for i in trainX.T]
        # for epoch show
        self.trainXPrediction = trainX
        self.trainYPrediction = trainY
        self.testXPrediction  = testX
        self.testYPrediction  = testY
        #process the trainX data and trainY data
        self.trainX   = self.Normalization(trainX)
        self.trainY   = self.oneHotDataProcessing(trainY)
        # self.trainY = trainY
        self.weights  = [np.random.uniform(-0.5, 0.5, [y, x]) for x, y in zip(layers[:-1], layers[1:])]
        self.biases   = [np.zeros([y, 1]) for y in layers[1:]]
        # self.biases   = [np.random.uniform(-1, 1, [y, 1]) for y in layers[1:]]
        # print self.weights[0]
        self.cntLayer = len(self.layers) - 1
        self.error    = None

    def fitTransform(self):
        for i in range(self.epoch):
            for trainX, trainY in zip(self.trainX, self.trainY):
                # 2 step to train the network
                # 1.forwardUpdate the network params
                netLayerInput, netLayerOuput = self.forwardUpdate(trainX)
                # 2. backForwardUpdata the network params
                self.backForwardUpdate(netLayerInput, netLayerOuput, trainY)
            # * print every epoch TrainSet OR TestSet Accuracy *
            print ("Epoch {0} : testSet accuracy: {1} / {2}   |   trainSet accuracy: {3} / {4}".\
                format(i, self.prediction(testX=self.testXPrediction,testY=self.testYPrediction)[1], len(self.testXPrediction),\
                    self.prediction(testX=self.trainXPrediction,testY=self.trainYPrediction)[1],len(self.trainXPrediction)))

    def forwardUpdate(self, trainX):
        tmpTrain = trainX
        layerOutput = []
        layerInput  = []
        # forward update 
        for layer in range(self.cntLayer):
            layerInput.append(tmpTrain)
            # cal the train value
            tmpTrain = np.dot(tmpTrain, self.weights[layer].T) + self.biases[layer].T
            # activate the train value - sigmoid
            tmpTrain = [self.sigmoid(i) for i in tmpTrain]
            # tmpTrain = [self.ReLU(i) for i in tmpTrain]
            # tmpTrain = [self.tanh(i) for i in tmpTrain]
            layerOutput.append(tmpTrain)
        return layerInput, layerOutput

    def backForwardUpdate(self, netLayerInput, netLayerOutput, trainY):
        trainY = np.array(trainY)
        #reverse the order to cal the gradient
        for layerIndex, netInput, netOutput in zip(range(self.cntLayer)[ : : -1],\
                                     netLayerInput[ : : -1], netLayerOutput[ : : -1]):
            netIn  = np.array(netInput)
            netOut = np.array(netOutput)

            # cal the error of y - last layer
            if layerIndex == (self.cntLayer - 1):
                self.error = self.sigmoidPrime(netOut) * self.costFunction(realY=trainY,\
                                                                       outputY=netOut)
                # self.error = self.ReLUPrime(netOut) * self.costFunction(realY=trainY,\
                #                                                       outputY=netOut)
                # self.error = self.tanhPrime(netOut) * self.costFunction(realY=trainY,\
                #                                                       outputY=netOut)                
            else:
                # update the error of hidden layer 
                # "layerIndex + 1" index the behind layer
                self.error = np.dot(self.error, self.weights[layerIndex + 1])
                self.error = self.sigmoidPrime(netOut) * self.error
                # self.error = self.ReLUPrime(netOut) * self.error
                # self.error = self.tanhPrime(netOut) * self.error

            # !! extract the No.2 Axis of error -- Error of each layer !!
            self.error = self.error[0]
            for n in range(len(self.weights[layerIndex])):
                self.weights[layerIndex][n] = self.weights[layerIndex][n] + netIn * self.lr * self.error[n]
                self.biases[layerIndex] = (self.biases[layerIndex].T + self.lr * self.error).T

    #-----------------------------------------------
    #----- Activation function and its prime format
    def sigmoid(self, inputX):
        return [1 / (1 + np.math.exp(-i)) for i in inputX]

    def sigmoidPrime(self, outputY):
        return outputY * (1 - outputY)

    def tanh(self, inputX):
        return np.tanh(inputX)

    def tanhPrime(self, outputY):
        return 1.0 - outputY ** 2

    def ReLU(self, inputX):
        inputXReLU = inputX
        inputXReLU[inputXReLU < 0] = 0
        return inputXReLU 

    def ReLUPrime(self, outputY):
        outputYReLU = outputY
        outputYReLU[outputYReLU >= 0] = 1
        outputYReLU[outputYReLU <  0] = 0
        return outputYReLU
    #---End of the activation function
    #-----------------------------------------------

    # cost function / loss function
    def costFunction(self, realY, outputY):
        return (realY - outputY)

    # input params  : trainSet X
    # output params : (X - mean(X)) / std(X)
    def Normalization(self, trainX):
        # reverse the trainX [40,4]->[4->40]
        # print data.shape
        data = trainX.T
        for i in range(len(data)):
            data[i] = (data[i] - self.mean[i]) / self.stdVar[i]
        return data.T

    # input params  : trainSet Y
    # output params : ont hot  Y e.g.[3] = [0, 0, 1, 0], [4] = [0, 0, 0, 1]
    def oneHotDataProcessing(self, trainY):
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
        stdVar = [np.var(i)  for i in testX]
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
                # tmp = [self.ReLU(i) for i in tmp]
                # tmp = [self.tanh(i) for i in tmp]
            result.append(np.argmax(tmp) + 1)
        for realY, predY in zip(testY, result):
            if realY == predY:
                res += 1
        accuracy = res / len(testY)
        return  accuracy, res

def AritificialNeuralNetworksModelMain():
    train, trainy, test, testy = tl.createDataSet()
    # layers, learningRate, trainX, trainY, testX, testY, epoch
    ANNModel = AritificialNeuralNetworks(layers=[4, 150, 4], learningRate=0.1, trainX=train,\
                                         trainY=trainy, testX=test, testY=testy, epoch = 500)
    # fit the model with training data
    ANNModel.fitTransform()
    # cal the accuracy
    accuracy = ANNModel.prediction(testX=test, testY=testy)[0]
    print ("The accuracy of the test dataSet :  " + str(accuracy))

if __name__ == '__main__':
    AritificialNeuralNetworksModelMain()

