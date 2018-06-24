# -*- coding: utf-8 -*-
# Author : tyty
# Date   : 2018-6-20
from  __future__ import division
import pickle
import pandas as pd
import numpy as np

def vectorized(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    e = np.zeros((4, 1))
    e[j-1] = 1.0
    return e

def createDataSet(splitSize=0.2):
    fruit = pd.read_table('./fruit.txt')
    #convert pd.DataFrame -> ndarray -> list 
    fruit.head()
    #print fruit.shape
    labels = ['mass', 'width', 'height', 'color_score', 'fruit_label']
    train = ['mass', 'width', 'height', 'color_score']
#    trainLabel = fruit[[]]
    #choose the labels fruits data
    trainData = fruit[labels]
    training_data = fruit[train]
#    numpytrainLabel = np.array(trainLabel)
    
    numpyTrainData = np.array(trainData)
    numpyTrainData2 = np.array(training_data)
    # dataSet = numpy_train_data.tolist()
    #list - dataSet
    recordNums = numpyTrainData.shape[0]
    trainDataIndex = range(recordNums)
    #train_data_index = [1, ..., 59]
    testDataIndex = []
    testNumber = int(recordNums * splitSize)
    for i in range(testNumber):
    	#choose test_number test e.g.s
    	randomNum = int(np.random.uniform(0, len(trainDataIndex)))
    	testDataIndex.append(trainDataIndex[randomNum])
    	del trainDataIndex[randomNum]
    
    trainSet = numpyTrainData[trainDataIndex]
    testSet  = numpyTrainData[testDataIndex]
    testlabel  = [a[-1] for a in testSet]
    trainlabel = [a[-1] for a in trainSet]
    training_data = numpyTrainData2[trainDataIndex]
    test_data = numpyTrainData2[testDataIndex]
#    print training_data
    for i in range(len(training_data[0])):
        temp = [a[i] for a in training_data]
        maxNumber = max(temp)
        minNumber = min(temp)
        #standardize the dataSet
        for j in range(len(training_data)):
            denominator = maxNumber - minNumber
            training_data[j][i] = (training_data[j][i] - minNumber) / denominator
#    print training_data
            
    trainSet = [np.reshape(a, (4, 1)) for a in training_data]
    for i in range(len(trainlabel)):
        trainlabel[i] = int(trainlabel[i])
        trainlabel[i] -= 1
    train2 = zip(trainSet, trainlabel)
    print train2

    
    trainL = [vectorized(int(y)) for y in trainlabel]
    training_data = zip(trainSet, trainL)
#    print training_data
    testSet = [np.reshape(a, (4, 1)) for a in test_data]
    
    for i in range(len(testlabel)):
        testlabel[i] = int(testlabel[i])
        testlabel[i] -= 1
#        print testlabel[i]
    test_data = zip(testSet, testlabel)
#    trainSet = trainSet.tolist()
#    testSet  = testSet.tolist()

#    trainLabel = [a[-1]  for a in trainSet]
#    trainSet   = [a[:-1] for a in trainSet]
#    testSet    = [a[:-1] for a in testSet]
    # print testSet
    # print testlabel
    return training_data, test_data, train2

trainSet, testSet, _ = createDataSet()