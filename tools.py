# -*- coding: utf-8 -*-
# Author : tyty
# Date   : 2018-6-20
from  __future__ import division
import pandas as pd
import numpy as np

def createDataSet(splitSize=0.2):
    fruit = pd.read_table('./fruit.txt')
    #convert pd.DataFrame -> list -> ndarray
    fruit.head()
    labels = ['mass', 'width', 'height', 'color_score', 'fruit_label']
    #choose the labels fruits data
    trainData = fruit[labels]
    numpyTrainData = np.array(trainData)
    # dataSet = numpy_train_data.tolist()
    #list - dataSet
    recordNums = numpyTrainData.shape[0]
    trainDataIndex = range(recordNums)
# MaxMinScalar func
#     for i in range(len(training_data[0])):
#         temp = [a[i] for a in training_data]
#         maxNumber = max(temp)
#         minNumber = min(temp)
#         #standardize the dataSet
#         for j in range(len(training_data)):
#             denominator = maxNumber - minNumber
#             training_data[j][i] = (training_data[j][i] - minNumber) / denominator
    #choose the random trainSet and testSet
    testDataIndex = []
    testNumber = int(recordNums * splitSize)
    for i in range(testNumber):
        #choose test_number test e.g.s
        randomNum = int(np.random.uniform(0, len(trainDataIndex)))
        testDataIndex.append(trainDataIndex[randomNum])
        del trainDataIndex[randomNum]
    trainSet = numpyTrainData[trainDataIndex]
    testSet  = numpyTrainData[testDataIndex]
    # trainSet = trainSet.tolist()
    # testSet  = testSet.tolist()
    trainLabel = [a[-1]  for a in trainSet]
    trainSet   = [a[:-1] for a in trainSet]
    testLabel  = [a[-1]  for a in testSet]
    testSet    = [a[:-1] for a in testSet]
    # list type to ndarray
    trainLabel = np.array(trainLabel)
    trainSet   = np.array(trainSet)
    testLabel  = np.array(testLabel)
    testSet    = np.array(testSet)
    
    return trainSet, trainLabel, testSet, testLabel

