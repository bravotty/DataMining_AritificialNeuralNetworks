# -*- coding: utf-8 -*-
from  __future__ import division

"""
Created on Thu Jun 14 14:58:25 2018

@author: tyty
"""

"""
mnist_loader
~~~~~~~~~~~~
A library to load the MNIST image data.  For details of the data
structures that are returned, see the doc strings for ``load_data``
and ``load_data_wrapper``.  In practice, ``load_data_wrapper`` is the
function usually called by our neural network code.
"""

#### Libraries
# Standard library
import pickle
import gzip
import pandas as pd
# Third-party libraries
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

#trainSet, testSet = createDataSet()
#print trainSet

def load_data():
    """Return the MNIST data as a tuple containing the training data,
    the validation data, and the test data.
    The ``training_data`` is returned as a tuple with two entries.
    The first entry contains the actual training images.  This is a
    numpy ndarray with 50,000 entries.  Each entry is, in turn, a
    numpy ndarray with 784 values, representing the 28 * 28 = 784
    pixels in a single MNIST image.
    The second entry in the ``training_data`` tuple is a numpy ndarray
    containing 50,000 entries.  Those entries are just the digit
    values (0...9) for the corresponding images contained in the first
    entry of the tuple.
    The ``validation_data`` and ``test_data`` are similar, except
    each contains only 10,000 images.
    This is a nice data format, but for use in neural networks it's
    helpful to modify the format of the ``training_data`` a little.
    That's done in the wrapper function ``load_data_wrapper()``, see
    below.
    """
    f = gzip.open('./data/mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = pickle.load(f)
    f.close()
    return (training_data, validation_data, test_data)

def load_data_wrapper():
    """Return a tuple containing ``(training_data, validation_data,
    test_data)``. Based on ``load_data``, but the format is more
    convenient for use in our implementation of neural networks.
    In particular, ``training_data`` is a list containing 50,000
    2-tuples ``(x, y)``.  ``x`` is a 784-dimensional numpy.ndarray
    containing the input image.  ``y`` is a 10-dimensional
    numpy.ndarray representing the unit vector corresponding to the
    correct digit for ``x``.
    ``validation_data`` and ``test_data`` are lists containing 10,000
    2-tuples ``(x, y)``.  In each case, ``x`` is a 784-dimensional
    numpy.ndarry containing the input image, and ``y`` is the
    corresponding classification, i.e., the digit values (integers)
    corresponding to ``x``.
    Obviously, this means we're using slightly different formats for
    the training data and the validation / test data.  These formats
    turn out to be the most convenient for use in our neural network
    code."""
    tr_d, va_d, te_d = load_data()
    print tr_d[0].shape
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = zip(training_inputs, training_results)
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = zip(validation_inputs, va_d[1])
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = zip(test_inputs, te_d[1])
    return (training_data, validation_data, test_data)

def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

