# -*- coding: utf-8 -*-
# Author : tyty
# Date   : 2018-6-21
from  __future__ import division
import numpy as np
import pandas as pd
import random
import tools as tl


class AritificialNeuralNetworks(object):
	def __init__(self, layers, learningRate, trainX, trainY, epoch):
		#input params
		self.layers  = layers
		self.lr      = learningRate
		self.trainX  = self.dataNormalization(trainX)
		self.trainY  = self.onHotDataProcessing(trainY)
		# self.trainY = trainY
		self.epoch   = epoch
		self.weights = [np.random.uniform(-1, 1, [y, x]) for x, y in zip(layers[:-1], layers[1:])]
		self.biases  = [np.zeros([y, 1]) for y in layers[1:]]
		# a = np.array(self.weights)
		# b = np.array(self.biases)
		# print a.shape

	def fitTransform(self):
		for i in xrange(self.epoch):
			for trainX, trainY in zip(self.trainX, self.trainY):
				# 2 step to train the network
				# 1.forwardUpdate the network params
				netLayerInput, netLayerOuput = self.forwardUpdate(trainX)
				# 2. backForwardUpdata the network params
				self.backForwardUpdate(netLayerInput, netLayerOuput, trainY)


	def forwardUpdate(self, trainX):
		d = trainX
		layerOutput = []
		layerInput  = []
		
		for layer in range(len(self.layers) - 1):
			layerInput.append(d)
			
			print self.weights[layer].T.shape
			d = np.dot(d, self.weights[layer].T) + self.biases[layer]
			d = [self.sigmoid(i) for i in d]
			layerOutput.append(d)
		return layerInput, layerOutput
		



	def backForwardUpdate():
		pass

	def sigmoid(self, inputX):
		return [ 1 / (1 + np.math.exp(-i)) for i in inputX]


	def dataNormalization(self, trainX):
		#reverse the trainX [40,4]->[4->40]
		#print data.shape
		data = trainX.T

		mean    = [np.mean(i) for i in data]
		# print self.mean
		stdVar  = [np.std(i)  for i in data]
		for i in range(len(data)):
			data[i] = (data[i] - mean[i]) / stdVar[i]
		return data.T

	def onHotDataProcessing(self, trainY):
		res = []
		# lenght of onehot data is self.layers[-1]
		# print self.layers[-1]
		for i in trainY:
			#initial the temp list -> 0
			temp = [0] *self.layers[-1]
			#cal the idx of '1'
			idx = int(i - 1)
			#mark the '1'
			temp[idx] = 1
			res.append(temp)
		# print res
		return res

def AritificialNeuralNetworksModelMain():
	train, trainy, test, testy = tl.createDataSet()
	ANNModel = AritificialNeuralNetworks([4, 6, 4], 0.1, train, trainy, 1000)
	ANNModel.fitTransform()



if __name__ == '__main__':
	AritificialNeuralNetworksModelMain()


