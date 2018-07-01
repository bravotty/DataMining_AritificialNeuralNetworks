# DataMining_AritificialNeuralNetworks

[![auc][aucsvg]][auc] [![License][licensesvg]][license]

[aucsvg]: https://img.shields.io/badge/tyty-AritificialNeuralNetworks-red.svg
[auc]: https://github.com/bravotty/DataMining_AritificialNeuralNetworks

[licensesvg]: https://img.shields.io/badge/License-MIT-blue.svg
[license]: https://github.com/bravotty/DataMining_AritificialNeuralNetworks/blob/master/LICENSE

```
A python implementation of AritificialNeuralNetworks - ANN
Env : Python2.6
```

## Usage     : 

PC python : 
```lisp
pip install numpy, pandas
```

Run .py
```lisp
python Network.py
```

## Defination :

AritificialNeuralNetworks Struct
```lisp
class AritificialNeuralNetworks(object):
    def __init__(self, layers, learningRate, trainX, trainY, testX, testY, epoch):
        # input params
        self.layers   = layers 
        self.lr       = learningRate
        self.epoch    = epoch
        self.mean     = [np.mean(i) for i in trainX.T]
        self.stdVar   = [np.std(i)  for i in trainX.T]
        self.trainXPrediction = trainX
        self.trainYPrediction = trainY
        self.testXPrediction  = testX
        self.testYPrediction  = testY
        self.trainX   = self.Normalization(trainX)
        self.trainY   = self.oneHotDataProcessing(trainY)
        self.weights  = [np.random.uniform(-0.5, 0.5, [y, x]) for x, y in zip(layers[:-1], layers[1:])]
        self.biases   = [np.zeros([y, 1]) for y in layers[1:]]
        self.cntLayer = len(self.layers) - 1
        self.error    = None
```

## Code Flie  :
```lisp
AritificialNeuralNetworks.py
  |--

tools.py 
  |--CreateDataSet Funtion
```

## License

[The MIT License](https://github.com/bravotty/DataMining_AritificialNeuralNetworks/blob/master/LICENSE)
