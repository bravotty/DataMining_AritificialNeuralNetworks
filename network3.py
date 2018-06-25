from  __future__ import division
import numpy as np
import random

def sigmoid(x):
    return [1/(1+np.math.exp(-i)) for i in x] 

#
def load_data():
    dataset=np.loadtxt('./fruit.txt',skiprows=1,usecols=[0,3,4,5,6])
    np.random.shuffle(dataset)
    count=len(dataset) 
    devide=10
    test_x=dataset[0:devide,1:]
    train_x=dataset[devide:,1:]
    test_y=dataset[0:devide,0]
    train_y=dataset[devide:,0]
    print test_x
    print test_y
    print train_x
    print train_y
    return train_x,train_y,test_x,test_y


class ANN:
    def __init__(self):
        self.layers=None   
        self.x=None        
        self.y=None        
        self.l=None        
        self.mean=None     
        self.std=None      
        self.weight=[]     
        self.bias=[]       

    
    def normalization(self,x):
        data=np.array(x).T
        self.mean=[np.mean(i) for i in data]
        self.std=[np.std(i) for i in data]
        for d in range(len(data)):
            data[d]=(data[d]-self.mean[d])/self.std[d]
        return data.T
    
    
    def oneHot(self,y):
        res_y=[]
        for i in y:
            temp=[0]*self.layers[-1];
            temp[int(i-1)]=1
            res_y.append(temp)
        return res_y

    def feedForword(self,input_data):
        d=input_data
        layer_output=[] 
        layer_input=[]  
        for layer in range(len(self.layers)-1):
            layer_input.append(d)
            d=np.dot(d,self.weight[layer].T)+self.bias[layer].T
            d=[sigmoid(i) for i in d]
            layer_output.append(d)
        return layer_input, layer_output

    def backForword(self,input_data,output_data,y):
        layer_weight_count=len(self.layers)-1    
        for idx,output,in_put in zip(range(layer_weight_count)[::-1],output_data[::-1],input_data[::-1]):
            output=np.array(output)
            in_put=np.array(in_put)
            y=np.array(y)
            
            if idx==layer_weight_count-1: 
                err=output*(1-output)*(y-output)
            
            else :
                err=np.dot(err,self.weight[idx].T)
                err=output*(1-output)*err
            
            for idn in range(len(self.weight[idx])):
                self.weight[idx][idn]=self.weight[idx][idn]+self.l*err[0][idn]*in_put
                self.bias[idx]=(self.bias[idx].T+self.l*err).T

    
    def evaluate(self,x,y):
        correct=0;
        res=self.pred(x)
        for v_y,r_y in zip(y,res):
            if v_y==r_y:
                correct+=1
        return correct/len(y)

    
    def pred(self,x):
        res=[]
    
        x=np.array(x).T
        for i in range(len(x)):
            x[i]=(x[i]-self.mean[i])/self.std[i]
        x=x.T
        for sample in x:
            d=sample
            for layer in range(len(self.layers)-1):
                d=np.dot(d,self.weight[layer].T)+self.bias[layer].T
                d=[sigmoid(i) for i in d]
            res.append(np.argmax(d)+1)
        return res

    
    
    def buildModle(self,layers,learn_rate,x,y,frequency):
        self.layers=layers
        self.l=learn_rate
        self.x=self.normalization(x)
        self.y=self.oneHot(y)
        self.frequency=frequency
        for i,j in layers[1:],layers[:-1]:
            
            layer_weight=np.random.uniform(-0.2,0.2,[i,j])
            layer_bias=np.random.uniform(-0.2,0.2,[i,1])
            self.weight.append(layer_weight)
            self.bias.append(layer_bias)
        self.bias=np.array(self.bias)
        for i in range(frequency):
            for train_x,train_y in zip(self.x,self.y):
                layer_input ,layer_output=self.feedForword(train_x)
                self.backForword(layer_input,layer_output,train_y)

def main():
    train_x,train_y,test_x,test_y=load_data()
    clt=ANN()
    clt.buildModle([4,4,4],0.5,train_x,train_y,1000)
    accuracy=clt.evaluate(train_x,train_y)
    print(accuracy)

if __name__=='__main__':
    main()
