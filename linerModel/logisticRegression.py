import numpy as np
from subFunctions import activationFunc

class logistic:
    def __init__(self):
        pass

    def learn(self,x,y):
        self.paramInit(np.array(x),np.array(y))
        pass

    def train(self,x,y):
        pass

    def loss(self,x):
        pass

    def paramInit(self,x:np.array,y:np.array):
        self.sample,self.feature = x.shape
        self.y = np.identity(len(np.unique(y)))[y]
        self.x = x
        self.w = np.random.randn(self.feature,len(np.unique(y)))
        self.b = np.random.randn(self.sample,len(np.unique(y)))

    def calc(self):
        return activationFunc.softmax(np.dot(self.x,self.w) + self.b)
