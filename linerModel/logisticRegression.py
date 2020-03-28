import numpy as np

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

