import numpy as np

class linerRegression:
    def __init__(self):
        pass

    def learn(self,x,y):
        matX = np.array(x)
        theta = np.linalg.inv(np.dot(matX.T,matX))
        theta = np.dot(np.dot(theta,matX.T),np.array(y))
        self.theta = theta
        self.e = np.array(y) - np.dot(matX,self.theta)

    def predict(self,x):
        return np.dot(np.array(x),self.theta) + self.e