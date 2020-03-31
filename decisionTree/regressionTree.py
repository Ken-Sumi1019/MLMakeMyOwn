import numpy as np
from decisionTree import decisionTree

class regressionTree(decisionTree.DecisionTree):
    def lossScore(self,index):
        return sum((self.label[index] - np.mean(self.label[index])) ** 2) / len(index)

    def decisionVal(self,indexes):
        return np.mean(self.label[indexes])
