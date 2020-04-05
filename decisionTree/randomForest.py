import numpy as np
from . import decisionTree

class Classification:
    def __init__(self,n_trees,maxdepth,minsize,select_seed = 0):
        self.n_trees = n_trees
        self.forest = [None] * n_trees
        self.maxdepth = maxdepth
        self.minsize = minsize
        np.random.seed(seed = select_seed)

    def train(self,x,y):
        self.data = np.array(x)
        self.label = np.array(y)
        self.makeForest()

    def makeForest(self):
        for i in range(self.n_trees):
            self.forest[i] = self.makeTree()

    def makeTree(self):
        tree = decisionTree.ClassificationTree(self.maxdepth,self.minsize)
        n = len(self.data)
        index = np.random.randint(0, n,n )
        tree.train(self.data[index],
                   self.label[index])
        return tree

    def predict(self,x):
        result = [None]*self.n_trees
        for i in range(self.n_trees):
            result[i] = self.forest[i].predict(x)
        result = np.array(result).T
        ans = np.array([0]*len(x))
        for i in range(len(x)):
            val,count = np.unique(result[i],return_counts=True)
            ans[i] = val[np.argmax(count)]
        return ans

    