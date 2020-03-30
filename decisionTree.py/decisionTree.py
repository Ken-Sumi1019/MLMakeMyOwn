import numpy as np

class DecisionTree:
    def __init__(self,maxdepth,minsize):
        self.maxdepth = maxdepth
        self.minsize = minsize

    """学習する"""
    def learn(self,data,label):
        self.data = np.array(data)
        self.label = np.array(label)
        self.categorys = np.unique(self.label)

    """木を構築する"""
    def makeTree(self):
        pass

    """頂点を追加する"""
    def addNode(self):
        pass

    """ジニ不純度を計算する"""
    def giniScore(self,index):
        n = len(index)
        return 1.0 - sum([np.count_nonzero(self.label[index] == c) / n for c in self.categorys])

    """最良の閾値を見つける"""
    def bestThreshold(self):
        pass

    """指定した閾値に合わせてグループを分ける"""
    def splitGroup(self):
        pass
