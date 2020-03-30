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
        self.tree = []

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
    def bestThreshold(self,indexs):
        bestScore = 100
        bestRight = [];bestLeft = []
        bestIdx = 0;bestFeature = 0
        for i in range(np.shape(self.data)[1]):
            for j in indexs:
                l,r = self.splitGroup(j,i,indexs)
                lgini = self.giniScore(l);rgini = self.giniScore(r)
                if bestScore > lgini + rgini:
                    bestIdx = j;bestFeature = i
                    bestRight = r;bestLeft = l
                    bestScore = lgini + rgini
        result = {"index" : bestIdx,"feature" : bestFeature,
                  "right" : bestRight,"left" : bestLeft}
        return result

    """指定した閾値に合わせてグループを分ける"""
    def splitGroup(self,index,feature,indexs):
        right = [];left = []
        for i in indexs:
            if self.data[i][feature] < self.data[index][feature]:
                left.append(i)
            else:
                right.append(i)
        return left,right
