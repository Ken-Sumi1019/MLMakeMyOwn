
class DecisionTree:
    def __init__(self,maxdepth,minsize):
        self.maxdepth = maxdepth
        self.minsize = minsize

    """学習する"""
    def learn(self,x,y):
        pass

    """木を構築する"""
    def makeTree(self):
        pass

    """ジニ不純度を計算する"""
    def giniScore(self):
        pass

    """最良の閾値を見つける"""
    def bestThreshold(self):
        pass

    """指定した閾値に合わせてグループを分ける"""
    def splitGroup(self):
        pass
