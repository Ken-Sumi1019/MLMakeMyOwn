import numpy as np

class treeNode:
    def __init__(self):
        self.right = None
        self.left = None
        self.feature = None
        self.idx = None

class DecisionTree:
    def __init__(self,maxdepth,minsize):
        self.maxdepth = maxdepth
        self.minsize = minsize

    """学習する"""
    def learn(self,data,label):
        self.data = np.array(data)
        self.label = np.array(label)
        self.categorys = np.unique(self.label)
        self.tree = {}
        self.makeTree()

    """木を構築する"""
    def makeTree(self):
        allIdx = list(range(len(self.data)))
        self.tree = self.addNode(1,allIdx,self.giniScore(allIdx))

    """頂点を追加する"""
    def addNode(self,depth,indexes,gini):
        node = self.bestThreshold(indexes)
        tree = {"gini" : gini,"feature":node["feature"],"val":node["val"]}
        if self.maxdepth <= depth:
            tree["right"] = self.leafNode(node["right"],node["rGini"])
            tree["left"] = self.leafNode(node["left"],node["lGini"])
            return tree
        if len(node["right"]) < self.minsize:
            tree["right"] = self.leafNode(node["right"],node["rGini"])
        else:
            tree["right"] = self.addNode(depth + 1,node["right"],node["rGini"])
        if len(node["left"]) < self.minsize:
            tree["left"] = self.leafNode(node["left"],node["lGini"])
        else:
            tree["left"] = self.addNode(depth + 1,node["left"],node["lGini"])
        return tree

    """葉ノードを作成"""
    def leafNode(self,indexes,gini):
        result = {"gini" : gini}
        val,count = np.unique(self.label[indexes],return_counts=True)
        pre = val[np.argmax(count)]
        result["predictVal"] = pre
        return result

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
                if len(l) == 0 or len(r) == 0:continue
                lgini = self.giniScore(l);rgini = self.giniScore(r)
                if bestScore > lgini + rgini:
                    bestIdx = j;bestFeature = i
                    bestRight = r;bestLeft = l
                    bestScore = lgini + rgini
        result = {"val" : self.data[bestIdx][bestFeature],"feature" : bestFeature,
                  "right" : bestRight,"left" : bestLeft,
                  "rGini" : rgini,"lGini" : lgini}
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

    """木を移動して予測値を探し出す"""
    def DFS_predict(self,data,node):
        if "predictVal" in node:
            return node["predictVal"]
        f = node["feature"];v = node["val"]
        if data[f] < v:
            return self.DFS_predict(data,node["left"])
        else:
            return self.DFS_predict(data,node["right"])

    def predict(self,x):
        ans = [0]*len(x)
        for i in range(len(x)):
            ans[i] = self.DFS_predict(x[i],self.tree)
        return ans