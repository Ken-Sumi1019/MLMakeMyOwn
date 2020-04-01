import numpy as np
from subFunctions import activationFunc

class logistic:
    def __init__(self):
        pass

    # 使う時に呼び出す学習をする関数
    def train(self,x,y,alpha = 1,Epsilon = 0.001):
        self.alpha = alpha
        self.Epcilon = Epsilon
        self.paramInit(np.array(x),np.array(y))
        self.optimize()
    
    # 使う時に呼び出す予測関数
    def predict(self,x):
        a = self.calc(np.array(x))
        ans = [0]*len(x)
        for i in range(len(x)):
            index = -1;v = -1
            for j in range(self.feature):
                if v < a[i][j]:
                    index = j;v = a[i][j]
            ans[i] = index
        return ans

    # 再急降下法を行って重みを最適化
    def optimize(self):
        lossb = 10**9
        while True:
            dw,db = self.lossDiff()
            self.w += self.alpha * dw
            self.b += self.alpha * db
            l = self.loss(self.x)
            ll = np.sqrt(np.dot(l.T,l))
            if lossb - ll < self.Epcilon:
                break
            lossb = ll

    # クロスエントロピー
    def loss(self,x):
        a = self.calc(x)
        b = self.y*activationFunc.llog(a) + (1 - self.y)*activationFunc.llog(1 - a)
        return np.sum(b,axis=0)[:,None]

    # 微分
    def lossDiff(self):
        a = self.y - self.calc(self.x)
        dw = np.dot(self.x.T,a) / self.sample
        db = np.dot(np.ones((1,self.sample)),a) / self.sample
        return dw,db

    # 新しいデータが渡された時の重みなどの初期化を行う関数
    def paramInit(self,x:np.array,y:np.array):
        self.sample,self.feature = x.shape
        self.y = np.identity(len(np.unique(y)))[y]
        self.x = x
        self.w = np.random.randn(self.feature,len(np.unique(y)))
        self.b = np.random.randn(1,len(np.unique(y)))

    # 計算をする
    def calc(self,x):
        return activationFunc.softmax(np.dot(x,self.w) + self.b)
