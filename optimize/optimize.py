import numpy as np
import copy
from optimize import subFunc

# 関数を最小化する

"""
準ニュートン法
いつの日か、準ニュートン法を使いこなせるといいなと思っている。
dim = 最小化したい関数の引数の次元数
func = 目的関数
diff = 目的関数の一回微分
B = 指定しなければ単位行列
x = 探索の初期値。指定しなければ全部0
"""
def quasiNewton(dim,func,diff,B = None,x = None):
    Epsilon = 0.0000001
    if B == None:
        B = np.eye(dim)
    if x == None:
        x = np.array([[0] for i in range(dim)])
    d = 0;xbfore = 0
    while True:
        d = -np.dot(np.linalg.pinv(B),diff(x))
        print("#")
        alpha = subFunc.wolf(func,diff,x,d,10,0.5,0.9,0.9)
        print(1)
        xbfore = x
        x = xbfore + alpha * d
        s = xbfore - x
        if np.sqrt(np.dot(s.T,s)) < Epsilon:return x
        y = diff(x) - diff(xbfore)
        B = B - (np.dot(s.T,s) * np.dot(B,B)) / np.dot(np.dot(s.T,B),s) + np.dot(y,y.T) / np.dot(s.T,y)

"""
勾配降下法
"""
def gradient(dim,diff,alpha,func = None,x = None):
    if x == None:
        x = np.array([[0] for i in range(dim)])
    Epsilon = 0.00001
    xbefore = 0
    while True:
        xbefore = x
        x = x - alpha * diff(x)
        c = x - xbefore
        
        c = np.dot(c.T,c)
        if np.sqrt(c) < Epsilon:
            return x