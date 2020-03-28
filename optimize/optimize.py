import numpy as np
import copy
from optimize import subFunc

# 関数を最小化する

# 準ニュートン法
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