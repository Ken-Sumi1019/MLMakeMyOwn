import numpy as np

""" シグモイド関数 """
def sigmoid(x):
    ans = 1/(1 + np.exp(-x))
    ans = np.where((ans == np.float('inf')) & (x > 0),1,ans)
    ans = np.where((ans == np.float('inf')) & (x < 0),0,ans)
    return ans

""" ソフトマックス関数 """
def softmax(Z):
    return np.exp(Z)/np.sum(np.exp(Z), axis=1)[:, None]

""" logがうまいこといくように """
def llog(x):
    return np.where(x <= 0.0,-1e9,np.log(x))