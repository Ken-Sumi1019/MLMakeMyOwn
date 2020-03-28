import numpy as np

# wolf条件を満たすステップ幅を見つける
def wolf(func,diff,x,d,amax,c1,c2,u):
    a = 0
    while a < 100:
        alpha = np.power(u,a) * amax
        check = False
        if func(x + alpha*d) <= func(x) + c1*alpha*np.dot(diff(x).T,d):
            check = True
        if check and c2*np.dot(diff(x).T,d) <= np.dot(diff(x + alpha*d).T,d):
            return alpha
        a += 1
    return 0