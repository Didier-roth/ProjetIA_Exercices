import numpy as np
import matplotlib.pyplot as plt

class Nnet:
    def __init__(self,w,th):
        self.th = th
        self.w = np.append(w,th)
        self.alpha = 0.5

    def eval(self,x):
        print("--------- eval ---------")
        for k in range(len(x)):
            inp = np.append(x[k],-1)
            a = inp @ self.w
            o = (a>=0)*1
            print("entrée :",inp)
            print("sortie :",o)
            print("poids :",self.w)
        print("------- fin eval -------")


    def train(self,S,T,epoch):
        for i in range (epoch):
            for k in range(len(S)):
                V = np.append(S[k],-1)
                a = V @ self.w
                z = (a>=0)*1
                #regle d'apprentissage
                self.w = self.w + self.alpha*(T[k]-z)*V




w = np.array([0,0])
th = 0
res = Nnet(w,th)
S = np.array([[2,5],[3,5],[4,4],[3,3],[2,2],[4,2],[3.88,7.3],[4.7,6.58],[5.88,5.48],[7,6],[6.8,7.34],[5.62,8.22]])
T = np.array([[0],[0],[0],[0],[0],[0],[1],[1],[1],[1],[1],[1]])

res.train(S,T,100)
res.eval(S)
print(res.w)

absc = np.arange(1,7.5,0.1)
ord = (res.w[2]-absc*res.w[0])/res.w[1]
plt.scatter(S[:,0],S[:,1])
plt.plot(absc,ord)
plt.ylim(top=10, bottom = 0)
plt.xlim(-4,10)
plt.show()