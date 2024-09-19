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
S = np.array([[0,0],[0,1],[1,0],[1,1]])
T = np.array([[0],[1],[1],[1]])

res.train(S,T,10)
res.eval(S)
print(res.w)

# absc = np.arange(-1,2.1,0.1)
# ord = (th-absc*w[0])/w[1]
# plt.scatter(x[:,0],x[:,1])
# plt.plot(absc,ord)
# plt.show()