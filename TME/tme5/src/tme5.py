import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mltools import plot_data, plot_frontiere, make_grid, gen_arti

def perceptron_loss(w,x,y):
    return np.max(0,-y*x@w)

def perceptron_grad(w,x,y):
    return np.max(0,-y*x)

class Lineaire(object):
    def __init__(self,loss=perceptron_loss,loss_g=perceptron_grad,max_iter=100,eps=0.01):
        self.max_iter, self.eps = max_iter,eps
        self.w = None
        self.loss,self.loss_g = loss,loss_g
        
    def fit(self,datax,datay):
        self.w=np.random.rand(datax.shape[1])
        w_list=[self.w]
        loss_values=[self.loss(self.w,datax,datay).sum()]
        for i in range(self.max_iter):
            self.w=self.w-self.eps*self.loss_g(self.w,datax,datay)
            w_list.append(self.w),loss_values.append(self.loss(self.w,datax,datay).sum())
        return w_list,loss_values

    def predict(self,datax):
        return np.sign(datax@self.w)

    def score(self,datax,datay):
        return np.mean(self.predict(datax)==datay)

def load_usps(fn):
    with open(fn,"r") as f:
        f.readline()
        data = [[float(x) for x in l.split()] for l in f if len(l.split())>2]
    tmp=np.array(data)
    return tmp[:,1:],tmp[:,0].astype(int)

def get_usps(l,datax,datay):
    if type(l)!=list:
        resx = datax[datay==l,:]
        resy = datay[datay==l]
        return resx,resy
    tmp =   list(zip(*[get_usps(i,datax,datay) for i in l]))
    tmpx,tmpy = np.vstack(tmp[0]),np.hstack(tmp[1])
    return tmpx,tmpy

def show_usps(data):
    plt.imshow(data.reshape((16,16)),interpolation="nearest",cmap="gray")



if __name__ =="__main__":
    uspsdatatrain = "C:/Users/hatem/OneDrive/Documents/Programmation/M1-S2/ML/TME/tme5/data/USPS_test.txt"
    uspsdatatest = "C:/Users/hatem/OneDrive/Documents/Programmation/M1-S2/ML/TME/tme5/data/USPS_train.txt"
    alltrainx,alltrainy = load_usps(uspsdatatrain)
    alltestx,alltesty = load_usps(uspsdatatest)
    neg = 5
    pos = 6
    datax,datay = get_usps([neg,pos],alltrainx,alltrainy)
    testx,testy = get_usps([neg,pos],alltestx,alltesty)

Mobutu=Lineaire()
Mobutu.fit(datax,datay)
acc=Mobutu.score(testx,testy)
print(uspsdatatest)