import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
class lin_regress:
    def __init__(self,n):
        self.size=n
        self.W=np.random.random((n,1))
        self.b=np.random.random()
    def forward(self,x):
        y_hat=x.dot(self.W)+self.b
        return y_hat
    def lossFun(self,x,y):
        y_hat=self.forward(x)
        loss=np.mean((y-y_hat)**2)
        return loss
    def grad(self,x,y):
        gradW=np.zeros((self.size,1))
        for j in range(self.size):
            gradW[j,0]=np.mean((x.dot(self.W)+self.b-y)*x[:,j])
        grad_b=np.mean((x.dot(self.W)+self.b-y))
        return gradW,grad_b
    def update(self,grad_W,grad_b,lr):
        self.W=self.W-lr*grad_W
        self.b=self.b-lr*grad_b

if __name__=="__main__":
    A=pd.read_csv("data.csv",index_col=0)
    A1=np.array(A.values)
    for j in range(13):
        ma=np.max(A1[:,j])
        mi=np.min(A1[:,j])
        A1[:,j]=(A1[:,j]-mi)/(ma-mi)
    np.random.shuffle(A1)
    # Xtrain=A1[0:400,0:13]
    # Ytrain=A1[0:400,13].reshape(-1,1)
    Xtest=A1[400::,0:13]
    Ytest=A1[400::,13].reshape(-1,1)
    model=lin_regress(13)
    ori_loss=model.lossFun(Xtest,Ytest)
    print(f"训练前误差为{ori_loss}")
    loss=[]
    lr=0.01
    batchsize=10
    mini_batch=[A1[k:k+batchsize] for k in range(0,400,batchsize)]
    for epoch in range(20):
        for iter,batch in enumerate(mini_batch):
            Xtrain = batch[:, 0:13]
            Ytrain = batch[0:400, 13].reshape(-1, 1)
            L=model.lossFun(Xtrain,Ytrain)
            loss.append(L)
            gW,g_b=model.grad(Xtrain,Ytrain)
            model.update(gW,g_b,lr)
    y_hat=model.forward(Xtest)
    plt.plot(loss)
    plt.show()
    # plt.plot(y_hat)
    # plt.plot(Ytest)
    # plt.show()
    loss_test=model.lossFun(Xtest,Ytest)
    print(f"误差为{loss_test}")