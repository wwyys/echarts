import numpy as np
import matplotlib.pyplot as plt
def loadData(files):
    x,y=[],[]
    lineArr=[]
    #打开文件————读取每行文件——以逗号为特征分开，存到lineArr
    with open(files) as fileIn:
        [lineArr.append(line.strip().split(',')) for line in fileIn.readlines()]
    np.random.shuffle(lineArr)   #随机打乱数据集、
    for line in lineArr:
        x.append([1.0,float(line[0]),float(line[1])])
        y.append(float(line[2]))
    return np.mat(x),np.mat(y).T

def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))
#梯度上升法    和下降法相同   抛物线开口向下
def logicRegression(x,y,alpha,iter):
    numSamples,numFeatures=np.shape(x)
    print(np.shape(x))
    weights=np.ones((numFeatures,1))
    for i in range(iter):
        fx=x*weights
        hx=sigmoid(fx)
        weights=weights+alpha*x.T*(y-hx)
    return weights
#随机梯度上升法    和下降法相同   抛物线开口向下
def stochlogicRegression(x,y,alpha,iter):
    numSamples,numFeatures=np.shape(x)
    print(np.shape(x))
    weights=np.ones((numFeatures,1))
    for i in range(iter):
        for j in range(numSamples):
            fx=x[j,:]*weights
            hx=sigmoid(fx)
            weights=weights+alpha*x[j,:].T*(y[j,:]-hx)
    return weights
def showLogicRegression(weights,x,y):
    numSamples, numFeatures = np.shape(x)
    for i in range(numSamples):
        if int(y[i,0]) == 0:
            plt.plot(x[i,1],x[i,2],'om')
        elif int(y[i,0]) == 1:
            plt.plot(x[i, 1], x[i, 2], 'ob')
    xa1=min(x[:,1])[0,0]
    xb1=max(x[:,1])[0,0]
    xa2=-((weights[0]+weights[1]*xa1)/weights[2]).tolist()[0][0]
    xb2=-((weights[0]+weights[1]*xb1)/weights[2]).tolist()[0][0]
    plt.plot([xa1,xb1],[xa2,xb2],'#FB4A42')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()
#在迭代好的回归函数之下训练的准确度
def accLogicRegression(weights,x,y):
    numSamples, numFeatures = np.shape(x)
    accuracy=0.0
    for i in range(numSamples):
        predict=sigmoid(x[i,:]*weights)[0,0]>0.5
        if predict == bool(y[i,0]):
            accuracy+=1
    print('准确率{0}%'.format(accuracy/numSamples*100))



if __name__ == '__main__':
    #sigmoid图
    x=np.linspace(-20,20,500)
    y=sigmoid(x)
    plt.plot(x,y)
    plt.show()

    x,y=loadData('./data/data.csv')
    plt.plot(x,y)
    plt.show()

    weights=logicRegression(x,y,alpha=0.01,iter=500)
    #showLogicRegression(weights, x, y)

    weights=stochlogicRegression(x, y, alpha=0.02, iter=500)
    showLogicRegression(weights, x, y)

    accLogicRegression(weights, x, y)

