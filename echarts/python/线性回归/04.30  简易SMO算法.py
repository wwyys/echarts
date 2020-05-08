import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

"""
函数功能：1.读取数据
简化版SMO算法
   伪代码：创建一个alpha  赋初值为0 当迭代的时候，迭代次数小于最大迭代数（W外循环）时，
   如果可以，对其进行内循环（即优化）。
       随机选择一个外循环（向量），同时优化这两个向量
       如果不能被优化，则退出内循环
    如果所有向量均未被优化。则增加迭代次数  继续下一循环
"""
"""
data
"""
def loadDataSet(fileName):
    dataMat=[]
    labelMat=[]
    fr=open(fileName)
    for line in fr.readlines():
        lineArr=line.strip().split('\t')   #分割数据
        dataMat.append([float(lineArr[0]),float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat,labelMat

def showDataSet(dataMat,labelMat):
    data_plus=[]
    data_minus=[]
    for i in range(len(dataMat)):
        if labelMat[i]>0:
            data_plus.append(dataMat[i])
        else:
            data_minus.append(dataMat[i])
    data_plus_np=np.array(data_plus)
    data_minus_np=np.array(data_minus)
    plt.scatter(np.transpose(data_plus_np)[0],np.transpose(data_plus_np)[1])
    plt.scatter(np.transpose(data_minus_np)[0], np.transpose(data_minus_np)[1])
    plt.show()
"""
随机选择A
i 对应A
m  对应A的个数
"""
import random
def selectJrand(i,m):
    j=i
    while(j==i):
        j=int(random.uniform(0,m))
    return j

"""
修剪A
约束范围   L<=a<=h  更新所有A值

"""
def clipAlpha(aj,H,L):
    if aj>H:
        aj=H
    if L>aj:
        aj=L
    return aj

"""
SMO
伪代码：创建一个alpha  赋初值为0 当迭代的时候，迭代次数小于最大迭代数（W外循环）时，
   如果可以，对其进行内循环（即优化）。
       随机选择一个外循环（向量），同时优化这两个向量
       如果不能被优化，则退出内循环
    如果所有向量均未被优化。则增加迭代次数  继续下一循环
    
dataMatIn  数据矩阵
classLabels   标签
C  惩罚系数
toler   容错率
macxIter   最大迭代次数
"""
def smoSimple(dataMatIn,classLabels,C,toler,maxIter):
    dataMatrix=np.mat(dataMatIn)
    labelMat=np.mat(classLabels).transpose()
    b=0
    m,n=np.shape(dataMatrix)
    alphas=np.mat(np.zeros((m,1)))  #m行1列的初始化向量
    iter_num=0
    while(iter_num<maxIter):
        alphaPairsChanged=0
        for i in range(m):
            fxi=float(np.multiarray(alphas,labelMat).T*(dataMatrix*dataMatrix[i,:].T))+b
            #计算预测值与实际值的误差
            Ei=fxi-float(labelMat[i])
        if(((labelMat[i]*Ei)<-toler) and (alphas[i]<C)) or ((labelMat[i]*Ei)>toler)and((alphas[i])>0):
            j=selectJrand(i,m)
            fxj=float(np.multiarray(alphas,labelMat).T*(dataMatrix*dataMatrix[j,:].T))+b
            Ej=fxj-float(labelMat[j])
            alphaIold=alphas[i].aopy()
            alphaJold = alphas[j].aopy()

            if(labelMat[i]!=labelMat[j]):
                L=max(0,alphas[j]-alphas[i])
                H=min(C,C+alphas[j]-alphas[i])
            else:
                L=max(0,alphas[j]+alphas[i]-C)
                H=min(C,alphas[j]+alphas[i])
            if L==H:
                print("L=H");continue







if __name__ == '__main__':
    dataMat,labelMat=loadDataSet('./data/testSet2.txt')
    showDataSet(dataMat, labelMat)

