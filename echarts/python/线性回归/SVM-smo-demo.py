import matplotlib.pyplot as plt
import numpy as  np
import random
from scipy import stats

'''
函数功能：1.读取数据
简化版SMO算法
伪代码：
    创建一个α 赋初值为0
    当迭代时，迭代次数小于最大跌代数(Ω外循环)
      若可继续优化，进行内循环
            随机选择一个向量
            同时优化这两个向量
            不能被优化，退出内循环
    若所有向量都无优化，增加迭代次数 继续下一次迭代
'''
'''
data
'''
def loadDataSet(fileName):
    dataMat=[]
    labelMat=[]
    fr=open(fileName)
    for line in fr.readlines():
        lineArr=line.strip().split("\t")
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

'''
随机选择alpha

'''
def selectJrand(i,m):
    j=i
    while(j==i):
        j=int(random.uniform(0,m))
    return j


'''
修剪alpha
约束范围  L<alpha<=h 更新所有alpha值
'''
def clipAlpha(aj,H,L):
    if aj>H:
        aj=H
    if L>aj:
        aj=L
    return aj
'''
函数功能：1.读取数据
简化版SMO算法
伪代码：
    创建一个α 赋初值为0
    当迭代时，迭代次数小于最大跌代数(Ω外循环)
      若可继续优化，进行内循环
            随机选择一个向量
            同时优化这两个向量
            不能被优化，退出内循环
    若所有向量都无优化，增加迭代次数 继续下一次迭代
dataMatIm数据矩阵
classLabels标签
c惩罚参数，toler容错率，maxIter最大迭代次数
'''
def smoSimple(dataMatIn,classLabels,C,toler,maxIter):
    dataMatrix=np.mat(dataMatIn)   #将list转为numpy矩阵matrix,mat
    labelMat=np.mat(classLabels).transpose() #转置
    b=0
    m,n=np.shape(dataMatrix)   #特征数据矩阵的行和列
    #所有alphas初始化为0
    #根据KTT条件，在最终的分类器函数中，每一个训练集样本都对应着一个系数
    #但是只能支持向量样本所对应的系数才有非零值，而其他的系数都为零
    alphas=np.mat(np.zeros((m,1)))#初始化alpha是否已经进行最优化处理
    iter_num=0
    while(iter_num<maxIter):
        alphaPairChanged=0#用来记录alpha是否已经进行最优化处理
        for i in range(m):#遍历m行所有样本 外循环
            #计算预测的类别 计算超平面f(x)=w.T*x+b alpha,LabelMat数值对应相乘的到+1、-1两类对应的alpha值
            #与dataMatrix*dataMatrix[i,:].T  所有样本数据点对应于i行向量到边界面L和H的距离
            fXi=float(np.multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[i,:].T))+b
            #计算预测值与实际值的误差
            Ei=fXi-float(labelMat[i])

            if((labelMat[i]*Ei)<-toler) and (alphas[i]<C) or ((labelMat[i]*Ei>toler)and (alphas[i]>0)):
                #判断alpha是否可以更改进行优化过程
                #如果误差很大，就可以对alpha进行优化 0<=alphas<=C labelMat[i]*Ei大于正的边界面，小于负的边界面误差容错率
                #意思是误差太大了，也是距离边界面太远了所以还得进行优化
                j=selectJrand(i,m)
                #在m行样本数据中随机选取除了i行以外的第二个alphas
                #进行再次计算预测结果也就是一行数据一行数据的找到边界上的支持向量
                fXj=float(np.multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[j,:].T))+b
                Ej=fXj-float(labelMat[j])    #计算预测值与实际值误差
                alphaIold=alphas[i].copy()   #python为应用型返回，所以要开辟新的空间存储
                alphaJold=alphas[j].copy()
                #保证新计算的alpha在0与C之间
                if (labelMat[i]!=labelMat[j]):   #类别不等
                    L=max(0,alphas[j]-alphas[i])   #下边界面>0
                    H=min(C,C+alphas[j]-alphas[i])  #在上边界面<c
                else:
                    L=max(0,alphas[j]+alphas[i]-C) #下边界面>0
                    H=min(C,alphas[j]+alphas[j]) #在上边界面<c
            if L==H:
                print("L=H") #相等重新迭代
                continue
            #alpha[j]的最优修改值
            #eta为优化修正量
            eta=2.0*dataMatrix[i,:]*dataMatrix[j,:].T-dataMatrix[j,:]*dataMatrix[j,:].T-dataMatrix[i,:]*dataMatrix[i,:].T
            if eta>=0:
                print("eta>=0")
                continue
            alphas[j]-=labelMat[j]*(Ei-Ej)/eta  #调优alphas[j]
            alphas[j]=clipAlpha(alphas[j],H,L) #用于调整alphas[j]大于H或者小于L的alpha值，使之等于H或者L
            if (abs(alphas[j]-alphaJold),0.0001):
                print("你调整的太小了")
                continue
            #否则对alphas[i]进行修正，与labelMat[j]向量方向相反，即一个增加，一个减小，并利用j的变化值来修正
            alphas[i]+=labelMat[j]*labelMat[i]*(alphaJold-alphas[j])

            #给出对应alphas对应的常数项值b
            b1=b-Ei-labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[i,:].T-labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[i,:]*dataMatrix[j,:].T
            b2 = b - Ej - labelMat[i] * (alphas[i] - alphaIold) * dataMatrix[i, :] * dataMatrix[j, :].T - labelMat[j] * (alphas[j] - alphaJold) * dataMatrix[j, :] * dataMatrix[j, :].T
            print('b1=',b1)
            print('b2=',b2)
            if (0<alphas[i]) and (C > alphas[i]):
                b=b1
            elif (0< alphas[j]) and (C>alphas[j]):
                b=b2
            else:
                b=(b1+b2)/2.0
            alphaPairsChanged +=1
            print('iter:%d i:%d,pairs changed %d'%(iter_num,i,alphaPairChanged))
        if (alphaPairChanged==0):
            iter_num +=1
        else:
            iter_num=0
        print('iter:%d'%iter_num)
    return b,alphas


if __name__=="__main__":
    dataMat,labelMat=loadDataSet("./data/testSet2.txt")
    showDataSet(dataMat, labelMat)
    print(labelMat)
    dataMatix=np.mat(dataMat)
    m,n=np.shape(dataMatix)
    alphas=np.mat(np.zeros((m,1)))
    print('alphas',alphas)
    b,alphas=smoSimple(dataMat,labelMat,0.6,0.001,40)
    print('b',b)