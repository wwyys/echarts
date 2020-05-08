from math import log

'''

函数说明：计算给定数据集的经验熵（香农熵）
Parameters：
      无
Returns:
     dataSet  -- 数据集
     labels   -- 分类属性
Author:zxl
Modify:20200310

'''
def createDataSet():
    dataSet=[[0,0,0,0,'no'],
             [0,0,0,1,'no'],
             [0,1,0,1,'yes'],
             [0,1,1,0,'yes'],
             [0,0,0,0,'no'],
             [1,0,0,0,'no'],
             [1,0,0,1,'no'],
             [1,1,1,1,'yes'],
             [1,0,1,2,'yes'],
             [1,0,1,2,'yes'],
             [2,0,1,2,'yes'],
             [2,0,1,1,'yes'],
             [2,1,0,1,'yes'],
             [2,1,0,2,'yes'],
             [2,0,0,0,'no']]
    labels=['不放贷','放贷']
    return dataSet,labels

def calcShannonEnt(dataSet):
    numEntires=len(dataSet)
    labelCounts={}
    for featVec in dataSet:
        currentLabel=featVec[-1]
        print("labelCounts.keys",labelCounts.keys())
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel]=0
        labelCounts[currentLabel]+=1
        print("labelCounts",labelCounts)
    shannonEnt=0.0
    for key in labelCounts:
        print("labelCountskey",key)
        prob=float(labelCounts[key])/numEntires
        shannonEnt-=prob*log(prob,2)
    return shannonEnt
if __name__ == '__main__':
    dataSet,features=createDataSet()
    print(dataSet)
    print(calcShannonEnt(dataSet))
