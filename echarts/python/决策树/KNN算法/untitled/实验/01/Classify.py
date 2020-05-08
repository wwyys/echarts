import numpy as np
import operator
def createDataSet():
    group=np.array([[1,101],[5,89],[108,5],[115,8]])
    labels=['爱情片','爱情片','动作片','动作片']
    return group,labels


if __name__ == '__main__':
    #创建数据集
    group,labels=createDataSet()
    #打印数据集
    print("数据样本group:",group)
    print("特征标签labels:",labels)


#KNN分类器
def classify0(inx,dataSet,labels,k):
    dataSetSize=dataSet.shape[0]
    print("dataSet的行数:",dataSetSize)
    diffMat=np.tile(inx,(dataSetSize,1))-dataSet
    print("tile函数",np.tile(inx,(dataSetSize,1)))
    print("diffMat",diffMat)
    sqDiffMat=diffMat**2
    sqDistances=sqDiffMat.sum(axis=1)
    distances=sqDistances**0.5
    print("distances距离",distances)
    sortedDistIndices=distances.argsort()
    print("sortedDistIndices0索引值",sortedDistIndices)
    # KNN算法
def ClassCount():
    ClassCount={}
    print("classCount0",classCount)
    print("labels",labels)
    print("labels2",labels[2])
    for i in range(k):
        voteIlabel=labels[sortedDistIndices[i]]
        classCount[voteIlabel]=classCount.get(voteIlabel,0)+1
        print("classCount[voteIlabel]",classCount[voteIlabel])
    sortedClassCount=sorted(classCount.items(),key=operator.itemgette(1),)
    print("classClount2",sortedClassCount)
    return sortedClassCount[0][0]

if __name__ == '__main__':
    #创建数据库
    group,labels=createDataSet()
    #打印数据
    #print("数据样本group:",group)
    #print("特征标签labels:",labels)
    test=[101,20]
    test_class=classify0(test,group,labels,3)
    print("test_class:",test_class)



'''如何实现像百度页面那样的算法应用？

'''
