# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# https://doc.scrapy.org/en/latest/topics/items.html


import numpy as np
import operator


def createDataSet():
    group=np.array([[1,101],[5,89],[108,5],[115,8]])
    labels=['爱情片','爱情片','动作片','动作片']
    return group,labels

def classify0(inX,dataSet,labels,k):
    dataSetSize=dataSet.shape[0]
    print("dataSet的行数:",dataSetSize)
    diffMat=np.tile(inX,(dataSetSize,1))-dataSet
    print("tile函数:",np.tile(inX,(dataSetSize,1)))
    print("diffMat",diffMat)
    sqDiffMat=diffMat**2
    sqDistances=sqDiffMat.sum(axis=1)
    distances=sqDistances**0.5
    print("distances距离",distances)
    sortedDistIndices=distances.argsort()
    print("sortedDistIndices0索引值",sortedDistIndices)
    classCount={}
    print("classCount0",classCount)
    print("labels",labels)
    print("labels2",labels[2])
    for i in range(k):
        voteIlabel=labels[sortedDistIndices[i]]
        classCount[voteIlabel]=classCount.get(voteIlabel,0)+1
        print("classCount[voteIlabel]",classCount)

    sortedClassCount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    print("classCount2",sortedClassCount)
    return sortedClassCount[0][0]

if __name__=="__main__":
    group,labels=createDataSet()
    test=[101,20]
    test_class=classify0(test,group,labels,3)
    print("test_class:",test_class)