# -*- coding: UTF-8 -*-
import numpy as np
import operator

from ChooseBestFeature考试 import createDataSet

"""
函数说明:kNN算法,分类器

Parameters:
    inX - 用于分类的数据(测试集)
    dataSet - 用于训练的数据(训练集)
    labes - 分类标签
    k - kNN算法参数,选择距离最小的k个点
Returns:
    sortedClassCount[0][0] - 分类结果


"""
def classify0(inX, dataSet, labels, k):  
   
   
    dataSetSize = dataSet.shape[0]       #chapter0  numpy.py函数shape
    
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    #print("tile函数",np.tile(inX, (dataSetSize, 1)))
    #print("diffMat", diffMat)
   
    sqDiffMat = diffMat**2
    
    sqDistances = sqDiffMat.sum(axis=1)
   
    distances = sqDistances**0.5
    
    
    sortedDistIndices = distances.argsort()
    

    classCount = {}
    
    for i in range(k):
        
        voteIlabel = labels[sortedDistIndices[i]]  
       
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
      
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
  
    return sortedClassCount[0][0]  



if __name__ == '__main__':
  
    group, labels = createDataSet()
   
    print("数据样本group：",group)
    print("特征标签labels：",labels)

    test = [101, 20]
    
    test_class = classify0(test, group, labels, 3)  
   
    print("test_class:",test_class)