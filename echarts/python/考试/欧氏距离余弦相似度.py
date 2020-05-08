import numpy as np
def createDataSet():
    #四组二维特征
    group=np.array([[1,101],[5,89],[108,5],[115,8]])
    #四组特征的标签
    labels=['爱情片','爱情片','动作片','动作片']
    return group,labels
    dataSize=dataSet.shape[0]
    diffMat=np.title(inX,(dataSetSize,1))-dataSet()
    sqDiffMat=diffMat**2
    #sum（）所有元素相加，sum（0）列相加，sum（1）行相加
    sqDistances=sqDiffMat.sum(axis=1)
    #开方，计算出距离
    distances=sqDistances**0.5
    sortedDistIndices=distances.argsort()
    classCount={}
    for i in range(k):
        #取出前k个元素的类别range（5）等价于range（0,5）   end：计数到end结束，但不包括end
        voteIlabel=labels[sortedDistIndices[i]]
        #sortedDistIndices[i]  索引值[2 3 1 0]
        classCount[voteIlabel]=classCount.get(voteIlabel,0)+1
        sortedClassCount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]   #取第一个

if __name__ == '__main__':
    #创建数据集
    group,labels=createDataSet()
    #打印数据
    #print("数据样本group:",group)
    #print("特征标签labels:",labels)
    test=[101,20]
    #knn 分类 group=np.array([[1,101],[5,89],[108,5],[115,8]])
    test_class=classify0(test,group,labels,3)   #2/3/4会怎样
    #打印分类结果
    print("test_class:",test_class)

