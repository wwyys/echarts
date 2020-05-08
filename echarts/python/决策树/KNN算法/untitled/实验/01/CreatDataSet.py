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
