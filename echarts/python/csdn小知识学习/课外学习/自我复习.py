'''
from math import sqrt
import numpy as np

欧几里得空间法   计算相似度

def sim_distance(x,y):
    #方法1：根据公式求解
    d1=np.sqrt(np.sum(np.square(x-y)))
    print(d1)
    #方法2：根据scipy库求解
    from scipy.spatial.distance import pdist
    X=np.vstack([x,y])
    #将x，y两个一维数组合并成一个2D数组
    d2=pdist(X)
    print(d2)

def sim_distance(prefs,person1,person2):
    si={}
    for it in prefs[person1]:
        if it in prefs[person2]:
            si[it]=1
    if len(si)==0:
        return 0
    pSum=math.sqrt(sum(pow(prefs[person1][it]-prefs[person2][it],2)for it in si))
    return 1.0/(1+pSum)
'''
#数据处理模块整体
import numpy as np
def file2matrix(filename):
    fr=open(filename)
    arrayOLines=fr.readlines()
    print("arrayOLines",arrayOLines)
    numberOfLines=len(arrayOLines)
    returnMat=np.zeros((numberOfLines,3))
    print("returnMat",returnMat)
    classLabelVector=[]
    index=0
    for line in arrayOLines:
        line=line.strip()
        print(line)
        listFromLine=line.split('\t')
        returnMat[index,:]=listFromLine[0:3]
        print(returnMat)
        if listFromLine[-1]=='猪队友':
            classLabelVector.append(1)
        elif listFromLine[-1]=='一般般':
            classLabelVector.append(2)
        elif listFromLine[-1]=='神队友':
            classLabelVector.append(3)
        index+=1
    return returnMat,classLabelVector
if __name__ == '__main__':
    filename="datingTestSet01.txt"
    datingDatMat,datingLabels=file2matrix(filename)
    print(datingDatMat)
    print(datingLabels)