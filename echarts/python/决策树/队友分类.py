import numpy as np
'''
def file2matrix(filename):
    fr = open(filename)
    arrayOlines = fr.readlines()
    numberOflines = len(arrayOlines)
    returnMat = np.zeros((numberOflines, 3))
    classLabelVector = []
    index = 0
    for line in arrayOlines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:3]
        if listFromLine[-1] == '猪队友':
            classLabelVector.append(1)
        elif listFromLine[-1] == '一般般':
            classLabelVector.append(2)
        elif listFromLine[-1] == '神队友':
            classLabelVector.append(3)
        index += 1
    return returnMat, classLabelVector
'''
def autoNorm(dataSet):
    minVals=dataSet.min(0)
    maxVals=dataSet.max(0)
    ranges=maxVals-minVals
    normDataSet=np.zeros(np.shape(dataSet))
    m=dataSet.shape[0]
    normDataSet=dataSet-np.tile(minVals,(m,1))
    normDataSet = normDataSet/np.tile(ranges,(m,1))
    return normDataSet,ranges,minVals
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
        listFromLine=line.split('\t')
        returnMat[index,:]=listFromLine[0:3]
        if listFromLine[3:4] == 'didntLike':
            classLabelVector.append(1)
        elif listFromLine[-1] == 'smallDoses':
            classLabelVector.append(2)
        elif listFromLine[-1] == 'largeDoses':
            classLabelVector.append(3)
        index+=1
    return returnMat,classLabelVector
'''
if __name__ == '__main__':
    filename="datingTestSet01.txt"
    datingDataMat,datingLabels=file2matrix(filename)
    normDataSet,ranges,minVals=autoNorm(datingDataMat)
    print("normDataSet",normDataSet)
    print("ranges",ranges)
    print("minVals",minVals)
'''
def createDataSet():
    group=np.array([[1,101],[5,89],[108,5],[115,8]])
    labels=['爱情片','爱情片','动作片','动作片']
    return group,labels

def classifyPerson():
    resultList=['猪队友','一般般','神队友']
    #三维特征用户输入
    onHook=float(input("挂机比率:"))
    takePosition=float(input("抢位置数:"))
    anchor=float(input("主播比率数:"))
    #打开的文件名
    filename="datingTestSet.txt"
    #打开并处理数据
    datingDataMat,datingLabels=file2matrix(filename)
    normMat,ranges,minVals=autoNorm(datingDataMat)
    #生成numpy数组，测试集
    inArr=np.array([takePosition,onHook,anchor])
    #测试集归一化
    norminArr=(inArr-minVals)/ranges
    classifierResult=classify0(norminArr,normMat,datingLabels,3)
    #打印结果
    print("这个人可能是你的%s" % (resultList[classifierResult-1]))
if __name__ == '__main__':
    classifyPerson()