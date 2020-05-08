# -*- coding:UTF-8 -*-
import numpy as np
import random
import re
from base import *

#考虑可能的重复单词，每遇到一个单词，增加向量中对应值，两排只为1，如上所示情况，叫做袋模型
def bagOfWords2VecMN(vocabList,inputSet):
    returnVec = [0] * len(vocabList)#创建一个和词汇表vocabList（用来对比的已有词汇表）相同长度的0向量
    for word in inputSet:  #遍历所有输入的inputSet待测单词文档
        if word in vocabList:  #判定单词出现在词汇表vocabList中
            returnVec[vocabList.index(word)] += 1 #输出文档向量对应值累加1  一般情况字典中会包含所有待测单词，所有会是1的累加，其他位置为初始化的0

    return returnVec

"""
函数说明：接收一个大字符串并将其解析为字符串列表

parameters:
    无
Return:
    无
line = re.sub
"""

def textParse(bigString):#将字符串转换为字符列表
    listOfTokens = re.split(r'\W+',bigString)#将特殊符号作为切分标志进行字符切分，即非字母，非数字
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]#除了单个字母，例如大写的I，其他单词变成小写
    #获得每个单词列表，条件是去除字符串长度小于2的字符（主要还是为了去除URL中连接条件=en等，所以》2）和标点，并转换为小写


"""
函数说明:测试朴素贝叶斯分类器
Parameters:
    无
Returns:
    无
"""

def spamTest():
    docList = [];
    classList = [];
    fullText = []
    for i in range(1,26):  #遍历25个txt文件 两个文件夹下有25+25个文件夹。导入并解析为词的列表
        wordList = textParse(open('./email/spam/%d.txt' % i, 'r').read())#读取每个垃圾邮件，并将字符串转换成字符串列表
        docList.append(wordList)
        fullText.append(wordList)
        classList.append(1)#标记垃圾邮件，1表示垃圾文件
        wordList = textParse(open('./email/ham/%d.txt' % i, 'r').read())#读取每个非垃圾邮件，并将字符串转换成字符串列表
        docList.append(wordList)
        fullText.append(wordList)
        classList.append(0) #标记非垃圾文件，1表示非垃圾文件
    vocabList = createVocabList(docList)#创建词汇表，不重复
    trainingSet = list(range(50));#构造训练集
    testSet = []#创建存储训练集的索引值的列表和测试集的索引值列表
    for i in range(10):#从50个邮件中，随机挑选出40个作为训练集，10个作为测试集
        randIndex = int(random.uniform(0,len(trainingSet)))#随机选取索引值 0-50间随机数
        testSet.append(trainingSet[randIndex])#添加测试集的索引值
        del(trainingSet[randIndex])#在训练集列表中删除添加到测试集的索引值  添加到测试集后删除
    trainMat = [];
    trainClasses = [] #创建训练集矩阵和训练集类别标签系向量

    #trainingSet中为[0,1,2,3,4,5,6,7,...,49]所以docIndex为序号数字
    for docIndex in trainingSet: #遍历训练集
        #将数据集字典和随即取出的10训练集合并
        trainMat.append(setOfWords2Vec(vocabList,docList[docIndex]))#将生成的词集模型添加到训练矩阵中
        trainClasses.append(classList[docIndex])#将类别添加到训练集类别标签系向量中  随即取出的10加入到训练集
    p0V,p1V,pSpam = trainNB0(np.array(trainMat),np.array(trainClasses))#训练朴素贝叶斯模型
    errorCount = 0 #错误分类计数
    for docIndex in testSet: #遍历测试集
        wordVector = bagOfWords2VecMN(vocabList,docList[docIndex]) #将测试文档做词袋模型进行遍历
        #wordVector = setOfWords2Vec(vocabList, docList[docIndex])
        if classifyNB(np.array(wordVector),p0V,p1V,pSpam) != classList[docIndex]: #如果分类错误
            errorCount += 1 #错误计数加1
            print("分类错误的测试集：",docList[docIndex])
    print('错误率：%.2f%%' % (float(errorCount) / len(testSet) * 100))

if __name__ == '__main__':
    docList = [];
    classList = [];
    for i in range(1, 26):  # 遍历25个txt文件 两个文件夹下有25+25个文件夹。导入并解析为词的列表
        wordList = textParse(open('./email/spam/%d.txt' % i, 'r').read())  # 读取每个垃圾邮件，并将字符串转换成字符串列表
        docList.append(wordList)
        classList.append(1)  # 标记垃圾邮件，1表示垃圾文件
        wordList = textParse(open('./email/ham/%d.txt' % i, 'r').read())  # 读取每个非垃圾邮件，并将字符串转换成字符串列表
        docList.append(wordList)
        classList.append(0)  # 标记非垃圾文件，1表示非垃圾文件
    vocabList = createVocabList(docList)  # 创建词汇表，不重复
    print(vocabList)
    spamTest()
