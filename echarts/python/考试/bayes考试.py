# -*- coding: GBK -*-
# 中输出乱码问题  .encode('gbk')
# !/usr/bin/env python
"""
#朴素贝叶斯
author: lkn
create on 2014-1-21 5:30

"""

from numpy import *  # 应用Numpy函数，所以必须将该语句放在最前面

"""
function：创建一个实验样本，返回的第一个变量为切分单词后的文档集合，第二个变量为标签集合，即差评和普通两个类集合，是人工标注的
author: lkn
create on 2014-1-21 5:30
"""


def loadDataSet():  
    postingList = [['我', '狗', '有', 'flea', '问题', '帮助', '请'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]  #
    classVec = [0, 1, 0, 1, 0,
                1] 

    return postingList, classVec
"""
function：创建一个包含所有文档中出现的，不重复的词列表，应用set（）返回不重复
"""


def createVocabList(dataSet):
    vocabSet = set([])  # create empty set
    for document in dataSet:
        vocabSet = vocabSet | set(document)  # union of the two sets  | 为or ∪  合并
    return list(vocabSet)


"""
function:对应已有单词字典，检测某个文档中单词是否出现，不考虑重复单词出现情况
input:输入为词汇表和某个文档
output：是文档向量 每个元素分别为 1,0，表示词汇表中的单词在输入的文档中是否出现
"""


def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList) 
    for word in inputSet: 
        if word in vocabList:  
            returnVec[vocabList.index(word)] = 1 
        else:
            print("the word: %s is not in my Vocabulary!" % word)

    return returnVec



def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0] * len(vocabList)  
    for word in inputSet:  
        if word in vocabList:  
            returnVec[vocabList.index(word)] += 1  
    return returnVec


"""
function:对应已有单词字典，检测某个文档中单词是否出现，
input:输入为文档矩阵maxtrix,以及由每篇文档类别标签所构成的向量
output：两个类别两个向量的概率和一个总的概率，标定的维度中差评性占总的维度的概率   0，1，0，1，0，1 
"""


def trainNB0(trainMatrix, trainCatagory):
    numTrainDocs = len(trainMatrix) 
  
    numWords = len(trainMatrix[0]) 
  
    pAbusive = sum(trainCatagory) / float(
        numTrainDocs) 
          
    p0Num = zeros(numWords)    
    p1Num = zeros(numWords)
    #p0Num = ones(numWords) 
    #p1Num = ones(numWords)  

    
    p0Demom = 0.0
    p1Demom = 0.0
    #p0Demom = 2  
    #p1Demom = 2
    for i in range(numTrainDocs):  
        if trainCatagory[i] == 1: 
            p1Num += trainMatrix[i]  
           
            p1Demom += sum(trainMatrix[i]) 
           
        else: 
            p0Num += trainMatrix[i] 
            p0Demom += sum(trainMatrix[i])  
    p1Vect = p1Num / p1Demom   
    p0Vect = p0Num / p0Demom

    print("P(B)差评先验概率",p1Vect)

    
    #p1Vect = log(p1Num / p1Demom)  
    # p0Vect = log(p0Num / p0Demom)

    return p0Vect, p1Vect, pAbusive


"""
贝叶斯分类器：


#    输入:vec2Classify  待分类的向量，p0Vec, p1Vec, pClass1 为trainNB0的output：两个向量的概率和一个总概率
 3 #        vec2Classify:     目标对象的词向量的数组形式
 4 #        p0Vect:    各单词在分类0的条件下出现的概率
 5 #        p1Vect:    各单词在分类1的条件下出现的概率
 6 #        pClass1:  文档属于分类1的概率
 7 #    输出:
 8 #        分类结果 0/1
"""


def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):#'完成贝叶斯公式剩余部分得到最终分类概率'

   
    p1 = sum(vec2Classify * p1Vec) + log(
        pClass1)  

    
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)  
    if p1 > p0:
        return 1
    else:
        return 0


# 测试函数  便利函数，
def testingNB():
    listOPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)

   
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))  

   
    p0V, p1V, pAb = trainNB0(array(trainMat), array(listClasses))  

    # 测试一
    testEntry = [ 'my', 'stupid','dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))  
    print(testEntry, "分类为：", classifyNB(thisDoc, p0V, p1V, pAb) )
    
    # 测试二
    testEntry = ['love',  'dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, "分类为：", classifyNB(thisDoc, p0V, p1V, pAb))



"""
测试
"""
if __name__ == '__main__':

    listOPosts, listClasses = loadDataSet()  # get word set  and  labels
    myVocabList = createVocabList(listOPosts)  
    print("输出无重复数据集\n")

    print( myVocabList)

    #print("输出待测的单词集合0维\n")

    #print(listOPosts[0])
    
    #print( "输出待测的单词集合5维\n")

    #print(listOPosts[5])

    
    #print(setOfWords2Vec(myVocabList, listOPosts[0]))
     
    #print(setOfWords2Vec(myVocabList, listOPosts[5]))


    """
    应用训练数据集进行分类训练概率测试
    """
    trainMat = [] 
    for postinDoc in listOPosts:  
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    print("输出所有词列表myVocabList词典数据\n")

    print("输出所有词列表",trainMat)

    #print(sum(listClasses))
  
    #print(len(trainMat))

    #print(range(len(trainMat)))

    p0V, p1V, pAb = trainNB0(trainMat, listClasses)  
    print( "输出差评性文档概率pAb=%s\n" % pAb)


    #print(pAb)

    print( "好评p0V",p0V )
   
    print("差评p1V",p1V)
    

    testingNB()
