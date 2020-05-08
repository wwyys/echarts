# -*- coding: GBK -*-
# �������������  .encode('gbk')
# !/usr/bin/env python
"""
#���ر�Ҷ˹
author: lkn
create on 2014-1-21 5:30

"""

from numpy import *  # Ӧ��Numpy���������Ա��뽫����������ǰ��

"""
function������һ��ʵ�����������صĵ�һ������Ϊ�зֵ��ʺ���ĵ����ϣ��ڶ�������Ϊ��ǩ���ϣ�����������ͨ�����༯�ϣ����˹���ע��
author: lkn
create on 2014-1-21 5:30
"""


def loadDataSet():  
    postingList = [['��', '��', '��', 'flea', '����', '����', '��'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]  #
    classVec = [0, 1, 0, 1, 0,
                1] 

    return postingList, classVec
"""
function������һ�����������ĵ��г��ֵģ����ظ��Ĵ��б�Ӧ��set�������ز��ظ�
"""


def createVocabList(dataSet):
    vocabSet = set([])  # create empty set
    for document in dataSet:
        vocabSet = vocabSet | set(document)  # union of the two sets  | Ϊor ��  �ϲ�
    return list(vocabSet)


"""
function:��Ӧ���е����ֵ䣬���ĳ���ĵ��е����Ƿ���֣��������ظ����ʳ������
input:����Ϊ�ʻ���ĳ���ĵ�
output�����ĵ����� ÿ��Ԫ�طֱ�Ϊ 1,0����ʾ�ʻ���еĵ�����������ĵ����Ƿ����
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
function:��Ӧ���е����ֵ䣬���ĳ���ĵ��е����Ƿ���֣�
input:����Ϊ�ĵ�����maxtrix,�Լ���ÿƪ�ĵ�����ǩ�����ɵ�����
output������������������ĸ��ʺ�һ���ܵĸ��ʣ��궨��ά���в�����ռ�ܵ�ά�ȵĸ���   0��1��0��1��0��1 
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

    print("P(B)�����������",p1Vect)

    
    #p1Vect = log(p1Num / p1Demom)  
    # p0Vect = log(p0Num / p0Demom)

    return p0Vect, p1Vect, pAbusive


"""
��Ҷ˹��������


#    ����:vec2Classify  �������������p0Vec, p1Vec, pClass1 ΪtrainNB0��output�����������ĸ��ʺ�һ���ܸ���
 3 #        vec2Classify:     Ŀ�����Ĵ�������������ʽ
 4 #        p0Vect:    �������ڷ���0�������³��ֵĸ���
 5 #        p1Vect:    �������ڷ���1�������³��ֵĸ���
 6 #        pClass1:  �ĵ����ڷ���1�ĸ���
 7 #    ���:
 8 #        ������ 0/1
"""


def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):#'��ɱ�Ҷ˹��ʽʣ�ಿ�ֵõ����շ������'

   
    p1 = sum(vec2Classify * p1Vec) + log(
        pClass1)  

    
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)  
    if p1 > p0:
        return 1
    else:
        return 0


# ���Ժ���  ����������
def testingNB():
    listOPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)

   
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))  

   
    p0V, p1V, pAb = trainNB0(array(trainMat), array(listClasses))  

    # ����һ
    testEntry = [ 'my', 'stupid','dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))  
    print(testEntry, "����Ϊ��", classifyNB(thisDoc, p0V, p1V, pAb) )
    
    # ���Զ�
    testEntry = ['love',  'dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, "����Ϊ��", classifyNB(thisDoc, p0V, p1V, pAb))



"""
����
"""
if __name__ == '__main__':

    listOPosts, listClasses = loadDataSet()  # get word set  and  labels
    myVocabList = createVocabList(listOPosts)  
    print("������ظ����ݼ�\n")

    print( myVocabList)

    #print("�������ĵ��ʼ���0ά\n")

    #print(listOPosts[0])
    
    #print( "�������ĵ��ʼ���5ά\n")

    #print(listOPosts[5])

    
    #print(setOfWords2Vec(myVocabList, listOPosts[0]))
     
    #print(setOfWords2Vec(myVocabList, listOPosts[5]))


    """
    Ӧ��ѵ�����ݼ����з���ѵ�����ʲ���
    """
    trainMat = [] 
    for postinDoc in listOPosts:  
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    print("������д��б�myVocabList�ʵ�����\n")

    print("������д��б�",trainMat)

    #print(sum(listClasses))
  
    #print(len(trainMat))

    #print(range(len(trainMat)))

    p0V, p1V, pAb = trainNB0(trainMat, listClasses)  
    print( "����������ĵ�����pAb=%s\n" % pAb)


    #print(pAb)

    print( "����p0V",p0V )
   
    print("����p1V",p1V)
    

    testingNB()
