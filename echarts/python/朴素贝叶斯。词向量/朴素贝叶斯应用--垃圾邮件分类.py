
import re
import numpy as np
import random

'''
作者：zxl
目标：创建数据集
内容：创建实验文本，返回的第一个变量为切分单词后的文档集合，第二个变量为标签集合，即侮辱性文章
postingList:样本集     classVec：每个样本对应的标签
时间： 2020.04.05
'''

def loadDataSet():#文档切分后，去除停用词和符号，所剩单词
    postingList=[['我', '狗', '有', 'flea', '问题', '帮助', '请'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0, 1]    #1 is abusive, 0 not  1为骂人的，侮辱的 0为普通的向量  分别标出每一维度的类别倾向，用于训练数据，所以 len（classVec）向量数为 len（postingList）维度的值相等
    return postingList,classVec

'''
功能：构造词典
dataset：训练集
output：词典 vocabSet

'''

def creatVocabList(dataset):
    vocabSet=set([])
    for document in dataset:
        vocabSet=vocabSet | set(document)
        print("字典如何实现的",vocabSet)
    return list(vocabSet)

'''
构造词集模型
功能：作词向量，对应一个文档（文档切分，jieba分词后)对应我已知文档数据集合（刚刚做好的），存在为1，不存在为0
vocabList：词典
input:  未知文档，文档数据集合  （输入样本）
output： 该样本的词集模型（returnVec）
'''
def setOfWords2Vec(vocabList,inputSet):
    returnVec=[0]*len(vocabList)  #构建一个文档字典集合大小的空间用于存储词向量，初始值为0
    print("词向量的空间构建",returnVec)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)]=1
            print("如何构建词向量",returnVec)
        else:
            print("这个词：%s 不在我的字典里" %word)
    return returnVec

'''
求解先验概率和条件概率
function:对应已有的单词字典，检测某个段落中的单词是否出现，获取每个单词在每个类别下的条件概率
input:输入的是文档矩阵，即为文档字典；每个段落文档对应的向量
output:好评和差评的概率和总的概率
trainMatrix：训练集转换成的词向量矩阵
trainCattagroy：每个样本的类别标签
'''
def trainNB0(trainMatrix,trainCattagroy):
    numTrainDocs=len(trainMatrix) #文档个数 6个
    print("文章总的段落数",numTrainDocs)
    numWords=len(trainMatrix[0])  #文档中词的个数 取得是第一个段落的词向量的个数
    print("文档词的个数",numWords)
    #p(A/B)=P(B/A)P(A)/P(B)  P(A)差评先验概率
    #差评的先验概率 010101  0对应好评==1-pAbusive  1对应的是差评  骂人的
    pAbusive = sum(trainCattagroy)/float(numTrainDocs)
    #print("差评的先验概率",pAbusive)

    #以下为初始化概率分子变量和分母变量，来计算p(wi/c1)和p(wi/c0)准备
    #p0Num=zeros(numWords)
    #print("p0Num",p0Num)
    #p1num=zeros(numWords)
    p0Num = np.ones(numWords)
    print("p0Num", p0Num)
    p1num = np.ones(numWords)
    #分类0/1的所有文档内的所有单词数统计
    #p0Demo=0.0
    #p1Demo=0.0
    p0Demo = 2.0  #初始化为2
    p1Demo = 2.0

    #遍历6个文档，分母变量是一个元素个数等于词汇表大小的numpy数组，遍历训练集trainMatrix中的所有文档
    for i in range(numTrainDocs):
        if trainCattagroy[i]==1:  #一旦某段词类型（差评或普通型）出现==1或==0  若文档属于分类1
            p1num +=trainMatrix[i]  #对应该次的进行计数 p1num p0num 加1 词向量累加
            print("对应词的计数",p1num)
            p1Demo+=sum(trainMatrix[i])
            print("对该词的进行计数",p1Demo)
        else:
            p0Num += trainMatrix[i]
            p0Demo +=sum(trainMatrix[i])
        #p1Vect = p1num/p1Demo
        #print("在差评条件下文档中单词概率",p1Vect)
        #p0Vect=p0Num/p0Demo
        #print("在好评条件下文档中单词概率",p0Vect)
        #为解决下溢出问题,及多个而非常小的数相乘，造成结果下溢或得不到想要结果，如下溢为0；解决方法：乘积取自然对数
        p1Vect = np.log(p1num / p1Demo)
        p0Vect = np.log(p0Num / p0Demo)
    return pAbusive,p1Vect,p0Vect #pA整个先验证概率


'''
对样本集进行分类
功能：朴素贝叶斯分类器
input:未知的段落  0概率  1概率  先验概率
output：0,1  好评和差评
'''
def classifyNB(vec2classify,p0Vec,p1Vec,pclass1):
    #为差评的分类 1
    #差评的概率，假设词之间独立的  p1=p(x*p1v)p(c)p(x)=(p1vect*vect2classify)*pAb
    p1=sum(vec2classify*p1Vec)+np.log(pclass1)

    #好评概率
    p0=sum(vec2classify*p0Vec)+np.log(1.0-pclass1)

    if p1>p0:
        return 1
    else:
        return 0

#训练样本数据集
def testingNB(testEntry):
    #拉取数据集
    list0Posts,listClasses=loadDataSet()
    #获取词典
    myVocablist=creatVocabList(list0Posts)
    #创建文档矩阵
    trainMat=[]
    for postinDoc in list0Posts:
        trainMat.append(setOfWords2Vec(myVocablist,postinDoc))  #检测关键词出现的情况
    pAb,p1V,p0V=trainNB0(np.array(trainMat),np.array(listClasses))  #训练数据集形成各类概率
    return pAb,p1V,p0V,myVocablist

def loadData():
    '''
    加载email文件夹下面的所有ham和spam文件
    ham下面的样本为0类别
    spam下面的样本为1类别
    : 样本集，标签集
    '''
    trainWordsList = []  # 训练集词集
    classVector = []  # 训练集中每一份文档的类别
    for i in range(1, 26):
        curWordList = (open("./email/ham/%d.txt" %i).read())
        trainWordsList.append(curWordList)
        classVector.append(0)
        curWordList = (open("./email/spam/%d.txt" % i).read())
        trainWordsList.append(curWordList)
        classVector.append(1)
    return trainWordsList, classVector


def spamTest():
    
    #在测试集中选择一部分数据进行交叉验证
    
    # 1.加载数据
    trainWordsList, classVector = loadData()
        # 2.生成词典
    wordDirectory = creatVocabList(trainWordsList)
        # 转换成词集模型
    wordSetVector = []
    for i in range(len(trainWordsList)):
        wordSetVector.append(setOfWords2Vec(wordDirectory, trainWordsList[i]))
        # 3.从50个测试样本集中随机抽取出10个样本
    trainSet = range(50)
    testSetVector = []  # 测试集样本， 其中存储的是测试集的词集模型
    testSetLabels = []  # 测试集的类别标签，用于后面检验分类的结果
    for i in range(10):
        randomNum = int(np.random.uniform(0, len(wordSetVector), 1))  # numpy中的均匀分布中进行采样
        print("当前随机数为：%f" % randomNum)
        testSetVector.append(wordSetVector[randomNum])
        testSetLabels.append(classVector[randomNum])
        # 添加完之后在测试集中删除掉这一条
        wordSetVector.__delitem__(randomNum)
        classVector.__delitem__(randomNum)
        # 4.将剩下的40个测试集转换成词典模型
    print("训练集的个数为：%d, 测试集的个数为：%d" % (len(wordSetVector), len(testSetVector)))
    print(wordSetVector)
        # 5.训练训练集样本，得到每一个单词为类别1时的概率
    pAb,p1V,p0V = setOfWords2Vec(wordSetVector, classVector)
    print("每个单词为类别1的概率:")
    print(pAb)
        # 6.测试10个测试集，用NB进行分类
    testSetCalLabels = []
    for i in range(len(testSetVector)):
        #curVec = testSetVector[i] | pAb
        class1Pro = sum(testSetVector[i] * pAb) + np.log(p0V)
        class0Pro = sum(testSetVector[i] * p1V) + np.log(1 - p0V)
        if class1Pro > class0Pro:
            testSetCalLabels.append(1)
        else:
            testSetCalLabels.append(0)
        print("当前测试样本的分类标签为：%d" % testSetCalLabels[i])
        # 7.检验分类结果，输出出错率
    errCount = 0
    for i in range(len(testSetCalLabels)):
        if testSetCalLabels[i] != testSetLabels[i]:
            print("第%d个测试样本分类错误" % i)
        errCount += 1
    print("分类错误率为: %f" % (errCount / 10))
    
if __name__ == '__main__':
    spamTest()





