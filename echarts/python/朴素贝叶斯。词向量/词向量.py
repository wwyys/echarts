from numpy import *

'''
创建实验文本，返回的第一个变量为切分单词后的文档集合，第二个变量为标签集合，即侮辱性文章
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
功能：创建一个文档的字典
input:文档数据 postingList
output：文档字典
        postingList---组合 去重
'''

def creatVocabList(dataset):
    vocabSet=set([])
    for document in dataset:
        vocabSet=vocabSet | set(document)
        print("字典如何实现的",vocabSet)
    return list(vocabSet)

'''
功能：作词向量，对应一个文档（文档切分，jieba分词后)对应我已知文档数据集合（刚刚做好的），存在为1，不存在为0
input:未知文档，文档数据集合
output：词向量
'''
def setOfWords2Vec(vocabList,inputSet):
    returnVec=[0]*len(vocabList)  #构建一个文档字典集合大小的空间用于存储词向量，初始值为0
    print("词向量空的空间构建",returnVec)
    for word in inputSet:
        if word  in  vocabList:
            returnVec[vocabList.index(word)]=1
            print("如何构建词向量",returnVec)
        else:
            print("这个词：%s 不再我的字典里" %word)
    return returnVec

'''
function:对应已有的单词字典，检测某个段落中的单词是否出现
input:输入的是文档矩阵，即为文档字典；每个段落文档对应的向量
output:好评和差评的概率和总的概率
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
    p0Num = ones(numWords)
    print("p0Num", p0Num)
    p1num = ones(numWords)
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
            print("应该词的进行计数",p1Demo)
        else:
            p0Num += trainMatrix[i]
            p0Demo +=sum(trainMatrix[i])
        #p1Vect = p1num/p1Demo
        #print("在差评条件下文档中单词概率",p1Vect)
        #p0Vect=p0Num/p0Demo
        #print("在好评条件下文档中单词概率",p0Vect)
        #为解决下溢出问题,及多个而非常小的数相乘，造成结果下溢或得不到想要结果，如下溢为0；解决方法：乘积取自然对数
        p1Vect = log(p1num / p1Demo)
        p0Vect = log(p0Num / p0Demo)
    return pAbusive,p1Vect,p0Vect #pA整个先验证概率

'''
功能：朴素贝叶斯分类器
input:未知的段落  0概率  1概率  先验概率
output：0,1  好评和差评
'''
def classifyNB(vec2classify,p0Vec,p1Vec,pclass1):
    #为差评的分类 1
    #差评的概率，假设词之间独立的  p1=p(x*p1v)p(c)p(x)=(p1vect*vect2classify)*pAb
    p1=sum(vec2classify*p1Vec)+log(pclass1)

    #好评概率
    p0=sum(vec2classify*p0Vec)+log(1.0-pclass1)

    if p1>p0:
        return 1
    else:
        return 0

def testingNB(testEntry):
    list0Posts,listClasses=loadDataSet()
    myVocablist=creatVocabList(list0Posts)
    #创建文档矩阵
    trainMat=[]
    for postinDoc in list0Posts:
        trainMat.append(setOfWords2Vec(myVocablist,postinDoc))  #检测关键词出现的情况
    pAb,p1V,p0V=trainNB0(array(trainMat),array(listClasses))  #训练数据集形成各类概率
    #测试方法1
    #testEntry=['my','stupid','dalmation']
    thisDoc=array(setOfWords2Vec(myVocablist,testEntry))
    print('输入的测试集testEntry类别是',classifyNB(thisDoc,p0V,p1V,pAb))
    #测试方法2
    testEntry2=['my','stupid','love','dalmation']
    thisDoc = array(setOfWords2Vec(myVocablist, testEntry2))
    print('输入的测试集testEntry2类别是', classifyNB(thisDoc, p0V, p1V, pAb))

if __name__=="__main__":
    '''
    postingList,classVec=loadDataSet()
    for each in postingList:
        print("我想知道",each)
    print("打印类别",classVec)
    myVocablist=creatVocabList(postingList)
    print("输出的文档没重复的字典集合",myVocablist)
    postingList_test=['你', '羊', '太阳','有', '问题', '帮助', '请']
    word2vec=setOfWords2Vec(myVocablist,postingList[0])
    word2vec = setOfWords2Vec(myVocablist, postingList_test)
    trainMat=[]
    for postinDoc in postingList:
        trainMat.append(setOfWords2Vec(myVocablist,postinDoc))
        print("输出所有的词的列表")
    pAb,p0V,p1V=trainNB0(trainMat,classVec)
    print("在差评条件下文档中单词概率", p1V)
    print("在好评条件下文档中单词概率", p0V)
    print("差评的先验概率", pAb)
    '''
    testEntry = ['my', 'stupid','love', 'dalmation']
    testingNB(testEntry)
