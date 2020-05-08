from numpy import *
def loadDataSet():  # 文档切分后,去除停用词和符号，所剩单词
    postingList = [['我', '狗', '有', 'flea', '问题', '帮助', '请'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]   #1 is abusive, 0 not  1为骂人的，侮辱的 0为普通的向量  分别标出每一维度的类别倾向，用于训练数据，所以 len（classVec）向量数为 len（postingList）维度的值相等
    return postingList, classVec


'''
功能: 创建一个文档的字典
input : 文档数据 postinglist
output: 文档的字典
                postinglist---组合 去重复set()
'''


def createVocabList(dataset):
    vocabSet = set([])
    for document in dataset:
        vocabSet = vocabSet | set(document)
        print('字典如何实现', vocabSet)
    return list(vocabSet)


'''
功能:做词向量，对应一个文档(文档切分,jieba分词后)对应我已知文档数据集合(刚刚做好的),存在 1 不存在 0
input:未知文档，文档数据集合
output:词向量
'''


def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)  # 构建一个文档字典集合大小的向量空间，用于存储词向量，初始值为0
    print('词向量空的空间构建', returnVec)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
            print('如何构建词向量', returnVec)
        else:
            print('这个词:%s 不在我的字典里' % word)
    return returnVec


'''

function:对应已有的单词字典，检测某个段落中的单词是否出现
input:输入的为文档矩阵，文档字典，每个段落文档对应的向量
output:好评和差评的概率和总的概率

'''


def trainNB0(trainMatrix, trainCattagory):
    numTrainDocs = len(trainMatrix)  # 文档的个数 段落数 共6个
    print('文档个数', numTrainDocs)
    numWords = len(trainMatrix[0])  # 文档中词的个数  取得是第一个段落词向量的个数
    print('文档词的个数', numWords)
    # P(A|B)=P(B|A)P(A)/P(B)  P(A)差评先验概率
    # 差评的先验概率  010101 0对应好评 1对应的是差评 骂人的
    PAbusive = sum(trainCattagory) / float(numTrainDocs)
    # print('差评的先验概率', PAbusive)

    #  以下为初始概率分子变量和分母变量，来计算P(wi|c1)和P(wi|c0)准备，由于w中元素众多，应用Numpy数组快速计算
    p0Num = zeros(numWords)  #ones()
    #p0Num = ones(numWords)  # 计算P(wi|ci)和P(wi|c0)相乘，若有一个概率为0，则乘积也为0的问题修改
    #print('p0Num', p0Num)
    p1Num = zeros(numWords)
    #p1Num = ones(numWords)  # 初始化分子为1
    # 分类0/1的所有文档内的所有单词数统计
    p0Demom=0.0
    p1Demom=0.0
    #p0Demom = 2.0  # 初始化分母为2
    #p1Demom = 2.0
    # 遍历6个文档，分母变量是一个元素个数等于词汇表大小的Numpy数组，遍历训练集trainMatrix中所有文档
    for i in range(numTrainDocs):
        if trainCattagory[i] == 1:  # 一旦某段词类型(差评或普通型)出现==1或==0  # 若文档属于分类1
            p1Num += trainMatrix[i]  # 对应该词的进行计数 p1Num p0Num 加1 # 词向量累加
            print("对应词的计数", p1Num)
            p1Demom += sum(trainMatrix[i])  # p(B)差评先验概率,p(A|B)P(A)/P(B)且所有文档总词数加1 #分类1文档单词数累加
            print('应该词的进行计数', p1Demom)
        else:
            p0Num += trainMatrix[i]
            p0Demom += sum(trainMatrix[i])
    p1Vect=p1Num/p1Demom
    print('在差评发生的条件下文档中次的概率', p1Vect)
    p0Vect=p0Num/p0Demom
        # print('在好评发生的条件下文档中次的概率', p0Vect)
        # 为了解决下溢出问题，即多个非常小的数相乘，造成结果下溢或得不到想要的结果，如下溢为0，  办法：累积取自然对数
        #p1Vect = log(p1Num / p1Demom)
        #p0Vect = log(p0Num / p0Demom)
    return p0Vect, p1Vect, PAbusive


'''
功能：朴素贝叶斯分类器
input:未知的段落 0概率 1概率 先验概率
output:0,1 好评和差评
'''


def classifyNB(vec2classify, p0Vec, p1Vec, pClass1):
    # 为差评的分类 1
    # 差评的概率，假设词之间独立的 p1=p(x*p1v)p(c)/p(x)=(x*p1V)p(c)=(p1Vect*vect2Classify)*pAb
    p1 = sum(vec2classify * p1Vec) + log(pClass1)
    # 好评概率多少?
    p0 = sum(vec2classify * p0Vec) + log(1.0 - pClass1)

    if p1 > p0:
        return 1
    else:
        return 0


def testingNB(testEntry):
    list0Posts,listClasses=loadDataSet()
    myVocablist=createVocabList(list0Posts)

    #创建文档矩阵
    trainMat=[]
    for postinDoc in list0Posts:
        trainMat.append(setOfWords2Vec(myVocablist,postinDoc))#检测关键词出现的情况

    p0V,p1V,pAb =trainNB0(array(trainMat),array(listClasses))#训练数据集，形成各类概率
    #测试方法1
    testEntry=['my','stupid','not','dalmation','dog','steak', 'how', 'to']
    thisDoc=array(setOfWords2Vec(myVocablist,testEntry))
    print("输入的测试集testEntry类别是",classifyNB(thisDoc,p0V,p1V,pAb))
    #测试方法2
    testEntry2 = ['my','love','good', 'dalmation']
    thisDoc = array(setOfWords2Vec(myVocablist, testEntry2))
    print("输入的测试集testEntry2类别是", classifyNB(thisDoc, p0V, p1V, pAb))
if __name__ == '__main__':
    postingList, classVec=loadDataSet()
    #for each in postingList:
        #print("我想知道",each)
    #print("打印类别",classVec)
    myVocablist=createVocabList(postingList)
    print("输出无重复数据集\n")
    print("输出的文档没重复的字典集合",myVocablist)
    #postingList_test=['你', '羊', '有', '太阳', '问题', '帮助', '请']
    #word2voc=setOfWords2Vec(myVocablist,postingList[0])
    #word2voc = setOfWords2Vec(myVocablist, postingList_test)
    trainMat=[]
    for postinDoc in postingList:
        trainMat.append(setOfWords2Vec(myVocablist,postinDoc))
        print("输出所有词的列表",trainMat)
    p0V,p1V,pAb=trainNB0(trainMat,classVec)
    print("在差评发生的条件下文档中次的概率",p1V)
    print("在好评发生的条件下文档中次的概率", p0V)
    print("差评的先验概率", pAb)

    testEntry = ['my','stupid','not','dalmation']
    testingNB(testEntry)