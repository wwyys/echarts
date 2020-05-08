def loadDataSet():
    postingList=[['我','狗','有','flea','问题','帮助','请'],
                 ['maybe','not','take','him','to','dog','park','stupid'],
                 ['my','dalmation','is','so','cute','I','love','him'],
                 ['stop','posting','stupid','worthless','garbage'],
                 ['mr','licks','ate','my','steak','how','to','stop','him'],
                 ['quit','buying','worthless','dog','food','stupid']]
    classVec=[0,1,0,1,0,1]
    return postingList,classVec

'''
功能：创建一个文档的字典
input:文档数据 postinglist
output：文档的字典
        postinglist---组合  去重复
'''
def creatVocabList(dataset):
    vocabSet=set([])
    for document in dataset:
        vocabSet=vocabSet | set(document)
        print("字典如何实现",vocabSet)
    return list(vocabSet)
'''
功能：作词向量，对应一个文档（文档切分，jieba分词后）对应我已知文档数据集合（刚刚做好的），存在1，不存在0
input：未知文档，文档数据集合
output：词向量
'''
def setOfWards2Vec(vocabList,inputSet):
    returnVec=[0]*len(vocabList)   #构建一个文档字典集合大小的向量空间，用于存储词向量，初始值为0
    print("词向量空的空间构建",returnVec)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)]=1
            print("如何构建词向量",returnVec)
        else:
            print("这个词:%s  不在我的字典里"% word)
    return returnVec

if __name__ == '__main__':
    postingLIst,classVec=loadDataSet()
    for each in postingLIst:
        print("我想知道",each)
    print("打印类别",classVec)
    myVocablist=creatVocabList(postingLIst)
    print("输出的文档没重复的字典集合",myVocablist)
    postingLIst_test=['你','样','有','太阳','问题','帮助','请']
    word2voc=setOfWards2Vec(myVocablist,postingLIst[0])
    word2voc = setOfWards2Vec(myVocablist, postingLIst_test)
