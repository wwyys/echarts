import os
import jieba
import numpy
from sklearn.feature_extraction.text import TfidfTransformer,CountVectorizer
from sklearn.naive_bayes import MultinomialNB

'''
1.数据集我们已经有了  爬虫爬取
2.构建MNB（）形参数据类型

'''

'''
功能：数据集已经有，读取数据，构建特征和类
input：有文件夹——路径
output:特征和类
'''
def loadDataSet(path_name,classtag):
    allfiles=os.listdir(path_name)
    processed_textset=[]
    allclasstags=[]
    for thisfile in allfiles:
        path_name="./data/"+classtag+"/"+thisfile
        print("path_name",path_name)
#打开文件：
        textfile=open(path_name,"r",encoding="utf-8").read()
        print("textfile",textfile)
        #词向量  一个个的词构成
        textcut=jieba.cut(textfile)
        print("textcut",textcut)
        text_with_space=""
        for word in textcut:
            text_with_space +=word + " "
            print("text_with_space",text_with_space)
        processed_textset.append(text_with_space)
        print("processed_textset",processed_textset)
        allclasstags.append(classtag)
    return processed_textset,allclasstags

'''
3.fit（）训练
4.predict(x)位置的x预测
功能：算法训练与预测
input:数据特征词向量  processed_textset  属于哪类 allclasstags 预测的文章
output:输出类别
'''
def sklearnNB(integrated_train_data,classtages_list,testset):
    count_vector=CountVectorizer() #向量程序构建一个包含特征索引的字典
    vextor_matrix=count_vector.fit_transform(integrated_train_data) #进行对数据进行标准化和归一化处理
    train_tfidf=TfidfTransformer(use_idf=False).fit_transform(vextor_matrix)  #从词词的计数到词频  将词频矩阵转化为权重矩阵
    clf=MultinomialNB().fit(train_tfidf,classtages_list)
    new_count_vector=count_vector.transform(testset)
    new_tfidf=TfidfTransformer(use_idf=False).fit_transform(new_count_vector)
    prideict_result=clf.predict(new_tfidf)
    print("prideict_result",prideict_result)



if __name__=="__main__":
    path="./data/testdata/testzx.txt"
    testset=[]
    textfile=open(path,"r",encoding="utf-8").read()
    textcut=jieba.cut(textfile)
    text_with_space=""
    for word in textcut:
        text_with_space+=word+" "
    testset.append(text_with_space)
    testset.append("他么每次下暴雨就听我们一期，二期每次都没听，真是服了！")


    path_name="./data/中性"
    path_name1 = "./data/正面"
    path_name2 = "./data/负面"

    processed_textset,allclasstags=loadDataSet(path_name,"中性")
    processed_textset1, allclasstags1 = loadDataSet(path_name1, "正面")
    processed_textset2, allclasstags2 = loadDataSet(path_name2, "负面")

    processed_textset0=processed_textset+processed_textset1+processed_textset2
    allclasstags0=allclasstags+allclasstags1+allclasstags2

    predict_result=sklearnNB(processed_textset0,allclasstags0,testset)
    print("predict_result",predict_result)