#coding="utf-8"
import os
import jieba
import numpy
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
"""
1.数据集我们已经有了...无...爬虫
2.构建NNB()形参数据类型
3.fit()训练
4.predict（X）未知的X预测
"""
"""
功能：数据集我们已经有了，读取数据。构建特征和类
Input：有文件夹--文件路径
output：特征  类
"""
def loadDataSet(path_name,classtag):
    allfiles = os.listdir((path_name))
    processed_textset=[]
    allclasstags=[]
    for thisfile in allfiles:
        path_name="./data/中性"+"/"+thisfile #获得文件名
        print("path_name",path_name)
    #打开文件：
        textfile = open(path_name, "r", encoding='utf-8').read()
        print("textfile",textfile)
        #词向量   一个个的词构成
        textcut = jieba.cut(textfile)
        print("textcut", textcut)
        text_with_spaces = ""
        for word in textcut:
            text_with_spaces += word + ""
            print("text_with_spaces", text_with_spaces)
        processed_textset.append(text_with_spaces)
        print("processed_textset", processed_textset)
        allclasstags.append(classtag)
    return processed_textset, allclasstags


if __name__ == '__main__':
    path_name = "data/中性"
    loadDataSet(path_name,"中性")