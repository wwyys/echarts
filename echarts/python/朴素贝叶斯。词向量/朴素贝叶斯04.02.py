#coding="utf-8"
import numpy as np
from sklearn.naive_bayes import MultinomialNB
#X对应的是特征  文章的文本   0  1
X=np.random.randint(5,size=(6,100))
print("X=",X)

#y是类
y=np.array([1,2,3,4,5,6])
print("y=",y)

clf=MultinomialNB()
print("clf=",clf)
#训练多项式朴素贝叶斯   结果产生
clf.fit(X,y)
print("clf.fit(X,y)=",clf.fit(X,y))
print("X[2:3]=",X[2:3])
MultinomialNB()
print(clf.predict(X[2:3]))