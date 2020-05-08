import numpy as np
#将X，y赋值为np数组
#在y中0表示没有下雨，1表示要下雨
#在X中0表示否，1表示是
X=np.array([[0,1,0,1],
            [1,1,1,0],
            [0,1,1,0],
            [0,0,0,1],
            [0,1,1,0],
            [0,1,0,1],
            [1,0,0,1]])
y=np.array([0,1,1,0,1,0,0])
#对不同分类计算每个特征为1的数量
counts={ }
for label in np.unique(y):
    counts[label]=X[y==label].sum(axis=0)
#打印计数结果
print("feature counts:\n{}".format(counts))


#测试数据
#导入贝努利贝叶斯
from sklearn.naive_bayes import BernoulliNB
#使用贝努利贝叶斯拟合数据
clf=BernoulliNB()
clf.fit(X,y)
#要进行预测的这一天，没有刮北风，也不闷热
Next_Day=[[1,0,1,0]]
pre=clf.predict(Next_Day)
print('\n\n\n')
print("代码运行结果:")
print("=======================\n")
if pre ==[1]:
    print("要下雨啦,快收衣服啊!")
else:
    print("放心,又是一个艳阳天!")
print("\n=======================")
print("\n\n\n")

#假设另外一天的数据如下
Another_day=[[1,1,1,1]]
#使用训练好的模型进行预测
pre2=clf.predict(Another_day)
print('\n\n\n')
print('代码运行结果:')
print("=======================\n")
if pre2==[1]:
    print("要下雨啦,快收衣服啊!")
else:
    print("放心，又是一个艳阳天!")
print("\n=======================")
print('\n\n\n')

print('\n\n\n')
print('代码运行结果:')
print('========================\n')
#模型预测分类的概率
print(clf.predict_proba(Next_Day))
print('\n=========================')
print('\n\n\n')