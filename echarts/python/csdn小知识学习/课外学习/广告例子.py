#coding=gbk
#导入数据包

import pandas as pd                                            #利用pandas数据进行读取与操作
import numpy as np                                             #利用numpy对数据进行操作
import matplotlib.pyplot as plt                                #利用pyplot对数据进行可视化与图像化
from pandas import DataFrame,Series
from sklearn.model_selection import train_test_split           #利用train_test_split将样本分为测试集与试验集
from sklearn.neighbors import KNeighborsRegressor              #使用sklearn进行数据集训练与模型导入



#通过read_csv来读取我们的目的数据集
adv_data = pd.read_csv("./Advertising.csv",encoding="ISO-8859-1")   #编码类型可能为encoding="ISO-8859-1"
new_adv_data = adv_data.iloc[:,:]                                     #数据可能进行剪切
'''
print(new_adv_data.describe())
print(new_adv_data.shape)
import seaborn as sns
##相关系数矩阵 r(相关系数) = x和y的协方差/(x的标准差*y的标准差) == cov（x,y）/σx*σy
# 相关系数0~0.3弱相关0.3~0.6中等程度相关0.6~1强相关
print(new_adv_data.corr())
# 通过加入一个参数kind='reg'，seaborn可以添加一条最佳拟合直线和95%的置信带。
sns.pairplot(new_adv_data, x_vars=['TV','radio','newspaper'], y_vars='Sales', size=7, aspect=0.8,kind = 'reg')
plt.show()
'''
X_train, X_test, Y_train, Y_test = train_test_split(adv_data.iloc[:, :3], adv_data.Sales,train_size=.80)



k = 5
knn = KNeighborsRegressor(k)
knn.fit(X_train, Y_train)
# 生成足够密集的点并进行预测
score = knn.score(X_test, Y_test)
print("score:",score)
Y_pred = knn.predict(X_test)
plt.figure()
plt.plot(range(len(Y_pred)),Y_pred,'b',label="predict")
plt.plot(range(len(Y_pred)),Y_test,'r',label="test")
plt.legend(loc="upper right") #显示图中的标签
plt.xlabel("the number of sales")
plt.ylabel('value of sales')
plt.savefig("ROC.jpg")
plt.show()


