#coding=gbk
#�������ݰ�

import pandas as pd                                            #����pandas���ݽ��ж�ȡ�����
import numpy as np                                             #����numpy�����ݽ��в���
import matplotlib.pyplot as plt                                #����pyplot�����ݽ��п��ӻ���ͼ��
from pandas import DataFrame,Series
from sklearn.model_selection import train_test_split           #����train_test_split��������Ϊ���Լ������鼯
from sklearn.neighbors import KNeighborsRegressor              #ʹ��sklearn�������ݼ�ѵ����ģ�͵���



#ͨ��read_csv����ȡ���ǵ�Ŀ�����ݼ�
adv_data = pd.read_csv("./Advertising.csv",encoding="ISO-8859-1")   #�������Ϳ���Ϊencoding="ISO-8859-1"
new_adv_data = adv_data.iloc[:,:]                                     #���ݿ��ܽ��м���
'''
print(new_adv_data.describe())
print(new_adv_data.shape)
import seaborn as sns
##���ϵ������ r(���ϵ��) = x��y��Э����/(x�ı�׼��*y�ı�׼��) == cov��x,y��/��x*��y
# ���ϵ��0~0.3�����0.3~0.6�еȳ̶����0.6~1ǿ���
print(new_adv_data.corr())
# ͨ������һ������kind='reg'��seaborn�������һ��������ֱ�ߺ�95%�����Ŵ���
sns.pairplot(new_adv_data, x_vars=['TV','radio','newspaper'], y_vars='Sales', size=7, aspect=0.8,kind = 'reg')
plt.show()
'''
X_train, X_test, Y_train, Y_test = train_test_split(adv_data.iloc[:, :3], adv_data.Sales,train_size=.80)



k = 5
knn = KNeighborsRegressor(k)
knn.fit(X_train, Y_train)
# �����㹻�ܼ��ĵ㲢����Ԥ��
score = knn.score(X_test, Y_test)
print("score:",score)
Y_pred = knn.predict(X_test)
plt.figure()
plt.plot(range(len(Y_pred)),Y_pred,'b',label="predict")
plt.plot(range(len(Y_pred)),Y_test,'r',label="test")
plt.legend(loc="upper right") #��ʾͼ�еı�ǩ
plt.xlabel("the number of sales")
plt.ylabel('value of sales')
plt.savefig("ROC.jpg")
plt.show()


