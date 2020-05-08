#coding=gbk
'''
encoding��utf_8
author:wy
time:2020.04.14
'''
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
adv_data=pd.read_csv('./Advertising.csv')
# ��ϴ����Ҫ������
new_adv_data = adv_data.iloc[:, 1:]
#�õ���������Ҫ�����ݼ��Ҳ鿴��ǰ�����Լ�������״
print(new_adv_data.head())
#new_adv_data.drop(['Unnamed:0'],axis=1)  #'δ���� :0'���Ƕ���ġ����ԣ����ǰ���һ��ɾ��
print('head:', new_adv_data.head(), '\nShape:', new_adv_data.shape)
#���ӹ����ú����۶�ɢ��ͼ
plt.figure(figsize=(16, 8))
plt.scatter(new_adv_data['TV'], new_adv_data['Sales'], c='black')
plt.xlabel('Money spent on TV ads ($)')
plt.ylabel('Sales($)')
plt.savefig('mutil_linear_reg2_TV_Sales.jpg')
plt.show()
'''
��ͼ�п��Կ��������ӹ����ú����۶��������
���ڴ����ݣ����Եó��������ơ������ݼ�����һ�����ֱ�߲��鿴��ʽ����������ô�򵥡�
The linear model is Y =7.0326+0.047537X
'''
X = new_adv_data['TV'].values.reshape(-1, 1)
y = new_adv_data['Sales'].values.reshape(-1, 1)
reg = LinearRegression()
reg.fit(X, y)
print('The linear model is: Y = [{:.5}+{:.5}X'.format(reg.intercept_[0], reg.coef_[0][0]))
predictions = reg.predict(X)
plt.figure(figsize=(16, 8))
plt.scatter(new_adv_data['TV'], new_adv_data['Sales'], c='black')
plt.plot(new_adv_data['TV'], predictions, c='blue', linewidth=2)
plt.xlabel('Money spent on TV ads ($)')
plt.ylabel('Sales($)')
plt.savefig('mutil_linear_reg2_TV_Sales.jpg')
plt.show()
X = new_adv_data['TV']
y = new_adv_data['Sales']
X2 = sm.add_constant(X)
est = sm.OLS(y, X2)
est2 = est.fit()
print(est2.summary())

#��Ԫ���Իع�
#��������
print(new_adv_data.describe())
#ȱʧֵ����
print(new_adv_data[new_adv_data.isnull() == True].count())
new_adv_data.boxplot()
plt.savefig('mutil_linear_reg2_boxplot.jpg')
plt.show()
print(new_adv_data.corr())
sns.pairplot(new_adv_data, x_vars=['TV', 'Radio', 'Newspaper'], y_vars='Sales', height=7, aspect=0.8, kind='reg')
plt.savefig('mutil_linear_reg2_pairplot.jpg')
plt.show()
'''
��ģ
��һԪ���Իع�Ľ�ģ����һ������������ֵ��Ŀ��������������scikit-learn����ִ�����Իع�ģ��
�õ����µ�ʽ��
The linear model is:Y = 2.9389 + 0.045765*TV + 0.18853*Radio + -0.0010375*Newpaper
'''
Xs = new_adv_data.drop(['Sales'], axis=1)  #, 'Unnamed' ȥ��Sales��
y = new_adv_data['Sales'].values.reshape(-1, 1)
reg = LinearRegression()
reg.fit(Xs, y)
print('The linear model is: Y = [{:.5}+{:.5}*Radio+{:.5}*TV+{:.5}*Newspaper'.format(reg.intercept_[0], reg.coef_[0][0], reg.coef_[0][1], reg.coef_[0][2]))
X = np.column_stack((new_adv_data['TV'], new_adv_data['Radio'], new_adv_data['Newspaper']))
y = new_adv_data['Sales']
X2 = sm.add_constant(X)
est = sm.OLS(y, X2)
est2 = est.fit()
print('Rֵ��pֵ��Fͳ����\n', est2.summary())

#train_size��ʾѵ������ռ���ݼ��ı���
#print('newP_adv_data.ix[:,:3]'new_adv_data.ix[:,:3])
#print('newP_adv_data.Sales'new_adv_data.Sales)
X_train, X_test, Y_train, Y_test = train_test_split(new_adv_data.iloc[:, :3], new_adv_data.Sales, train_size=.80)
print('ԭʼ��������', new_adv_data.iloc[:, :3].shape,
      ',ѵ����������:', X_train.shape,
      ',������������:', X_test.shape)
print('ԭʼ��������', new_adv_data.Sales.shape,
      ',ѵ����ǩ����:', Y_train.shape,
      ',���Ա�ǩ����:', Y_test.shape)
model = LinearRegression()
model.fit(X_train, Y_train)
a = model.intercept_  #�ؾ�
b = model.coef_  #�ع�ϵ��
print('��������:�ؾ�', a, ',�ع�ϵ��:', b)
#y=2.668+0.0448*TV+0.187*Radio-0.00242*Newspaper
score = model.score(X_test, Y_test)
print(score)
#�����Իع����Ԥ��
Y_pred = model.predict(X_test)
print(Y_pred)
plt.plot(range(len(Y_pred)), Y_pred, 'b', label='predict')
#��ʾͼ��
plt.savefig('mutil_linear_reg2_predict.jpg')
plt.show()
plt.figure()
plt.plot(range(len(Y_pred)), Y_pred, 'b', label='predict')
plt.plot(range(len(Y_pred)), Y_test, 'b', label='test')
plt.legend(loc='upper right')  #��ʾͼ�еı�ǩ
plt.xlabel('the number of sales')
plt.ylabel('value of sales')
plt.savefig('mutil_linear_reg2_ROC.jpg')
plt.show()
