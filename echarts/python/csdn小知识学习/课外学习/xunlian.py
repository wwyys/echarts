#coding=gbk
'''
encoding：utf_8
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
# 清洗不需要的数据
new_adv_data = adv_data.iloc[:, 1:]
#得到我们所需要的数据集且查看其前几列以及数据形状
print(new_adv_data.head())
#new_adv_data.drop(['Unnamed:0'],axis=1)  #'未命名 :0'列是多余的。所以，我们把这一列删除
print('head:', new_adv_data.head(), '\nShape:', new_adv_data.shape)
#电视广告费用和销售额散点图
plt.figure(figsize=(16, 8))
plt.scatter(new_adv_data['TV'], new_adv_data['Sales'], c='black')
plt.xlabel('Money spent on TV ads ($)')
plt.ylabel('Sales($)')
plt.savefig('mutil_linear_reg2_TV_Sales.jpg')
plt.show()
'''
从图中可以看到，电视广告费用和销售额明显相关
基于此数据，可以得出线性相似。对数据集给出一条拟合直线并查看等式参数就是这么简单。
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

#多元线性回归
#数据描述
print(new_adv_data.describe())
#缺失值检验
print(new_adv_data[new_adv_data.isnull() == True].count())
new_adv_data.boxplot()
plt.savefig('mutil_linear_reg2_boxplot.jpg')
plt.show()
print(new_adv_data.corr())
sns.pairplot(new_adv_data, x_vars=['TV', 'Radio', 'Newspaper'], y_vars='Sales', height=7, aspect=0.8, kind='reg')
plt.savefig('mutil_linear_reg2_pairplot.jpg')
plt.show()
'''
建模
和一元线性回归的建模过程一样，定义特征值和目标向量，并利用scikit-learn库来执行线性回归模型
得到如下等式：
The linear model is:Y = 2.9389 + 0.045765*TV + 0.18853*Radio + -0.0010375*Newpaper
'''
Xs = new_adv_data.drop(['Sales'], axis=1)  #, 'Unnamed' 去除Sales列
y = new_adv_data['Sales'].values.reshape(-1, 1)
reg = LinearRegression()
reg.fit(Xs, y)
print('The linear model is: Y = [{:.5}+{:.5}*Radio+{:.5}*TV+{:.5}*Newspaper'.format(reg.intercept_[0], reg.coef_[0][0], reg.coef_[0][1], reg.coef_[0][2]))
X = np.column_stack((new_adv_data['TV'], new_adv_data['Radio'], new_adv_data['Newspaper']))
y = new_adv_data['Sales']
X2 = sm.add_constant(X)
est = sm.OLS(y, X2)
est2 = est.fit()
print('R值，p值和F统计量\n', est2.summary())

#train_size表示训练集所占数据集的比例
#print('newP_adv_data.ix[:,:3]'new_adv_data.ix[:,:3])
#print('newP_adv_data.Sales'new_adv_data.Sales)
X_train, X_test, Y_train, Y_test = train_test_split(new_adv_data.iloc[:, :3], new_adv_data.Sales, train_size=.80)
print('原始数据特征', new_adv_data.iloc[:, :3].shape,
      ',训练数据特征:', X_train.shape,
      ',测试数据特征:', X_test.shape)
print('原始数据特征', new_adv_data.Sales.shape,
      ',训练标签特征:', Y_train.shape,
      ',测试标签特征:', Y_test.shape)
model = LinearRegression()
model.fit(X_train, Y_train)
a = model.intercept_  #截距
b = model.coef_  #回归系数
print('最佳拟合线:截距', a, ',回归系数:', b)
#y=2.668+0.0448*TV+0.187*Radio-0.00242*Newspaper
score = model.score(X_test, Y_test)
print(score)
#对线性回归进行预测
Y_pred = model.predict(X_test)
print(Y_pred)
plt.plot(range(len(Y_pred)), Y_pred, 'b', label='predict')
#显示图像
plt.savefig('mutil_linear_reg2_predict.jpg')
plt.show()
plt.figure()
plt.plot(range(len(Y_pred)), Y_pred, 'b', label='predict')
plt.plot(range(len(Y_pred)), Y_test, 'b', label='test')
plt.legend(loc='upper right')  #显示图中的标签
plt.xlabel('the number of sales')
plt.ylabel('value of sales')
plt.savefig('mutil_linear_reg2_ROC.jpg')
plt.show()
