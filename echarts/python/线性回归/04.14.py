# -*- coding: UTF-8 -*-
from pandas import DataFrame,Series
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
examDict = {'学习时间': [0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 1.75,

                     2.00, 2.25, 2.50, 2.75, 3.00, 3.25, 3.50, 4.00, 4.25, 4.50, 4.75, 5.00, 5.50],

            '分数': [10, 22, 13, 43, 20, 22, 33, 50, 62,

                   48, 55, 75, 62, 73, 81, 76, 64, 82, 90, 93]}

#数据格式的转换
examDF=DataFrame(examDict)
#画图  为了我们能够更好地选择模型
plt.scatter(examDF.分数,examDF.学习时间,color='b',label="Exam Data")
# y x
plt.xlabel("Hours")
plt.ylabel("Score")
#显示图像
plt.savefig("Mlina_reg1_demo.jpg")
plt.show()
#回归方程  模型   y=ax+b
#误差  =  实际值-预测值（拟合值）
#误差的平方  sse==∑（实际值-预测值）^2
#最下二乘法：使得误差平方和最小（最佳拟合）
exam_X=examDF.loc[:,'学习时间']
exam_Y=examDF.loc[:,'分数']
#将原来的数据集划分为训练集和测试集
X_train,X_test,Y_train,Y_test=train_test_split(exam_X,exam_Y,train_size=0.8)
print("原始数据特征",exam_X.shape,
      "训练数据特征",X_train.shape,
      "测试数据特征",X_test.shape)
print("原始标签特征",exam_Y.shape,
      "训练标签特征",Y_train.shape,
      "测试标签特征",Y_test.shape)
#画图  为了我们能够更好地选择模型
plt.scatter(X_train,Y_train,color='yellow',label="train Data")
plt.scatter(X_test,Y_test,color='red',label="test Data")
# y x
plt.legend(loc=2)
plt.xlabel("Hours")
plt.ylabel("Score")
#显示图像
plt.savefig("mutil_linear_reg1_tests.jpg")
plt.show()
#回归方程  模型   y=ax+b
model=LinearRegression()
#误差  =  实际值-预测值（拟合值）
#误差的平方  sse==∑（实际值-预测值）^2
#最下二乘法：使得误差平方和最小（最佳拟合）
X_train=X_train.values.reshape(-1,1)
X_test=X_test.values.reshape(-1,1)
model.fit(X_train,Y_train)
a=model.intercept_ #截距
b=model.coef_ #回归系数
print("最佳拟合线:截距",a,",回归系数:",b)
#测试
y_train_pred=model.predict(X_train)
plt.scatter(X_train,y_train_pred,color='red',linewidths=3,label="line")
#测试数据散点图
plt.scatter(X_test,Y_test,color='red',label="test Data")
# 添加图标标签y x
plt.legend(loc=2)
plt.xlabel("Hours")
plt.ylabel("Score")
plt.show()
score=model.score(X_test,Y_test)
print(score)
